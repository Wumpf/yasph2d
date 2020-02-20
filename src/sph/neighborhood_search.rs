use crate::units::*;

pub type ParticleIndex = u32;
pub type CellIndex = u32;

#[derive(Copy, Clone)]
struct Particle {
    pidx: ParticleIndex,
    cidx: CellIndex,
}

#[derive(Copy, Clone)]
struct CellPos {
    x: u16,
    y: u16,
}
impl CellPos {
    #[inline]
    fn to_cidx(self) -> CellIndex {
        super::morton::encode(self.x, self.y)
    }
}

#[derive(Copy, Clone)]
struct Cell {
    first_particle: ParticleIndex,
    cidx: CellIndex,
}

struct GridProperties {
    radius: Real,
    cell_size_inv: Real,
    grid_min: Point,
}
impl GridProperties {
    #[inline]
    fn position_to_cellpos(&self, position: Point) -> CellPos {
        let cellspace = (position - self.grid_min) * self.cell_size_inv;
        CellPos {
            x: cellspace.x as u16,
            y: cellspace.y as u16,
        }
    }

    #[inline]
    fn position_to_cidx(&self, position: Point) -> CellIndex {
        self.position_to_cellpos(position).to_cidx()
    }
}

#[derive(Default)]
struct PointSet {
    particles: Vec<Particle>,
    cells: Vec<Cell>,
}

impl PointSet {
    pub fn update(&mut self, grid: &GridProperties, positions: &[Point]) {
        // Adjust size. (not paralized since expected to be small)
        match self.particles.len().cmp(&positions.len()) {
            std::cmp::Ordering::Greater => {
                unimplemented!("Removing particles not impemented yet");
            }
            std::cmp::Ordering::Less => {
                self.particles.reserve(positions.len());
                for new_pidx in self.particles.len()..positions.len() {
                    self.particles.push(Particle {
                        pidx: new_pidx as ParticleIndex,
                        cidx: 0,
                    }); // todo: compute added particles on the spot and leave out later?
                }
            }
            std::cmp::Ordering::Equal => (),
        }

        // Update cell indices. Todo: Parallize
        for p in self.particles.iter_mut() {
            p.cidx = grid.position_to_cidx(positions[p.pidx as usize]);
        }

        // Sort by cell index. Todo: Parallize
        self.particles.sort_unstable_by_key(|a| a.cidx);

        // create cells. Todo: Parallize by doing a prefix sum first
        self.cells.clear();
        let mut prev_cidx = CellIndex::max_value();
        for (pidx, p) in self.particles.iter().enumerate() {
            if p.cidx != prev_cidx {
                self.cells.push(Cell {
                    first_particle: pidx as ParticleIndex,
                    cidx: p.cidx,
                });
                prev_cidx = p.cidx;
            }
        }
        self.cells.push(Cell {
            first_particle: self.particles.len() as ParticleIndex,
            cidx: CellIndex::max_value(),
        }); // sentinel cell
    }

    // finds cell array index first cell that has an equal or bigger for a given CellIndex
    fn find_next_cell(cells: &[Cell], cidx: CellIndex) -> usize {
        const LINEAR_SEARCH_THRESHHOLD: usize = 16;
        let mut min = 0;
        let mut max = cells.len(); // exclusive
        let mut range = max - min;
        while range > LINEAR_SEARCH_THRESHHOLD {
            range /= 2;
            let mid = min + range;
            match unsafe { cells.get_unchecked(mid) }.cidx.cmp(&cidx) {
                std::cmp::Ordering::Greater => max = mid,
                std::cmp::Ordering::Less => min = mid,
                std::cmp::Ordering::Equal => return mid,
            }
        }
        for pos in min..max {
            if unsafe { cells.get_unchecked(pos) }.cidx >= cidx {
                return pos;
            }
        }
        max
    }

    pub fn foreach_potential_neighbor(&self, grid: &GridProperties, position: Point, mut f: impl FnMut(ParticleIndex) -> ()) {
        let cidx_min = grid.position_to_cidx(position - Vector::new(grid.radius, grid.radius));
        let cidx_min_xbits = cidx_min & super::morton::MORTON_XBITS;
        let cidx_min_ybits = cidx_min & super::morton::MORTON_YBITS;
        let cidx_max = grid.position_to_cidx(position + Vector::new(grid.radius, grid.radius));
        let cidx_max_xbits = cidx_max & super::morton::MORTON_XBITS;
        let cidx_max_ybits = cidx_max & super::morton::MORTON_YBITS;

        const MAX_CONSECUTIVE_CELL_MISSES: u32 = 8;

        // Note: Already tried doing this with iterators. it's hard to do and slow!
        let mut cell_arrayidx = Self::find_next_cell(&self.cells, cidx_min);
        let mut cell = self.cells[cell_arrayidx];

        while cell.cidx <= cidx_max {
            // skip until cell is in rect
            let mut num_misses = 0;
            while !super::morton::is_in_rect_presplit(cell.cidx, cidx_min_xbits, cidx_min_ybits, cidx_max_xbits, cidx_max_ybits) {
                num_misses += 1;

                // Try next. Prefer to just grind the array, but at some point use bigmin to jump ahead.
                if num_misses > MAX_CONSECUTIVE_CELL_MISSES {
                    let expected_next_cidx = super::morton::find_bigmin(cell.cidx, cidx_min, cidx_max);
                    cell_arrayidx += Self::find_next_cell(&self.cells[cell_arrayidx..], expected_next_cidx);
                    assert!(expected_next_cidx > cell.cidx);
                } else {
                    cell_arrayidx += 1;
                }
                cell = self.cells[cell_arrayidx];

                if cell.cidx > cidx_max {
                    return;
                }
            }

            // find particle range
            let first_particle = cell.first_particle;
            loop {
                cell_arrayidx += 1; // we won't be here for long, no point in doing profound skipping.
                cell = self.cells[cell_arrayidx];
                if !super::morton::is_in_rect_presplit(cell.cidx, cidx_min_xbits, cidx_min_ybits, cidx_max_xbits, cidx_max_ybits) {
                    break;
                }
            }
            let last_particle = cell.first_particle;
            assert_ne!(cell.cidx, cidx_max); // it if was equal, then there would be a cell at cidx_max that is not in the rect limited by cidx_max

            // Consume particles.
            for p in first_particle..last_particle {
                f(self.particles[p as usize].pidx);
            }

            // We know current cell isn't in the rect, so skip it.
            cell_arrayidx += 1;
            if cell_arrayidx >= self.cells.len() {
                break;
            }
            cell = self.cells[cell_arrayidx];
        }
    }
}

pub struct NeighborhoodSearch {
    grid: GridProperties,

    dynamic_particles: PointSet,
    boundary_particles: PointSet,
}

impl NeighborhoodSearch {
    /// * radius:               Radius that determines if a point is a neighbor
    /// * expected_max_density: Num particles expected per square unit
    pub fn new(radius: Real, //    , expected_max_density: Real
    ) -> NeighborhoodSearch {
        let cell_size = radius * 2.0; // todo: experiment with larger cells

        //const INDICES_PER_CACHELINE: u32 = 64 / std::mem::size_of::<ParticleIndex>() as u32;
        //let mut num_expected_in_cell = (cell_size * cell_size * expected_max_density + 0.5) as u32;
        //num_expected_in_cell = (num_expected_in_cell + INDICES_PER_CACHELINE-1) / INDICES_PER_CACHELINE * INDICES_PER_CACHELINE;

        NeighborhoodSearch {
            grid: GridProperties {
                radius,
                cell_size_inv: 1.0 / cell_size,
                // todo: we can create a huge domain, but still there is a limited domain! should be safe about this and have a max
                // limit is there because our morton curve wraps around at some point and then things get complicated (aka don't want to deal with this!)
                grid_min: Point::new(-100.0, -100.0),
            },
            dynamic_particles: Default::default(),
            boundary_particles: Default::default(),
        }
    }

    pub fn update_boundary(&mut self, positions: &[Point]) {
        microprofile::scope!("NeighborhoodSearch", "update_boundary");
        self.boundary_particles.update(&self.grid, positions);
    }

    pub fn update(&mut self, positions: &[Point]) {
        microprofile::scope!("NeighborhoodSearch", "update");
        self.dynamic_particles.update(&self.grid, positions);
    }

    pub fn foreach_potential_neighbor(&self, position: Point, f: impl FnMut(ParticleIndex) -> ()) {
        self.dynamic_particles.foreach_potential_neighbor(&self.grid, position, f)
    }

    pub fn foreach_potential_boundary_neighbor(&self, position: Point, f: impl FnMut(ParticleIndex) -> ()) {
        self.boundary_particles.foreach_potential_neighbor(&self.grid, position, f)
    }
}
