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
    x: CellIndex,
    y: CellIndex,
}

#[derive(Copy, Clone)]
struct Cell {
    first_particle: ParticleIndex,
    cidx: CellIndex,
}

pub struct NeighborhoodSearch {
    radius: Real,
    cell_size_inv: Real,

    particles: Vec<Particle>,
    cells: Vec<Cell>,

    grid_min: Point,
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
            radius,
            cell_size_inv: 1.0 / cell_size,
            particles: Vec::new(),
            cells: Vec::new(),

            grid_min: Point::new(-100.0, -100.0),
        }
    }

    #[inline]
    fn position_to_cellpos(grid_min: Point, cell_size_inv: Real, position: Point) -> CellPos {
        let cellspace = (position - grid_min) * cell_size_inv;
        CellPos {
            x: cellspace.x as CellIndex,
            y: cellspace.y as CellIndex,
        }
    }

    #[inline]
    fn cellpos_to_cidx(cellpos: CellPos) -> CellIndex {
        super::morton::encode(cellpos.x, cellpos.y)
    }

    #[inline]
    fn position_to_cidx(grid_min: Point, cell_size_inv: Real, position: Point) -> CellIndex {
        Self::cellpos_to_cidx(Self::position_to_cellpos(grid_min, cell_size_inv, position))
    }

    pub fn update(&mut self, positions: &[Point]) {
        // Adjust size. (not paralized since expected to be small)
        if self.particles.len() < positions.len() {
            self.particles.reserve(positions.len());
            for new_pidx in self.particles.len()..positions.len() {
                self.particles.push(Particle {
                    pidx: new_pidx as ParticleIndex,
                    cidx: 0,
                }); // todo: compute added particles on the spot and leave out later?
            }
        } else if self.particles.len() > positions.len() {
            unimplemented!("Removing particles not impemented yet");
        }

        // Update cell indices. Todo: Parallize
        for p in self.particles.iter_mut() {
            p.cidx = Self::position_to_cidx(self.grid_min, self.cell_size_inv, positions[p.pidx as usize]);
        }

        // Sort by cell index. Todo: Parallize
        self.particles.sort_by_key(|a| a.cidx);

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
    fn search_cell(cells: &[Cell], cidx: CellIndex) -> usize {
        let mut min = 0;
        let mut max = cells.len();
        const LINEAR_SEARCH_THRESHHOLD: usize = 16;

        loop {
            let range = max - min;
            if range <= LINEAR_SEARCH_THRESHHOLD {
                //for (pos, cell) in cells.iter().enumerate().take(max).skip(min) {
                for pos in min..max {
                    if cells[pos].cidx >= cidx {
                        return pos;
                    }
                }
                return max;
            }
            let mid = min + range / 2;
            if cells[mid].cidx == cidx {
                return mid;
            } else if cells[mid].cidx < cidx {
                min = mid;
            } else {
                max = mid;
            }
        }
    }

    pub fn foreach_potential_neighbor(&self, position: Point, mut f: impl FnMut(usize) -> ()) {
        const XBITS: u32 = 0b01010101_01010101_01010101_01010101;
        const YBITS: u32 = 0b10101010_10101010_10101010_10101010;

        let cellpos_min = Self::position_to_cellpos(self.grid_min, self.cell_size_inv, position - Vector::new(self.radius, self.radius));
        let cidx_min_xbits = super::morton::part_1by1(cellpos_min.x);
        let cidx_min_ybits = super::morton::part_1by1(cellpos_min.y) << 1;
        let cidx_min = cidx_min_xbits | cidx_min_ybits;

        let cellpos_max = Self::position_to_cellpos(self.grid_min, self.cell_size_inv, position + Vector::new(self.radius, self.radius));
        let cidx_max_xbits = super::morton::part_1by1(cellpos_max.x);
        let cidx_max_ybits = super::morton::part_1by1(cellpos_max.y) << 1;
        let cidx_max = cidx_max_xbits | cidx_max_ybits;

        // todo use
        //const MAX_CONSECUTIVE_MISSES: u32 = 3;

        let first_cell = Self::search_cell(&self.cells, cidx_min);
        let mut num_misses = 0;
        let mut expected_next = 0;
        for i in first_cell..(self.cells.len() - 1) {
            let cell = self.cells[i];
            if cell.cidx > cidx_max {
                break;
            }

            let cidx_xbits = cell.cidx & XBITS;
            let cidx_ybits = cell.cidx & YBITS;

            if cidx_xbits < cidx_min_xbits || cidx_xbits > cidx_max_xbits || cidx_ybits < cidx_min_ybits || cidx_ybits > cidx_max_ybits {
                assert!(expected_next <= super::morton::find_bigmin(cell.cidx, cidx_min, cidx_max));
                num_misses += 1;
                expected_next = super::morton::find_bigmin(cell.cidx, cidx_min, cidx_max);
                continue;
            }

            assert!(num_misses == 0 || cell.cidx >= expected_next);
            num_misses = 0;

            for p in cell.first_particle..self.cells[i + 1].first_particle {
                f(self.particles[p as usize].pidx as usize); // todo this integer casting thing is getting out of hand
            }
        }
    }
}
