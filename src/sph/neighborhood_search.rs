use crate::units::*;

pub type ParticleIndex = u32;
pub type CellIndex = u32;

#[derive(Copy, Clone)]
struct Particle {
    pidx: ParticleIndex,
    cidx: CellIndex,
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

    #[inline(always)]
    fn position_to_cell(grid_min: Point, cell_size_inv: Real, position: Point) -> (CellIndex, CellIndex) {
        let cellspace = (position - grid_min) * cell_size_inv;
        (cellspace.x as CellIndex, cellspace.y as CellIndex)
    }

    #[inline(always)]
    fn position_to_cidx(grid_min: Point, cell_size_inv: Real, position: Point) -> CellIndex {
        let cell = Self::position_to_cell(grid_min, cell_size_inv, position);
        super::morton::encode(cell.0, cell.1)
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
        let cidx_min = Self::position_to_cidx(self.grid_min, self.cell_size_inv, position - Vector::new(self.radius, self.radius));
        let cidx_max = Self::position_to_cidx(self.grid_min, self.cell_size_inv, position + Vector::new(self.radius, self.radius));

        // todo use
        // LITMAX/BIGMIN algorithm
        // http://hermanntropf.de/media/multidimensionalrangequery.pdf

        let first_cell = Self::search_cell(&self.cells, cidx_min);
        for i in first_cell..(self.cells.len() - 1) {
            let cell = self.cells[i];
            if cell.cidx > cidx_max {
                break;
            }
            for p in cell.first_particle..self.cells[i + 1].first_particle {
                f(self.particles[p as usize].pidx as usize); // todo this integer casting thing is getting out of hand
            }
        }
    }
}
