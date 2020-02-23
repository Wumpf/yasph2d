use crate::units::*;

use super::scratch_buffer::ScratchBufferStore;

pub type ParticleIndex = u32;
pub type CellIndex = u32;

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
    first_particle: usize,
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
    cell_indices: Vec<CellIndex>,
    particle_indices: Vec<ParticleIndex>, // TODO: Remove, make this temp
    cells: Vec<Cell>,
}

impl PointSet {
    fn apply_sorting<T: Copy>(sorting: &[ParticleIndex], scratch_buffer: &mut Vec<T>, buffer_to_sort: &mut Vec<T>) {
        assert_eq!(scratch_buffer.len(), buffer_to_sort.len());
        assert_eq!(sorting.len(), buffer_to_sort.len());
        for (pos, &i) in scratch_buffer.iter_mut().zip(sorting.iter()) {
            *pos = buffer_to_sort[i as usize];
        }
        std::mem::swap(scratch_buffer, buffer_to_sort);
    }

    // note: Applying sorting is a bit costly and only solvers know which attributes are discarded/recomputed and which need the new sorting applied.
    // todo: parallize things
    pub fn update(
        &mut self,
        scratch_buffers: &mut ScratchBufferStore,
        grid: &GridProperties,
        positions: &mut Vec<Point>,
        particle_attributes_vector: &mut Vec<&mut Vec<Vector>>,
        particle_attributes_real: &mut Vec<&mut Vec<Real>>,
    ) {
        // we know that most particles have not changed since last frame
        // -> use insertion sort and building permutation array into it as well!
        // (benchmarking confirmed that this is a lot faster than particles.sort_unstable_by_key)
        // ... just a bit harder to parallize this way.
        self.cell_indices.resize(positions.len(), 0);
        self.particle_indices.resize(positions.len(), 0);
        self.cells.clear();

        //let mut num_swaps = 0;
        for (i, &pos) in positions.iter().enumerate() {
            let cidx = grid.position_to_cidx(pos);

            self.particle_indices[i] = i as ParticleIndex;
            self.cell_indices[i] = cidx;

            for j in (0..i).rev() {
                if self.cell_indices[j] > self.cell_indices[j + 1] {
                    self.cell_indices.swap(j, j + 1);
                    self.particle_indices.swap(j, j + 1);
                } else {
                    break;
                }
            }

            // Version with memcpy instead of swaps (significantly slower in benchmarks - probably not compiler friendly)
            // let mut j = i;
            // while j > 0 && self.cell_indices[j - 1] > cidx {
            //     j -= 1;
            // }
            // if i != j {
            //     unsafe {
            //         core::ptr::copy(self.cell_indices.as_ptr().add(j), self.cell_indices.as_mut_ptr().add(j + 1), i - j);
            //     }
            //     unsafe {
            //         core::ptr::copy(
            //             self.particle_indices.as_ptr().add(j),
            //             self.particle_indices.as_mut_ptr().add(j + 1),
            //             i - j,
            //         );
            //     }
            //     // safe version:
            //     self.cell_indices.copy_within(j..i, j + 1);
            //     //self.particle_indices.copy_within(j..i, j + 1);
            // }
            // self.cell_indices[j] = cidx;
            // self.particle_indices[j] = i as ParticleIndex;
        }
        //println!("num swaps {}", num_swaps);

        self.cells.push(Cell {
            first_particle: positions.len(),
            cidx: CellIndex::max_value(),
        }); // sentinel cell

        // Apply sorting.
        {
            microprofile::scope!("NeighborhoodSearch", "apply sorting");
            {
                let mut scratch_buffer = scratch_buffers.get_buffer_point(positions.len());
                Self::apply_sorting(&self.particle_indices, &mut scratch_buffer.buffer, positions);
            }
            {
                let mut scratch_buffer = scratch_buffers.get_buffer_vector(positions.len());
                for attribute_buffer in particle_attributes_vector.iter_mut() {
                    Self::apply_sorting(&self.particle_indices, &mut scratch_buffer.buffer, *attribute_buffer);
                }
            }
            {
                let mut scratch_buffer = scratch_buffers.get_buffer_real(positions.len());
                for attribute_buffer in particle_attributes_real.iter_mut() {
                    Self::apply_sorting(&self.particle_indices, &mut scratch_buffer.buffer, *attribute_buffer);
                }
            }
        }

        // create cells.
        // we could do this during the sort and use the prefix sums for some clever jumping. Tried it and wasn't great (both perf & impl niceness)
        self.cells.clear();
        let mut prev_cidx = CellIndex::max_value();
        for (pidx, &cidx) in self.cell_indices.iter().enumerate() {
            if cidx != prev_cidx {
                self.cells.push(Cell { first_particle: pidx, cidx });
                prev_cidx = cidx;
            }
        }
        self.cells.push(Cell {
            first_particle: positions.len(),
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
                f(p as ParticleIndex);
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

    // todo: allow boundaries to have properties
    pub fn update_boundary(&mut self, scratch_buffers: &mut ScratchBufferStore, positions: &mut Vec<Point>) {
        microprofile::scope!("NeighborhoodSearch", "update_boundary");
        self.boundary_particles
            .update(scratch_buffers, &self.grid, positions, &mut Vec::new(), &mut Vec::new());
    }

    pub fn update(
        &mut self,
        scratch_buffers: &mut ScratchBufferStore,
        positions: &mut Vec<Point>,
        particle_attributes_vector: &mut Vec<&mut Vec<Vector>>,
        particle_attributes_real: &mut Vec<&mut Vec<Real>>,
    ) {
        microprofile::scope!("NeighborhoodSearch", "update");
        self.dynamic_particles.update(
            scratch_buffers,
            &self.grid,
            positions,
            particle_attributes_vector,
            particle_attributes_real,
        );
    }

    pub fn foreach_potential_neighbor(&self, position: Point, f: impl FnMut(ParticleIndex) -> ()) {
        self.dynamic_particles.foreach_potential_neighbor(&self.grid, position, f)
    }

    pub fn foreach_potential_boundary_neighbor(&self, position: Point, f: impl FnMut(ParticleIndex) -> ()) {
        self.boundary_particles.foreach_potential_neighbor(&self.grid, position, f)
    }
}
