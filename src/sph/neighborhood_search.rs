use super::scratch_buffer::ScratchBufferStore;
use crate::units::*;

use cgmath::prelude::*;

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

    #[inline]
    fn from_cidx(cidx: CellIndex) -> CellPos {
        CellPos {
            x: super::morton::decode_x(cidx) as u16,
            y: super::morton::decode_y(cidx) as u16,
        }
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
struct CellGrid {
    cells: Vec<Cell>,
}

impl CellGrid {
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
        particle_attributes_vector: &mut [&mut Vec<Vector>],
        particle_attributes_real: &mut [&mut Vec<Real>],
    ) {
        microprofile::scope!("NeighborhoodSearch", "CellGrid::update");

        let particle_indices = &mut scratch_buffers.get_buffer_uint(positions.len());
        let cell_indices = &mut scratch_buffers.get_buffer_uint(positions.len());

        // we know that most particles have not changed since last frame
        // -> use insertion sort and building permutation array into it as well!
        // (benchmarking confirmed that this is a lot faster than particles.sort_unstable_by_key)
        // ... just a bit harder to parallize this way.
        for (i, &pos) in positions.iter().enumerate() {
            let cidx = grid.position_to_cidx(pos);

            particle_indices.buffer[i] = i as ParticleIndex;
            cell_indices.buffer[i] = cidx;

            for j in (0..i).rev() {
                if cell_indices.buffer[j] > cell_indices.buffer[j + 1] {
                    cell_indices.buffer.swap(j, j + 1);
                    particle_indices.buffer.swap(j, j + 1);
                } else {
                    break;
                }
            }

            // Version with memcpy instead of swaps (significantly slower in benchmarks - probably not compiler friendly)
            // let mut j = i;
            // while j > 0 && cell_indices.buffer[j - 1] > cidx {
            //     j -= 1;
            // }
            // if i != j {
            //     unsafe {
            //         core::ptr::copy(cell_indices.buffer.as_ptr().add(j), cell_indices.buffer.as_mut_ptr().add(j + 1), i - j);
            //     }
            //     unsafe {
            //         core::ptr::copy(
            //             self.indices.buffer.as_ptr().add(j),
            //             self.indices.buffer.as_mut_ptr().add(j + 1),
            //             i - j,
            //         );
            //     }
            //     // safe version:
            //     //cell_indices.buffer.copy_within(j..i, j + 1);
            //     //self.indices.buffer.copy_within(j..i, j + 1);
            // }
            // cell_indices.buffer[j] = cidx;
            // self.indices.buffer[j] = i as ParticleIndex;
        }

        // Apply sorting.
        {
            microprofile::scope!("NeighborhoodSearch", "apply sorting");
            {
                let mut scratch_buffer = scratch_buffers.get_buffer_point(positions.len());
                Self::apply_sorting(&particle_indices.buffer, &mut scratch_buffer.buffer, positions);
            }
            {
                let mut scratch_buffer = scratch_buffers.get_buffer_vector(positions.len());
                for attribute_buffer in particle_attributes_vector.iter_mut() {
                    Self::apply_sorting(&particle_indices.buffer, &mut scratch_buffer.buffer, *attribute_buffer);
                }
            }
            {
                let mut scratch_buffer = scratch_buffers.get_buffer_real(positions.len());
                for attribute_buffer in particle_attributes_real.iter_mut() {
                    Self::apply_sorting(&particle_indices.buffer, &mut scratch_buffer.buffer, *attribute_buffer);
                }
            }
        }

        // create cells.
        // we could do this during the sort and use the prefix sums for some clever jumping. Tried it and wasn't great (both perf & impl niceness)
        self.cells.clear();
        let mut prev_cidx = CellIndex::max_value();
        for (pidx, &cidx) in cell_indices.buffer.iter().enumerate() {
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

    pub fn foreach_particle_in_cell_box(&self, cidx_min: CellIndex, cidx_max: CellIndex, mut f: impl FnMut(usize) -> ()) {
        let cidx_min_xbits = cidx_min & super::morton::MORTON_XBITS;
        let cidx_min_ybits = cidx_min & super::morton::MORTON_YBITS;
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
                f(p);
            }

            // We know current cell isn't in the rect, so skip it.
            cell_arrayidx += 1;
            if cell_arrayidx >= self.cells.len() {
                break;
            }
            cell = self.cells[cell_arrayidx];
        }
    }

    fn get_cell_neighborbox(cidx: CellIndex) -> (CellIndex, CellIndex) {
        let pos = CellPos::from_cidx(cidx);
        let min = CellPos { x: pos.x - 1, y: pos.y - 1 };
        let max = CellPos { x: pos.x + 1, y: pos.y + 1 };
        (min.to_cidx(), max.to_cidx())
    }

    pub fn foreach_potential_neighbor(&self, grid: &GridProperties, position: Point, f: impl FnMut(usize) -> ()) {
        let cidx_min = grid.position_to_cidx(position - Vector::new(grid.radius, grid.radius));
        let cidx_max = grid.position_to_cidx(position + Vector::new(grid.radius, grid.radius));
        self.foreach_particle_in_cell_box(cidx_min, cidx_max, f);
    }
}

#[derive(Default)]
pub struct NeighborLists {
    neighborhood_list_starts: Vec<u32>,
    neighborhood_lists: Vec<ParticleIndex>,
}

impl NeighborLists {
    fn update(&mut self, grid: &GridProperties, positions: &[Point], cell_grid: &CellGrid) {
        microprofile::scope!("NeighborhoodSearch", "NeighborLists::update");
        assert_eq!(cell_grid.cells.last().unwrap().first_particle, positions.len());

        self.neighborhood_list_starts.resize(positions.len() + 1, 0);
        self.neighborhood_lists.clear();
        let radius_sq = grid.radius * grid.radius;
        let mut local_neighbor_sets: Vec<Vec<ParticleIndex>> = Vec::new(); // todo: don't realloc those?

        for cell_slice in cell_grid.cells.windows(2) {
            let current_cell = cell_slice[0];
            let next_cell = cell_slice[1];
            let num_particles_in_cell = next_cell.first_particle - current_cell.first_particle;

            for neighbor_set in local_neighbor_sets.iter_mut() {
                neighbor_set.clear();
            }
            local_neighbor_sets.resize_with(num_particles_in_cell, || Vec::<ParticleIndex>::with_capacity(100)); // todo: what's a good capacity value?

            let (cidx_min, cidx_max) = CellGrid::get_cell_neighborbox(current_cell.cidx);

            cell_grid.foreach_particle_in_cell_box(cidx_min, cidx_max, |neighbor_pidx| {
                for pidx in current_cell.first_particle..next_cell.first_particle {
                    if pidx != neighbor_pidx && positions[pidx].distance2(positions[neighbor_pidx]) <= radius_sq {
                        local_neighbor_sets[pidx - current_cell.first_particle].push(neighbor_pidx as ParticleIndex);
                    }
                }
            });

            // write out neighbors.
            // todo: compress
            for pidx in current_cell.first_particle..next_cell.first_particle {
                self.neighborhood_list_starts[pidx] = self.neighborhood_lists.len() as u32;
                self.neighborhood_lists
                    .extend_from_slice(&local_neighbor_sets[pidx - current_cell.first_particle]);
            }
        }
        *self.neighborhood_list_starts.last_mut().unwrap() = self.neighborhood_lists.len() as u32;
    }

    #[inline]
    pub fn foreach_neighbor(&self, particle: ParticleIndex, mut f: impl FnMut(ParticleIndex) -> ()) {
        let first = self.neighborhood_list_starts[particle as usize] as usize;
        let last = self.neighborhood_list_starts[particle as usize + 1] as usize;
        for i in first..last {
            f(unsafe { *self.neighborhood_lists.get_unchecked(i) });
        }
    }
}

pub struct NeighborhoodSearch {
    grid: GridProperties,

    // todo: Erase boundary/particle knowledge and just work with registered point sets.
    cellgrid_particles: CellGrid,
    cellgrid_boundary: CellGrid,

    particle_particle_neighbors: NeighborLists,
    particle_boundary_neighbors: NeighborLists,
}

impl NeighborhoodSearch {
    /// * radius:               Radius that determines if a point is a neighbor
    /// * expected_max_density: Num particles expected per square unit
    pub fn new(radius: Real, //    , expected_max_density: Real
    ) -> NeighborhoodSearch {
        let cell_size = radius * 2.0; // todo: experiment with larger cells

        //const particle_INDICES.buffer_PER_CACHELINE: u32 = 64 / std::mem::size_of::<ParticleIndex>() as u32;
        //let mut num_expected_in_cell = (cell_size * cell_size * expected_max_density + 0.5) as u32;
        //num_expected_in_cell = (num_expected_in_cell + particle_INDICES.buffer_PER_CACHELINE-1) / particle_INDICES.buffer_PER_CACHELINE * particle_INDICES.buffer_PER_CACHELINE;

        NeighborhoodSearch {
            grid: GridProperties {
                radius,
                cell_size_inv: 1.0 / cell_size,
                // todo: we can create a huge domain, but still there is a limited domain! should be safe about this and have a max
                // limit is there because our morton curve wraps around at some point and then things get complicated (aka don't want to deal with this!)
                grid_min: Point::new(-100.0, -100.0),
            },

            cellgrid_particles: Default::default(),
            cellgrid_boundary: Default::default(),

            particle_particle_neighbors: NeighborLists::default(),
            particle_boundary_neighbors: NeighborLists::default(),
        }
    }

    // todo: allow boundaries to have properties
    pub fn update_boundary(&mut self, scratch_buffers: &mut ScratchBufferStore, positions: &mut Vec<Point>) {
        microprofile::scope!("NeighborhoodSearch", "update_boundary");
        self.cellgrid_boundary.update(scratch_buffers, &self.grid, positions, &mut [], &mut []);
    }

    pub fn update_particle_neighbors(
        &mut self,
        scratch_buffers: &mut ScratchBufferStore,
        particle_positions: &mut Vec<Point>,
        particle_attributes_vector: &mut [&mut Vec<Vector>],
        particle_attributes_real: &mut [&mut Vec<Real>],
        boundary_positions: &[Point],
    ) {
        microprofile::scope!("NeighborhoodSearch", "update_particle_neighbors");
        self.cellgrid_particles.update(
            scratch_buffers,
            &self.grid,
            particle_positions,
            particle_attributes_vector,
            particle_attributes_real,
        );
        self.particle_particle_neighbors
            .update(&self.grid, particle_positions, &self.cellgrid_particles);
        // if !boundary_positions.is_empty() {
        //     self.particle_boundary_neighbors
        //         .update(&self.grid, particle_positions, &self.cellgrid_boundary, boundary_positions);
        // }
    }

    #[inline]
    pub fn foreach_neighbor(&self, particle: ParticleIndex, f: impl FnMut(ParticleIndex) -> ()) {
        self.particle_particle_neighbors.foreach_neighbor(particle, f);
    }

    pub fn foreach_potential_neighbor(&self, position: Point, f: impl FnMut(usize) -> ()) {
        self.cellgrid_particles.foreach_potential_neighbor(&self.grid, position, f)
    }

    pub fn foreach_potential_boundary_neighbor(&self, position: Point, f: impl FnMut(usize) -> ()) {
        self.cellgrid_boundary.foreach_potential_neighbor(&self.grid, position, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::prelude::*;
    use rand::prelude::*;

    #[test]
    fn potential_neighbors_contains_neighbors() {
        const NUM_POSITIONS: usize = 1000;
        const DENSITY: Real = 10.0;
        const SEARCH_RADIUS: Real = 1.0;

        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(123456789);
        let mut positions: Vec<Point> = std::iter::repeat_with(|| Point::from_vec(rng.gen::<Vector>() * (NUM_POSITIONS as Real / DENSITY).sqrt()))
            .take(NUM_POSITIONS)
            .collect();

        let mut scratch_buffer_store = ScratchBufferStore::new();
        let mut searcher = NeighborhoodSearch::new(SEARCH_RADIUS);
        searcher.update_particle_neighbors(&mut scratch_buffer_store, &mut positions, &mut [], &mut [], &[]);

        for &search_pos in positions.iter() {
            let mut potential_neighbors = Vec::new();
            searcher.foreach_potential_neighbor(search_pos, |p| potential_neighbors.push(p));

            // validate
            for (i, &p) in positions.iter().enumerate() {
                if p.distance2(search_pos) <= SEARCH_RADIUS * SEARCH_RADIUS {
                    assert!(potential_neighbors.contains(&i));
                }
            }
        }
    }

    #[test]
    fn neighbors_contains_neighbors() {
        const NUM_POSITIONS: usize = 1000;
        const DENSITY: Real = 10.0;
        const SEARCH_RADIUS: Real = 1.0;

        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(123456789);
        let mut positions: Vec<Point> = std::iter::repeat_with(|| Point::from_vec(rng.gen::<Vector>() * (NUM_POSITIONS as Real / DENSITY).sqrt()))
            .take(NUM_POSITIONS)
            .collect();

        let mut scratch_buffer_store = ScratchBufferStore::new();
        let mut searcher = NeighborhoodSearch::new(SEARCH_RADIUS);
        searcher.update_particle_neighbors(&mut scratch_buffer_store, &mut positions, &mut [], &mut [], &[]);

        for (particle, &search_pos) in positions.iter().enumerate() {
            let mut neighbors = Vec::new();
            searcher.foreach_neighbor(particle as ParticleIndex, |p| neighbors.push(p));

            // validate
            let mut neighbors_bruteforce = Vec::new();
            for (i, &p) in positions.iter().enumerate() {
                if i != particle && p.distance2(search_pos) <= SEARCH_RADIUS * SEARCH_RADIUS {
                    neighbors_bruteforce.push(i as ParticleIndex);
                }
            }
            assert_eq!(neighbors, neighbors_bruteforce);
        }
    }
}
