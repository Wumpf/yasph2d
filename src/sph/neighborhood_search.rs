use super::appendbuffer::AppendBuffer;
use super::scratch_buffer::ScratchBufferStore;
use crate::units::*;

use cgmath::prelude::*;

pub type ParticleIndex = u32;
pub type MortonCellIndex = u32;

#[derive(Copy, Clone)]
struct MortonCellPos {
    x: u16,
    y: u16,
}
impl MortonCellPos {
    #[inline]
    fn to_cidx(self) -> MortonCellIndex {
        super::morton::encode(self.x, self.y)
    }

    #[inline]
    fn from_cidx(cidx: MortonCellIndex) -> MortonCellPos {
        MortonCellPos {
            x: super::morton::decode_x(cidx) as u16,
            y: super::morton::decode_y(cidx) as u16,
        }
    }
}

#[derive(Copy, Clone)]
struct MortonCell {
    first_particle: usize,
    cidx: MortonCellIndex,
}

// Runs of particle indices for a MortonCell and its eight neighbors.
struct MortonCellNeihborhoodRuns {
    // In a 3x3 2D morton box there are at max 5 continous runs (can be less!)
    particle_index_runs: [(usize, usize); 5],
    num_runs: usize, // remove? not really needed I guess
}

struct GridProperties {
    radius: Real,
    cell_size_inv: Real,
    grid_min: Point,
}
impl GridProperties {
    #[inline]
    fn position_to_mortoncellpos(&self, position: Point) -> MortonCellPos {
        let cellspace = (position - self.grid_min) * self.cell_size_inv;
        MortonCellPos {
            x: cellspace.x as u16,
            y: cellspace.y as u16,
        }
    }

    #[inline]
    fn position_to_cidx(&self, position: Point) -> MortonCellIndex {
        self.position_to_mortoncellpos(position).to_cidx()
    }
}

#[derive(Default)]
struct CompactMortonCellGrid {
    cells: Vec<MortonCell>,
}

impl CompactMortonCellGrid {
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
        microprofile::scope!("NeighborhoodSearch", "CompactMortonCellGrid::update");

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
            // Note: Tried memcpy instead of swaps but was significantly slower in benchmarks - probably not compiler friendly
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
        let mut prev_cidx = MortonCellIndex::max_value();
        for (pidx, &cidx) in cell_indices.buffer.iter().enumerate() {
            if cidx != prev_cidx {
                self.cells.push(MortonCell { first_particle: pidx, cidx });
                prev_cidx = cidx;
            }
        }
        self.cells.push(MortonCell {
            first_particle: positions.len(),
            cidx: MortonCellIndex::max_value(),
        }); // sentinel cell
    }

    // finds cell array index first cell that has an equal or bigger for a given MortonCellIndex
    fn find_next_cell(cells: &[MortonCell], cidx: MortonCellIndex) -> usize {
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

    fn get_particle_runs_in_neighborbox(&self, cidx: MortonCellIndex) -> MortonCellNeihborhoodRuns {
        let pos = MortonCellPos::from_cidx(cidx);
        let cidx_min = MortonCellPos { x: pos.x - 1, y: pos.y - 1 }.to_cidx();
        let cidx_max = MortonCellPos { x: pos.x + 1, y: pos.y + 1 }.to_cidx();

        let cidx_min_xbits = cidx_min & super::morton::MORTON_XBITS;
        let cidx_min_ybits = cidx_min & super::morton::MORTON_YBITS;
        let cidx_max_xbits = cidx_max & super::morton::MORTON_XBITS;
        let cidx_max_ybits = cidx_max & super::morton::MORTON_YBITS;

        const MAX_CONSECUTIVE_CELL_MISSES: u32 = 8;

        // Note: Already tried doing this with iterators. it's hard to do and slow!
        let mut cell_arrayidx = Self::find_next_cell(&self.cells, cidx_min);
        let mut cell = self.cells[cell_arrayidx];

        let mut runs = MortonCellNeihborhoodRuns {
            particle_index_runs: [(0, 0); 5],
            num_runs: 0,
        };

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
                    return runs;
                }
            }

            // find particle run
            runs.particle_index_runs[runs.num_runs].0 = cell.first_particle;
            loop {
                cell_arrayidx += 1; // we won't be here for long, no point in doing profound skipping.
                cell = self.cells[cell_arrayidx];
                if !super::morton::is_in_rect_presplit(cell.cidx, cidx_min_xbits, cidx_min_ybits, cidx_max_xbits, cidx_max_ybits) {
                    break;
                }
            }
            runs.particle_index_runs[runs.num_runs].1 = cell.first_particle;
            runs.num_runs += 1;
            if runs.num_runs == runs.particle_index_runs.len() {
                break;
            }

            assert_ne!(cell.cidx, cidx_max); // it if was equal, then there would be a cell at cidx_max that is not in the rect limited by cidx_max

            // We know current cell isn't in the rect, so skip it.
            cell_arrayidx += 1;
            if cell_arrayidx >= self.cells.len() {
                break;
            }
            cell = self.cells[cell_arrayidx];
        }

        runs
    }

    // todo: remove, impl already no longer optimal
    pub fn foreach_potential_neighbor(&self, grid: &GridProperties, position: Point, mut f: impl FnMut(usize) -> ()) {
        let runs = self.get_particle_runs_in_neighborbox(grid.position_to_cidx(position));
        for range in runs.particle_index_runs.iter() {
            for j in range.0..range.1 {
                f(j);
            }
        }
    }
}

pub struct NeighborLists {
    neighborhood_list_starts: Vec<u32>,
    neighborhood_lists: AppendBuffer<ParticleIndex>,
}

impl NeighborLists {
    fn new() -> NeighborLists {
        NeighborLists {
            neighborhood_list_starts: Vec::with_capacity(1024),
            neighborhood_lists: AppendBuffer::with_capacity(1024 * 32),
        }
    }

    fn try_update(
        &mut self,
        grid: &GridProperties,
        positions: &[Point],
        cell_grid: &CompactMortonCellGrid,
        neighbor_positions: &[Point],
        neighbor_cell_grid: &CompactMortonCellGrid,
    ) -> Result<usize, usize> {
        self.neighborhood_list_starts.resize(positions.len() + 1, 0);
        self.neighborhood_lists.clear();
        let radius_sq = grid.radius * grid.radius;

        const MAX_NUM_NEIGHBORS: usize = 128; // todo: At least pretend to be scientific about this value.
        let mut neighbor_set = [0; MAX_NUM_NEIGHBORS];

        // Look at cell pairs in the compact cell grid since next cell tells us how many particles are in the current.
        for cell_slice in cell_grid.cells.windows(2) {
            let current_cell = cell_slice[0];
            let next_cell = cell_slice[1];

            // set of all potential neighbors
            let particle_runs = neighbor_cell_grid.get_particle_runs_in_neighborbox(current_cell.cidx);

            // for each particle in this cell...
            for i in current_cell.first_particle..next_cell.first_particle {
                let posi = unsafe { *positions.get_unchecked(i) };

                // gather real neighbors
                const MIN_DISTANCE: Real = 1.0e-10; // used to filter for degenerated cases & self intersect
                let mut num_neighbors = 0;
                'neighbor_search: for range in particle_runs.particle_index_runs.iter() {
                    for j in range.0..range.1 {
                        let posj = unsafe { *neighbor_positions.get_unchecked(j) };
                        let distsq = posi.distance2(posj);
                        if distsq <= radius_sq && distsq > MIN_DISTANCE {
                            neighbor_set[num_neighbors] = j as u32;
                            num_neighbors += 1;
                            if num_neighbors == MAX_NUM_NEIGHBORS {
                                //println!("particle has too many neighbors");
                                break 'neighbor_search;
                            }
                        }
                    }
                }

                // safe neighbors
                // todo: compress
                self.neighborhood_list_starts[i] = self.neighborhood_lists.len() as u32;
                self.neighborhood_lists.extend_from_slice(&neighbor_set[..num_neighbors])?;
            }
        }

        *self.neighborhood_list_starts.last_mut().unwrap() = self.neighborhood_lists.len() as u32;

        Ok(self.neighborhood_lists.len())
    }

    fn update(
        &mut self,
        grid: &GridProperties,
        positions: &[Point],
        cell_grid: &CompactMortonCellGrid,
        neighbor_positions: &[Point],
        neighbor_cell_grid: &CompactMortonCellGrid,
    ) {
        microprofile::scope!("NeighborhoodSearch", "NeighborLists::update");
        assert_eq!(cell_grid.cells.last().unwrap().first_particle, positions.len());

        while self
            .try_update(grid, positions, cell_grid, neighbor_positions, neighbor_cell_grid)
            .is_err()
        {
            let new_capacity = self.neighborhood_lists.capacity() * 2;
            println!(
                "Neighbor list capacity was too small. Was {}, trying again with {}",
                self.neighborhood_lists.capacity(),
                new_capacity
            );
            self.neighborhood_lists.resize(new_capacity * 2)
        }
    }

    #[inline]
    pub fn foreach_neighbor(&self, particle: ParticleIndex, mut f: impl FnMut(ParticleIndex) -> ()) {
        unsafe {
            let first = *self.neighborhood_list_starts.get_unchecked(particle as usize) as usize;
            let last = *self.neighborhood_list_starts.get_unchecked((particle + 1) as usize) as usize;
            let neighborhood_lists = self.neighborhood_lists.as_slice();
            for i in first..last {
                f(*neighborhood_lists.get_unchecked(i));
            }
        }
    }
}

pub struct NeighborhoodSearch {
    grid: GridProperties,

    // todo: Erase boundary/particle knowledge and just work with registered point sets.
    cellgrid_particles: CompactMortonCellGrid,
    cellgrid_boundary: CompactMortonCellGrid,

    particle_particle_neighbors: NeighborLists,
    particle_boundary_neighbors: NeighborLists,
}

impl NeighborhoodSearch {
    /// * radius:               Radius that determines if a point is a neighbor
    /// * expected_max_density: Num particles expected per square unit
    pub fn new(radius: Real, //    , expected_max_density: Real
    ) -> NeighborhoodSearch {
        let cell_size = radius;

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

            particle_particle_neighbors: NeighborLists::new(),
            particle_boundary_neighbors: NeighborLists::new(),
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
        self.particle_particle_neighbors.update(
            &self.grid,
            particle_positions,
            &self.cellgrid_particles,
            particle_positions,
            &self.cellgrid_particles,
        );
        if !boundary_positions.is_empty() {
            self.particle_boundary_neighbors.update(
                &self.grid,
                particle_positions,
                &self.cellgrid_particles,
                boundary_positions,
                &self.cellgrid_boundary,
            );
        }
    }

    #[inline]
    pub fn foreach_neighbor(&self, particle: ParticleIndex, f: impl FnMut(ParticleIndex) -> ()) {
        self.particle_particle_neighbors.foreach_neighbor(particle, f);
    }

    #[inline]
    pub fn foreach_boundary_neighbor(&self, particle: ParticleIndex, f: impl FnMut(ParticleIndex) -> ()) {
        self.particle_boundary_neighbors.foreach_neighbor(particle, f);
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
