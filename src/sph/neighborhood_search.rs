use cgmath::prelude::*;
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::ops::Range;

use super::appendbuffer::AppendBuffer;
use super::scratch_buffer::ScratchBufferStore;
use crate::units::*;

pub type ParticleIndex = u32;
pub type MortonCellIndex = u32;

#[derive(Copy, Clone, PartialEq)]
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
    first_particle: ParticleIndex,
    cidx: MortonCellIndex,
}

// Runs of particle indices for a MortonCell and its eight neighbors.
struct MortonCellNeighborhoodRuns {
    // In a 3x3 2D morton box there are at max 5 continuos runs (can be less!)
    particle_index_runs: [(ParticleIndex, ParticleIndex); 5],
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

    fn new() -> Self {
        CompactMortonCellGrid {
            cells: vec![MortonCell {
                first_particle: 0,
                cidx: MortonCellIndex::max_value(),
            }],
        }
    }

    // note: Applying sorting is a bit costly and only solvers know which attributes are discarded/recomputed and which need the new sorting applied.
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

        // Testruns, averaged over all steps of 30 frames in a test scene
        // Manual insertion sort, using a separate cell_indices buffer: avg 0.190ms, max 0.416ms
        // sort_unstable_by_key:                                        avg 0.667ms, max 0.889ms
        // sort_by_cached_key:                                          avg 0.171ms, max 0.405ms
        // sort_unstable_by_key, precomputed key:                       avg 0.159ms, max 0.336ms
        // par_sort_unstable_by_key, precomputed key:                   avg 0.128ms, max 0.242ms
        {
            microprofile::scope!("NeighborhoodSearch", "sort particle indices");
            for (i, (idx, cidx)) in particle_indices.buffer.iter_mut().zip(cell_indices.buffer.iter_mut()).enumerate() {
                *idx = i as u32;
                *cidx = grid.position_to_cidx(*unsafe { positions.get_unchecked(i) });
            }
            let b = &cell_indices.buffer;
            particle_indices
                .buffer
                .par_sort_unstable_by_key(|i| *unsafe { b.get_unchecked(*i as usize) });
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
        {
            microprofile::scope!("NeighborhoodSearch", "create morton cells");
            self.cells.clear();
            let mut prev_morton_pos = MortonCellPos {
                x: u16::max_value(),
                y: u16::max_value(),
            };
            for (pidx, &position) in positions.iter().enumerate() {
                let morton_pos = grid.position_to_mortoncellpos(position);
                if morton_pos != prev_morton_pos {
                    self.cells.push(MortonCell {
                        first_particle: pidx as ParticleIndex,
                        cidx: morton_pos.to_cidx(),
                    });
                    prev_morton_pos = morton_pos;
                }
            }
            self.cells.push(MortonCell {
                first_particle: positions.len() as ParticleIndex,
                cidx: MortonCellIndex::max_value(),
            }); // sentinel cell
        }
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

    fn get_particle_runs_in_neighborbox(&self, cidx: MortonCellIndex) -> MortonCellNeighborhoodRuns {
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

        let mut runs = MortonCellNeighborhoodRuns {
            particle_index_runs: [(0, 0); 5],
        };
        let mut run_idx = 0;

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
            runs.particle_index_runs[run_idx].0 = cell.first_particle;
            loop {
                cell_arrayidx += 1; // we won't be here for long, no point in doing profound skipping.
                cell = self.cells[cell_arrayidx];
                if !super::morton::is_in_rect_presplit(cell.cidx, cidx_min_xbits, cidx_min_ybits, cidx_max_xbits, cidx_max_ybits) {
                    break;
                }
            }
            runs.particle_index_runs[run_idx].1 = cell.first_particle;
            run_idx += 1;
            if run_idx == runs.particle_index_runs.len() {
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
}

// For each particle we compute a list of all neighbors and store them into the neighborhood_lists.
// For each particle, this list starts ranges for
// dynamic particles:
// start_index..(start_index+count_dynamic)
// and for static particles:
// (start_index+count_dynamic)..(start_index+count_total)
#[derive(Copy, Clone, Default)]
struct NeighborRange {
    start_index: u32,
    count_dynamic: u16,
    count_total: u16,
}

impl NeighborRange {
    // fn range_total(&self) -> Range<usize> {
    //     (self.start_index as usize)..(self.start_index + self.count_total as u32) as usize
    // }

    fn range_dynamic(&self) -> Range<usize> {
        (self.start_index as usize)..(self.start_index + self.count_dynamic as u32) as usize
    }

    fn range_static(&self) -> Range<usize> {
        (self.start_index + self.count_dynamic as u32) as usize..(self.start_index + self.count_total as u32) as usize
    }
}

// We *know* every thread/task is writing a disjunct set of elements in this array.
// But since size of these sets variies and we're accessing it effectively at random, it's really hard to convince the compiler of our guarantee.
struct NeighborListRanges {
    list: UnsafeCell<Vec<NeighborRange>>,
}
unsafe impl Sync for NeighborListRanges {}
unsafe impl Send for NeighborListRanges {}

pub struct NeighborLists {
    neighborhood_list_ranges: NeighborListRanges,
    neighborhood_lists: AppendBuffer<u32>,
}

impl NeighborLists {
    fn new() -> NeighborLists {
        NeighborLists {
            neighborhood_list_ranges: NeighborListRanges {
                list: UnsafeCell::new(Vec::with_capacity(1024)),
            },
            neighborhood_lists: AppendBuffer::new(),
        }
    }

    fn try_update(
        &mut self,
        grid: &GridProperties,
        cellgrid_dynamic: &CompactMortonCellGrid,
        cellgrid_static: &CompactMortonCellGrid,
        positions_dynamic: &[Point],
        positions_static: &[Point],
    ) -> Result<usize, usize> {
        microprofile::scope!("NeighborhoodSearch", "NeighborLists::try_update");

        const MAX_NUM_NEIGHBORS: u16 = 64; // todo: At least pretend to be scientific about this value.
        const MIN_DISTANCE: Real = 1.0e-10; // used to filter for degenerated cases & self intersect

        self.neighborhood_list_ranges
            .list
            .get_mut()
            .resize(positions_dynamic.len(), Default::default());
        self.neighborhood_lists.clear();
        self.neighborhood_lists.resize(positions_dynamic.len() * MAX_NUM_NEIGHBORS as usize); // TODO: Smaller? Needs we need to handle error on overflow.
        let radius_sq = grid.radius * grid.radius;

        // Look at cell pairs in the compact cell grid since next cell tells us how many particles are in the current.
        // Going by cells instead of particles allows us to query neighbor cells / runs of neighbor particles only once
        cellgrid_dynamic.cells.par_windows(2).for_each(|cell_slice| {
            let current_cell = cell_slice[0];
            let next_cell = cell_slice[1];

            let mut neighbor_set = [0; MAX_NUM_NEIGHBORS as usize];

            // set of all potential neighbors
            let particle_runs_dynamic = cellgrid_dynamic.get_particle_runs_in_neighborbox(current_cell.cidx);
            let particle_runs_static = cellgrid_static.get_particle_runs_in_neighborbox(current_cell.cidx);

            // for each particle in this cell...
            for i in current_cell.first_particle..next_cell.first_particle {
                let query_pos = unsafe { *positions_dynamic.get_unchecked(i as usize) };

                // gather real neighbors
                let mut count_dynamic = 0;
                'neighbor_search_dynamic: for range in particle_runs_dynamic.particle_index_runs {
                    for j in range.0..range.1 {
                        let posj = unsafe { *positions_dynamic.get_unchecked(j as usize) };
                        let distsq = query_pos.distance2(posj);
                        if distsq <= radius_sq && distsq > MIN_DISTANCE {
                            neighbor_set[count_dynamic as usize] = j;
                            count_dynamic += 1;
                            if count_dynamic == MAX_NUM_NEIGHBORS {
                                println!("particle has too many neighbors");
                                break 'neighbor_search_dynamic;
                            }
                        }
                    }
                }
                let mut count_total = count_dynamic;
                'neighbor_search_static: for range in particle_runs_static.particle_index_runs {
                    for j in range.0..range.1 {
                        let posj = unsafe { *positions_static.get_unchecked(j as usize) };
                        let distsq = query_pos.distance2(posj);
                        if distsq <= radius_sq && distsq > MIN_DISTANCE {
                            neighbor_set[count_total as usize] = j;
                            count_total += 1;
                            if count_total == MAX_NUM_NEIGHBORS {
                                println!("particle has too many neighbors");
                                break 'neighbor_search_static;
                            }
                        }
                    }
                }

                // save neighbors
                let start_index = self.neighborhood_lists.extend_from_slice(&neighbor_set[..count_total as usize]).unwrap() as u32;
                let neighbor_range = NeighborRange {
                    start_index,
                    count_dynamic,
                    count_total,
                };
                unsafe {
                    *(&mut *self.neighborhood_list_ranges.list.get()).get_unchecked_mut(i as usize) = neighbor_range;
                }
            }
        });

        Ok(self.neighborhood_lists.len())
    }

    fn update(
        &mut self,
        grid: &GridProperties,
        cellgrid_dynamic: &CompactMortonCellGrid,
        cellgrid_static: &CompactMortonCellGrid,
        positions_dynamic: &[Point],
        positions_static: &[Point],
    ) {
        microprofile::scope!("NeighborhoodSearch", "NeighborLists::update");
        assert_eq!(cellgrid_dynamic.cells.last().unwrap().first_particle as usize, positions_dynamic.len());
        assert_eq!(cellgrid_static.cells.last().unwrap().first_particle as usize, positions_static.len());

        while self
            .try_update(grid, cellgrid_dynamic, cellgrid_static, positions_dynamic, positions_static)
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

    // Doesn't work out because boundary positions are in a separate buffer
    // pub fn neighbors_all<'a>(&'a self, particle: ParticleIndex) -> &[u32] {
    //     unsafe {
    //         let range = (*self.neighborhood_list_ranges.list.get()).get_unchecked(particle as usize);
    //         self.neighborhood_lists.as_slice().get_unchecked(range.range_total())
    //     }
    // }

    pub fn neighbors_dynamic<'a>(&'a self, particle: ParticleIndex) -> &'a [u32] {
        unsafe {
            let range = (*self.neighborhood_list_ranges.list.get()).get_unchecked(particle as usize);
            self.neighborhood_lists.as_slice().get_unchecked(range.range_dynamic())
        }
    }

    pub fn neighbors_static<'a>(&'a self, particle: ParticleIndex) -> &'a [u32] {
        unsafe {
            let range = (*self.neighborhood_list_ranges.list.get()).get_unchecked(particle as usize);
            self.neighborhood_lists.as_slice().get_unchecked(range.range_static())
        }
    }

    pub fn num_neighbors(&self, particle: ParticleIndex) -> u16 {
        unsafe { (*self.neighborhood_list_ranges.list.get()).get_unchecked(particle as usize) }.count_total
    }
}

pub struct NeighborhoodSearch {
    grid: GridProperties,

    cellgrid_dynamic: CompactMortonCellGrid,
    cellgrid_static: CompactMortonCellGrid,

    neighbor_lists: NeighborLists,
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

            cellgrid_dynamic: CompactMortonCellGrid::new(),
            cellgrid_static: CompactMortonCellGrid::new(),

            neighbor_lists: NeighborLists::new(),
        }
    }

    pub fn update_static(&mut self, scratch_buffers: &mut ScratchBufferStore, positions: &mut Vec<Point>) {
        microprofile::scope!("NeighborhoodSearch", "update_boundary");
        self.cellgrid_static.update(scratch_buffers, &self.grid, positions, &mut [], &mut []);
    }

    pub fn update_dynamic(
        &mut self,
        scratch_buffers: &mut ScratchBufferStore,
        positions_dynamic: &mut Vec<Point>,
        particle_attributes_vector: &mut [&mut Vec<Vector>],
        particle_attributes_real: &mut [&mut Vec<Real>],
        positions_static: &[Point],
    ) {
        microprofile::scope!("NeighborhoodSearch", "update_particle_neighbors");
        self.cellgrid_dynamic.update(
            scratch_buffers,
            &self.grid,
            positions_dynamic,
            particle_attributes_vector,
            particle_attributes_real,
        );
        self.neighbor_lists.update(
            &self.grid,
            &self.cellgrid_dynamic,
            &self.cellgrid_static,
            positions_dynamic,
            positions_static,
        );
    }

    #[inline(always)]
    pub fn neighbor_lists(&self) -> &NeighborLists {
        &self.neighbor_lists
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

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
        searcher.update_dynamic(&mut scratch_buffer_store, &mut positions, &mut [], &mut [], &[]);

        for (particle, &search_pos) in positions.iter().enumerate() {
            let neighbors = searcher.neighbor_lists().neighbors_dynamic(particle as u32).to_vec();

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
