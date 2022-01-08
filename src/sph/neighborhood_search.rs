use cgmath::prelude::*;
use rayon::prelude::*;
use std::cell::UnsafeCell;

use super::appendbuffer::AppendBuffer;
use super::scratch_buffer::ScratchBufferStore;
use crate::units::*;

pub type ParticleIndex = u32;
pub type CellIndex = u32;

const MIN_DISTANCE_SQ: Real = 1.0e-10; // used to filter for degenerated cases & self intersect

#[derive(Copy, Clone)]
struct CellPos {
    x: u16,
    y: u16,
}
impl CellPos {
    #[inline]
    fn to_cellidx(self) -> CellIndex {
        super::morton::encode(self.x, self.y)
    }

    // #[inline]
    // fn from_cellidx(cidx: CellIndex) -> CellPos {
    //     CellPos {
    //         x: super::morton::decode_x(cidx) as u16,
    //         y: super::morton::decode_y(cidx) as u16,
    //     }
    // }
}

struct GridProperties {
    cell_size: Real,
    cell_size_inv: Real,
    num_cells_x: u16,
    num_cells_y: u16,
    grid_min: Point,
    //grid_max: Point,
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
    fn position_to_cellidx(&self, position: Point) -> CellIndex {
        self.position_to_cellpos(position).to_cellidx()
    }
}

const BLOCK_SIZE_BITS: usize = 2;
const BLOCK_SIZE: usize = 1 << BLOCK_SIZE_BITS;

#[derive(Copy, Clone)]
struct Cell {
    // 24bit for particle index, 8bit for particle count
    data: u32,
}

impl Cell {
    fn new(num_particles: u32, first_particle: ParticleIndex) -> Self {
        debug_assert!(num_particles < std::u8::MAX as u32);
        debug_assert!(first_particle < (1 << 24));
        Cell {
            data: num_particles + (first_particle << 8),
        }
    }

    fn num_particles(self) -> u32 {
        self.data & 0x0000_00ff
    }

    fn first_particle(self) -> u32 {
        self.data >> 8
    }

    fn add_particle(&mut self) {
        debug_assert!(self.num_particles() < 256);
        self.data += 1;
    }
}

#[derive(Copy, Clone)]
struct Block {
    cells: [Cell; BLOCK_SIZE * BLOCK_SIZE],
}

impl Block {
    fn new() -> Self {
        Block {
            cells: [Cell::new(0, 0); BLOCK_SIZE * BLOCK_SIZE],
        }
    }

    fn get_cell_idx_in_block(i: CellIndex) -> u32 {
        i & (0xffff_ffff >> (32 - BLOCK_SIZE_BITS * 2))
    }
}

impl std::ops::Index<CellIndex> for Block {
    type Output = Cell;
    fn index<'a>(&'a self, i: CellIndex) -> &'a Cell {
        &self.cells[Block::get_cell_idx_in_block(i) as usize]
    }
}

impl std::ops::IndexMut<CellIndex> for Block {
    fn index_mut<'a>(&'a mut self, i: CellIndex) -> &'a mut Cell {
        &mut self.cells[Block::get_cell_idx_in_block(i) as usize]
    }
}

struct CellGrid {
    // Top level grid, always filled out, covers the entire domain in low_level_cells
    block_grid: Vec<u32>,
    // Low level cells, added on demand. First element is always present as a sentinel.
    blocks: Vec<Block>,
}

impl CellGrid {
    pub fn new(grid: &GridProperties) -> Self {
        let num_blocks_x = (grid.num_cells_x as usize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let num_blocks_y = (grid.num_cells_y as usize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let num_blocks = num_blocks_x * num_blocks_y;

        CellGrid {
            block_grid: vec![0; num_blocks],
            blocks: vec![Block::new()],
        }
    }

    fn apply_sorting<T: Copy>(sorting: &[ParticleIndex], scratch_buffer: &mut Vec<T>, buffer_to_sort: &mut Vec<T>) {
        assert_eq!(scratch_buffer.len(), buffer_to_sort.len());
        assert_eq!(sorting.len(), buffer_to_sort.len());
        for (pos, &i) in scratch_buffer.iter_mut().zip(sorting.iter()) {
            *pos = buffer_to_sort[i as usize];
        }
        std::mem::swap(scratch_buffer, buffer_to_sort);
    }

    fn get_grid_idx(i: CellIndex) -> usize {
        (i >> (BLOCK_SIZE_BITS * 2)) as usize
    }

    fn get_block_idx(&self, i: CellIndex) -> u32 {
        *self.block_grid.get(Self::get_grid_idx(i)).unwrap_or(&0)
    }

    fn get_cell(&self, i: CellIndex) -> Cell {
        unsafe { self.blocks.get_unchecked(self.get_block_idx(i) as usize)[i] }
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

        {
            // TODO: Garbage collect blocks?
            microprofile::scope!("NeighborhoodSearch", "reset grid");
            self.blocks.fill(Block::new());
        }

        let particle_indices = &mut scratch_buffers.get_buffer_uint(positions.len());

        // Testruns, averaged over all steps of 30 frames in a test scene
        // Manual insertion sort, using a separate cell_indices buffer: avg 0.190ms, max 0.416ms
        // sort_unstable_by_key:                                        avg 0.667ms, max 0.889ms
        // sort_by_cached_key:                                          avg 0.171ms, max 0.405ms
        {
            microprofile::scope!("NeighborhoodSearch", "sort particle indices");
            for (i, idx) in particle_indices.buffer.iter_mut().enumerate() {
                *idx = i as u32;
            }
            particle_indices
                .buffer
                .sort_by_cached_key(|i| grid.position_to_cellidx(*unsafe { positions.get_unchecked(*i as usize) }));
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

        // create/update cells.
        {
            microprofile::scope!("NeighborhoodSearch", "create cells");

            for (pidx, &pos) in positions.iter().enumerate() {
                let cidx = grid.position_to_cellidx(pos);
                let grid_index = Self::get_grid_idx(cidx); // TODO out of bounds?
                let block_index = self.block_grid[grid_index];

                if block_index == 0 {
                    self.block_grid[grid_index] = self.blocks.len() as u32;
                    let mut new_block = Block::new();
                    new_block[cidx] = Cell::new(1, pidx as u32);
                    self.blocks.push(new_block);
                } else {
                    let cell = &mut self.blocks[block_index as usize][cidx];
                    if cell.data == 0 {
                        *cell = Cell::new(1, pidx as u32);
                    } else {
                        cell.add_particle();
                    }
                }
            }
        }
    }
}

// todo: Is there a way around this construct from hell?
// We *know* every thread/task is writing a disjunct set of elements in this array.
// But since size of these sets variies, it's really hard to convince the compiler of our guarantee.
struct NeighborListRanges {
    list: UnsafeCell<Vec<(u32, u32)>>,
}
unsafe impl Sync for NeighborListRanges {}
unsafe impl Send for NeighborListRanges {}

pub struct NeighborLists {
    neighborhood_list_ranges: NeighborListRanges,
    neighborhood_lists: AppendBuffer<ParticleIndex>,
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
        positions: &[Point],
        neighbor_positions: &[Point],
        neighbor_cell_grid: &CellGrid,
    ) -> Result<usize, usize> {
        microprofile::scope!("NeighborhoodSearch", "NeighborLists::try_update");

        const MAX_NUM_NEIGHBORS: usize = 64; // todo: At least pretend to be scientific about this value.

        unsafe {
            (*self.neighborhood_list_ranges.list.get()).resize(positions.len() + 1, (0, 0));
        }
        self.neighborhood_lists.clear();
        self.neighborhood_lists.resize(positions.len() * MAX_NUM_NEIGHBORS); // TODO: Smaller. Needs we need to handle error on overflow.
        let radius_sq = grid.cell_size * grid.cell_size;

        // TODO: Would it be more efficient to walk per block?
        positions.par_iter().enumerate().for_each(|(i, posi)| {
            let mut neighbor_set = [0; MAX_NUM_NEIGHBORS];
            let mut num_neighbors = 0;

            let particle_cell_pos = grid.position_to_cellpos(*posi);

            // TODO: wouldn't need to clamp the cell pos range if there was a guard band of sorts!
            // TODO: Not optimal order
            'neighbor_search: for x in particle_cell_pos.x.saturating_sub(1)..(particle_cell_pos.x + 2).min(grid.num_cells_x) {
                for y in particle_cell_pos.y.saturating_sub(1)..(particle_cell_pos.y + 2).min(grid.num_cells_y) {
                    let cidx = CellPos { x, y }.to_cellidx();
                    let cell = neighbor_cell_grid.get_cell(cidx);
                    let first_particle = cell.first_particle();

                    for neighbor_idx in first_particle..(first_particle + cell.num_particles()) {
                        let posj = unsafe { *neighbor_positions.get_unchecked(neighbor_idx as usize) };
                        let distsq = posi.distance2(posj);
                        if distsq <= radius_sq && distsq > MIN_DISTANCE_SQ {
                            if num_neighbors == MAX_NUM_NEIGHBORS {
                                println!("particle has too many neighbors");
                                break 'neighbor_search;
                            }
                            neighbor_set[num_neighbors] = neighbor_idx;
                            num_neighbors += 1;
                        }
                    }
                }
            }

            // save neighbors
            // todo: compress?
            let neighborhood_list_offset = self.neighborhood_lists.extend_from_slice(&neighbor_set[..num_neighbors]).unwrap();
            unsafe {
                *(&mut *self.neighborhood_list_ranges.list.get()).get_unchecked_mut(i) =
                    (neighborhood_list_offset as u32, (neighborhood_list_offset + num_neighbors) as u32);
            }
        });

        Ok(self.neighborhood_lists.len())
    }

    fn update(&mut self, grid: &GridProperties, positions: &[Point], neighbor_positions: &[Point], neighbor_cell_grid: &CellGrid) {
        microprofile::scope!("NeighborhoodSearch", "NeighborLists::update");

        while self.try_update(grid, positions, neighbor_positions, neighbor_cell_grid).is_err() {
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
    pub fn foreach_neighbor(&self, particle: ParticleIndex, mut f: impl FnMut(ParticleIndex)) {
        unsafe {
            let ranges = &*self.neighborhood_list_ranges.list.get();
            let range = *ranges.get_unchecked(particle as usize);
            let neighborhood_lists = self.neighborhood_lists.as_slice();
            for i in range.0..range.1 {
                f(*neighborhood_lists.get_unchecked(i as usize));
            }
        }
    }

    pub fn num_neighbors(&self, particle: ParticleIndex) -> u32 {
        unsafe {
            let ranges = &*self.neighborhood_list_ranges.list.get();
            let range = *ranges.get_unchecked(particle as usize);
            range.1 - range.0
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
    pub fn new(radius: Real, //    , expected_max_density: Real
    ) -> NeighborhoodSearch {
        let cell_size = radius;

        // todo: we can create a huge domain, but still there is a limited domain! should be safe about this and have a max
        // limit is there because our morton curve wraps around at some point and then things get complicated (aka don't want to deal with this!)
        let grid_min = Point::new(-200.0, -200.0);
        let grid_max = Point::new(200.0, 200.0);

        let grid = GridProperties {
            cell_size,
            cell_size_inv: 1.0 / cell_size,

            num_cells_x: ((grid_max.x - grid_min.x) / cell_size).ceil() as u16,
            num_cells_y: ((grid_max.y - grid_min.y) / cell_size).ceil() as u16,

            grid_min,
            // grid_max,
        };

        let cellgrid_particles = CellGrid::new(&grid);
        let cellgrid_boundary = CellGrid::new(&grid);

        NeighborhoodSearch {
            grid,
            cellgrid_particles,
            cellgrid_boundary,
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
        self.particle_particle_neighbors
            .update(&self.grid, particle_positions, particle_positions, &self.cellgrid_particles);
        if !boundary_positions.is_empty() {
            self.particle_boundary_neighbors
                .update(&self.grid, particle_positions, boundary_positions, &self.cellgrid_boundary);
        }
    }

    #[inline]
    pub fn foreach_neighbor(&self, particle: ParticleIndex, f: impl FnMut(ParticleIndex)) {
        self.particle_particle_neighbors.foreach_neighbor(particle, f);
    }

    #[inline]
    pub fn num_neighbors(&self, particle: ParticleIndex) -> u32 {
        self.particle_particle_neighbors.num_neighbors(particle)
    }

    #[inline]
    pub fn foreach_boundary_neighbor(&self, particle: ParticleIndex, f: impl FnMut(ParticleIndex)) {
        self.particle_boundary_neighbors.foreach_neighbor(particle, f);
    }

    #[inline]
    pub fn num_boundary_neighbors(&self, particle: ParticleIndex) -> u32 {
        self.particle_boundary_neighbors.num_neighbors(particle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn neighbors_contains_neighbors() {
        const NUM_POSITIONS: usize = 5000;
        const DENSITY: Real = 10.0;
        const SEARCH_RADIUS: Real = 1.0;

        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(123456789);
        let mut positions: Vec<Point> = std::iter::repeat_with(|| Point::from_vec(rng.gen::<Vector>() * (NUM_POSITIONS as Real / DENSITY).sqrt()))
            .take(NUM_POSITIONS)
            .collect();

        let mut scratch_buffer_store = ScratchBufferStore::new();
        let mut searcher = NeighborhoodSearch::new(SEARCH_RADIUS);
        searcher.update_particle_neighbors(&mut scratch_buffer_store, &mut positions, &mut [], &mut [], &mut []);

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

            // TODO: algo used to be order preserving. let's get back to that?
            neighbors.sort();
            neighbors_bruteforce.sort();
            assert_eq!(neighbors, neighbors_bruteforce);
        }
    }
}
