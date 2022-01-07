use cgmath::prelude::*;
use criterion::{black_box, criterion_group, Criterion};
use rand::prelude::*;

use yasph2d::sph::neighborhood_search::NeighborhoodSearch;
use yasph2d::sph::scratch_buffer::ScratchBufferStore;
use yasph2d::units::*;

fn bench_neighborhood_search(c: &mut Criterion) {
    const NUM_POSITIONS: usize = 20000;
    const DENSITY: Real = 10.0;
    let search_radius = black_box(1.0);

    let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(123456789);
    let mut positions: Vec<Point> = std::iter::repeat_with(|| Point::from_vec(rng.gen::<Vector>() * (NUM_POSITIONS as Real / DENSITY).sqrt()))
        .take(NUM_POSITIONS)
        .collect();

    let mut scratch_buffer_store = ScratchBufferStore::new();
    let mut searcher = NeighborhoodSearch::new(search_radius);
    searcher.update_particle_neighbors(&mut scratch_buffer_store, &mut positions, &mut [], &mut [], &[]);

    c.bench_function(
        &format!(
            "neighborhood_search.update (warm), {} positions, {} density, {} search_radius",
            NUM_POSITIONS, DENSITY, search_radius
        ),
        |b| b.iter(|| searcher.update_particle_neighbors(&mut scratch_buffer_store, &mut positions, &mut [], &mut [], &[])),
    );

    c.bench_function(
        &format!(
            "neighborhood_search.foreach_neighbor, {} positions, {} density, {} search_radius",
            NUM_POSITIONS, DENSITY, search_radius
        ),
        |b| {
            let mut pindex = 0; // cycle through position for a more balanced result
            b.iter(|| {
                let mut accum: Vector = Zero::zero();
                searcher.foreach_neighbor(pindex, |i| {
                    accum += positions[i as usize].to_vec();
                });
                pindex = (pindex + 1) % NUM_POSITIONS as u32;
                accum
            })
        },
    );
}

fn config() -> Criterion {
    Criterion::default().warm_up_time(core::time::Duration::new(0, 1000))
}

criterion_group!(
    name = neighborhood_search;
    config = config();
    targets = bench_neighborhood_search
);
