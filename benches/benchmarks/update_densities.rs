use criterion::{black_box, criterion_group, Criterion};
use ggez::graphics::Rect;

use yasph2d::{
    sph::{self, smoothing_kernel::CubicSpline},
    units::*,
};

fn bench_update_densities(c: &mut Criterion) {
    let mut fluid_world = sph::FluidParticleWorld::new(
        2.0,     // smoothing factor
        10000.0, // #particles/m²
        100.0,   // density of water (? this is 2d, not 3d where it's 1000 kg/m³)
    );
    // This would probably explode in a real simulation, but we don't care here.
    fluid_world.add_fluid_rect(&Rect::new(0.0, 0.0, 1.0, 1.0), 0.5);
    fluid_world.add_boundary_thick_line(Point::new(-0.5, 0.5), Point::new(1.5, 0.5), 20);
    fluid_world.update_neighborhood_datastructure(Vec::new(), Vec::new());

    let kernel = black_box(CubicSpline::new(fluid_world.properties.smoothing_length()));

    c.bench_function(
        &format!(
            "bench_update_densities - FluidParticleWorld with {} fluid particles and {} boundary particles",
            fluid_world.particles.num_dynamic_particles(),
            fluid_world.particles.num_boundary_particles()
        ),
        |b| b.iter(|| fluid_world.update_densities(kernel)),
    );
}

criterion_group!(update_densities, bench_update_densities);
