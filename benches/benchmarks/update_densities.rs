use criterion::{black_box, criterion_group, Criterion};
use ggez::graphics::Rect;

use yasph2d::{
    sph::{
        self,
        smoothing_kernel::{CubicSpline, Poly6},
    },
    units::*,
};

#[derive(Copy, Clone)]
pub struct NoOpKernel {}

impl sph::smoothing_kernel::Kernel for NoOpKernel {
    #[inline]
    fn evaluate(&self, _r_sq: Real, _r: Real) -> Real {
        1.0
    }

    #[inline]
    fn gradient(&self, _ri_to_rj: Vector, _r_sq: Real, _r: Real) -> Vector {
        Vector::new(1.0, 1.0)
    }

    #[inline]
    fn laplacian(&self, _r_sq: Real, _r: Real) -> Real {
        1.0
    }
}

// 10000 is the value SPlisHSPlasH uses, so probably also what DFSPH paper used
// (seems like a rather.. unfortunate?.. value)
// https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/blob/d5172c9/SPlisHSPlasH/SPHKernels.h#L613
const FAKE_LOOKUP_TABLE_KERNEL_RESOLUTION: usize = 10000;

#[derive(Copy, Clone)]
pub struct FakeLookupTableKernel {
    values: [Real; FAKE_LOOKUP_TABLE_KERNEL_RESOLUTION],
    smoothing_length_inv: Real,
}
impl FakeLookupTableKernel {
    fn new(smoothing_length: Real) -> Self {
        FakeLookupTableKernel {
            values: [0.123; FAKE_LOOKUP_TABLE_KERNEL_RESOLUTION],
            smoothing_length_inv: 1.0 / smoothing_length,
        }
    }
}

impl sph::smoothing_kernel::Kernel for FakeLookupTableKernel {
    #[inline]
    fn evaluate(&self, _r_sq: Real, r: Real) -> Real {
        let pos = ((r * self.smoothing_length_inv) as usize).min(FAKE_LOOKUP_TABLE_KERNEL_RESOLUTION - 2);
        //  0.5 * (self.values[pos] + self.values[pos + 1]) // Why average two values?? That's what the original impl does!
        self.values[pos]
    }

    #[inline]
    fn gradient(&self, _ri_to_rj: Vector, _r_sq: Real, _r: Real) -> Vector {
        unimplemented!();
    }

    #[inline]
    fn laplacian(&self, _r_sq: Real, _r: Real) -> Real {
        unimplemented!();
    }
}

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
            "bench_update_densities(CubicSpline) - FluidParticleWorld with {} fluid particles and {} boundary particles",
            fluid_world.particles.num_dynamic_particles(),
            fluid_world.particles.num_boundary_particles()
        ),
        |b| b.iter(|| fluid_world.update_densities(kernel)),
    );

    let kernel = black_box(Poly6::new(fluid_world.properties.smoothing_length()));
    c.bench_function(
        &format!(
            "bench_update_densities(Poly6) - FluidParticleWorld with {} fluid particles and {} boundary particles",
            fluid_world.particles.num_dynamic_particles(),
            fluid_world.particles.num_boundary_particles()
        ),
        |b| b.iter(|| fluid_world.update_densities(kernel)),
    );

    let kernel = black_box(NoOpKernel {});
    c.bench_function(
        &format!(
            "bench_update_densities(NoOpKernel) - FluidParticleWorld with {} fluid particles and {} boundary particles",
            fluid_world.particles.num_dynamic_particles(),
            fluid_world.particles.num_boundary_particles()
        ),
        |b| b.iter(|| fluid_world.update_densities(kernel)),
    );

    let kernel = black_box(FakeLookupTableKernel::new(fluid_world.properties.smoothing_length()));
    c.bench_function(
        &format!(
            "bench_update_densities(FakeLookupTableKernel) - FluidParticleWorld with {} fluid particles and {} boundary particles",
            fluid_world.particles.num_dynamic_particles(),
            fluid_world.particles.num_boundary_particles()
        ),
        |b| b.iter(|| fluid_world.update_densities(kernel)),
    );
}

criterion_group!(update_densities, bench_update_densities);
