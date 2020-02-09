use crate::units::*;
use cgmath::prelude::*;
use ggez::graphics::Rect;
use rayon::prelude::*;

use super::smoothing_kernel::Kernel;

pub struct Particles {
    pub positions: Vec<Point>,
    pub velocities: Vec<Vector>,
    pub accellerations: Vec<Vector>,
    pub densities: Vec<Real>, // Local densities ρ

    pub boundary_particles: Vec<Point>, // also called "shadow particles", immovable particles used for boundaries
}

impl Particles {
    const OVERLAP_THRESHOLD: Real = 0.00001;

    #[inline(always)]
    pub(super) fn foreach_neighbor_particle(positions: &[Point], smoothing_length_sq: Real, ri: Point, mut f: impl FnMut(usize, Real, Vector) -> ()) {
        for (j, rj) in positions.iter().enumerate() {
            let ri_to_rj = rj - ri;
            let r_sq = ri_to_rj.magnitude2();
            if r_sq > smoothing_length_sq || r_sq < Self::OVERLAP_THRESHOLD {
                // Skips self and and degenerated overlaps
                continue;
            }
            f(j, r_sq, ri_to_rj);
        }
    }

    #[inline(always)]
    pub(super) fn foreach_neighbor_particle_noindex(
        positions: &[Point],
        smoothing_length_sq: Real,
        ri: Point,
        mut f: impl FnMut(Real, Vector) -> (),
    ) {
        for rj in positions.iter() {
            let ri_to_rj = rj - ri;
            let r_sq = ri_to_rj.magnitude2();
            if r_sq > smoothing_length_sq || r_sq < Self::OVERLAP_THRESHOLD {
                // Skips self and and degenerated overlaps
                continue;
            }
            f(r_sq, ri_to_rj);
        }
    }

    #[inline(always)]
    pub(super) fn foreach_neighbor_particle_compact(positions: &[Point], smoothing_length_sq: Real, ri: Point, mut f: impl FnMut(Real) -> ()) {
        for rj in positions.iter() {
            let r_sq = rj.distance2(ri);
            if r_sq > smoothing_length_sq || r_sq < Self::OVERLAP_THRESHOLD {
                // Skips self and and degenerated overlaps
                continue;
            }
            f(r_sq);
        }
    }
}

pub struct FluidParticleWorld {
    pub particles: Particles,

    smoothing_length: Real, // typically expressed as 'h'
    particle_density: Real, // #particles/m² for resting fluid
    fluid_density: Real,    // kg/m² for the resting fluid (ρ, rho)

    pub gravity: Vector, // global gravity force in m/s² (== N/kg)
}
impl FluidParticleWorld {
    pub fn new(
        smoothing_factor: Real,
        particle_density: Real, // #particles/m² for resting fluid
        fluid_density: Real,    // kg/m² for the resting fluid
    ) -> FluidParticleWorld {
        let smoothing_length = 2.0 * Self::particle_radius_from_particle_density(particle_density) * smoothing_factor;
        FluidParticleWorld {
            particles: Particles {
                positions: Vec::new(),
                velocities: Vec::new(),
                accellerations: Vec::new(),
                densities: Vec::new(),

                boundary_particles: Vec::new(),
            },

            smoothing_length,
            particle_density,
            fluid_density,

            gravity: Vector::new(0.0, -9.81),
        }
    }

    pub fn smoothing_length(&self) -> Real {
        self.smoothing_length
    }

    pub fn fluid_density(&self) -> Real {
        self.fluid_density
    }

    pub fn particle_mass(&self) -> Real {
        self.fluid_density / self.particle_density
    }

    fn particle_radius_from_particle_density(particle_density: Real) -> Real {
        // density is per m²
        0.5 / particle_density.sqrt()
    }

    fn num_particles_per_meter(&self) -> Real {
        self.particle_density.sqrt()
    }

    pub fn suggested_particle_render_radius(&self) -> Real {
        Self::particle_radius_from_particle_density(self.particle_density)
    }

    /// - `jitter`: Amount of jitter. 0 for perfect lattice. >1 and particles are no longer in a strict lattice.
    pub fn add_fluid_rect(&mut self, fluid_rect: &Rect, jitter_amount: Real) {
        // fluid_rect.w * fluid_rect.h / self.particle_density, but discretized per axis
        let num_particles_per_meter = self.num_particles_per_meter();
        let num_particles_x = std::cmp::max(1, (fluid_rect.w as Real * num_particles_per_meter) as usize);
        let num_particles_y = std::cmp::max(1, (fluid_rect.h as Real * num_particles_per_meter) as usize);
        let num_particles = num_particles_x * num_particles_y;

        self.particles.positions.reserve(num_particles);
        self.particles
            .velocities
            .resize(self.particles.velocities.len() + num_particles, Zero::zero());
        self.particles
            .densities
            .resize(self.particles.densities.len() + num_particles, Zero::zero());
        self.particles
            .accellerations
            .resize(self.particles.accellerations.len() + num_particles, Zero::zero());

        let bottom_left = Point::new(fluid_rect.x as Real, fluid_rect.y as Real);
        let step = (fluid_rect.w as Real / (num_particles_x as Real)).min(fluid_rect.h as Real / (num_particles_y as Real));
        let jitter_factor = step * jitter_amount;
        for y in 0..num_particles_y {
            for x in 0..num_particles_x {
                let jitter = (rand::random::<Vector>() * 0.5 + Vector::new(0.5, 0.5)) * jitter_factor;
                self.particles
                    .positions
                    .push(bottom_left + jitter + Vector::new(step * (x as Real), step * (y as Real)));
            }
        }
    }

    pub fn add_boundary_line(&mut self, start: Point, end: Point) {
        let distance = start.distance2(end);
        let num_particles_per_meter = self.num_particles_per_meter();
        let num_shadow_particles = std::cmp::max(1, (distance * num_particles_per_meter) as usize);
        self.particles.boundary_particles.reserve(num_shadow_particles);
        let step = (end - start) / (num_shadow_particles as Real);

        let mut pos = start;
        for _ in 0..num_shadow_particles {
            self.particles.boundary_particles.push(pos);
            pos += step;
        }
    }

    pub(super) fn update_densities(&mut self, kernel: impl Kernel + std::marker::Sync) {
        assert_eq!(self.particles.positions.len(), self.particles.densities.len());

        let mass = self.particle_mass();

        // Density contributions are symmetric, but that is hard to use in a parallel loop.
        let positions = &self.particles.positions;
        let smoothing_length_sq = self.smoothing_length * self.smoothing_length;
        let boundary_particles = &self.particles.boundary_particles;

        self.particles
            .densities
            .par_iter_mut()
            .zip(positions.par_iter())
            .for_each(|(density, ri)| {
                *density = kernel.evaluate(0.0, 0.0) * mass; // self-contribution

                Particles::foreach_neighbor_particle_compact(
                    positions,
                    smoothing_length_sq,
                    *ri,
                    #[inline(always)]
                    |r_sq| {
                        let density_contribution = kernel.evaluate(r_sq, r_sq.sqrt()) * mass;
                        *density += density_contribution;
                    },
                );
                Particles::foreach_neighbor_particle_compact(
                    boundary_particles,
                    smoothing_length_sq,
                    *ri,
                    #[inline(always)]
                    |r_sq| {
                        let density_contribution = kernel.evaluate(r_sq, r_sq.sqrt()) * mass;
                        *density += density_contribution;
                    },
                );
            });
    }
}
