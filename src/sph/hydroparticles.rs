use crate::units::*;
use ggez::graphics::Rect;
use ggez::nalgebra as na;
use rayon::prelude::*;

use super::smoothing_kernel::Kernel;
use super::smoothing_kernel as smoothing_kernel;

pub struct HydroParticles {
    pub positions: Vec<Point>,
    pub velocities: Vec<Vector>,
    pub accellerations: Vec<Vector>,
    pub densities: Vec<Real>, // Local densities ρ

    pub boundary_particles: Vec<Point>, // also called "shadow particles", immovable particles used for boundaries

    smoothing_length: Real, // typically expressed as 'h'
    particle_density: Real, // #particles/m² for resting fluid
    fluid_density: Real,    // kg/m² for the resting fluid (ρ, rho)

    density_kernel: smoothing_kernel::Poly6,

}
impl HydroParticles {
    pub fn new(
        smoothing_factor: Real,
        particle_density: Real, // #particles/m² for resting fluid
        fluid_density: Real,    // kg/m² for the resting fluid
    ) -> HydroParticles {
        let smoothing_length = 2.0 * Self::particle_radius_from_particle_density(particle_density) * smoothing_factor;
        HydroParticles {
            positions: Vec::new(),
            velocities: Vec::new(),
            accellerations: Vec::new(),
            densities: Vec::new(),

            boundary_particles: Vec::new(),

            smoothing_length: smoothing_length,
            particle_density: particle_density,
            fluid_density: fluid_density,
            
            density_kernel: smoothing_kernel::Poly6::new(smoothing_length),
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

        self.positions.reserve(num_particles);
        self.velocities.resize(self.velocities.len() + num_particles, na::zero());
        self.densities.resize(self.densities.len() + num_particles, na::zero());
        self.accellerations.resize(self.accellerations.len() + num_particles, na::zero());

        let bottom_left = Point::new(fluid_rect.x as Real, fluid_rect.y as Real);
        let step = (fluid_rect.w as Real / (num_particles_x as Real)).min(fluid_rect.h as Real / (num_particles_y as Real));
        let jitter_factor = step * jitter_amount;
        for y in 0..num_particles_y {
            for x in 0..num_particles_x {
                let jitter = (Vector::new_random() * 0.5 + Vector::new(0.5, 0.5)) * jitter_factor;
                self.positions
                    .push(bottom_left + jitter + na::Vector2::new(step * (x as Real), step * (y as Real)));
            }
        }
    }

    pub fn add_boundary_line(&mut self, start: &Point, end: &Point) {
        let distance = na::distance(start, end);
        let num_particles_per_meter = self.num_particles_per_meter();
        let num_shadow_particles = std::cmp::max(1, (distance * num_particles_per_meter) as usize);
        self.boundary_particles.reserve(num_shadow_particles);
        let step = (end - start) / (num_shadow_particles as Real);

        let mut pos = *start;
        for _ in 0..num_shadow_particles {
            self.boundary_particles.push(pos);
            pos += step;
        }
    }

    pub(crate) fn update_densities(&mut self) {
        assert_eq!(self.positions.len(), self.densities.len());

        let mass = self.particle_mass();

        // Density contributions are symmetric, but that is hard to use in a parallel loop.
        let positions = &self.positions;
        let kernel = &self.density_kernel;
        let smoothing_length_sq = self.smoothing_length * self.smoothing_length;
        let boundary_particles = &self.boundary_particles;

        self.densities.par_iter_mut().zip(positions.par_iter()).for_each(|(density, ri)| {
            *density = kernel.evaluate(0.0, 0.0) * mass; // self-contribution
            for rj in positions.iter() {
                let r_sq = na::distance_squared(ri, rj);
                if r_sq > smoothing_length_sq {
                    continue;
                }
                let density_contribution = kernel.evaluate(r_sq, r_sq.sqrt()) * mass;
                *density += density_contribution;
            }
            for rj in boundary_particles.iter() {
                let r_sq = na::distance_squared(ri, rj);
                if r_sq > smoothing_length_sq {
                    continue;
                }
                let density_contribution = kernel.evaluate(r_sq, r_sq.sqrt()) * mass;
                *density += density_contribution;
            }
        });
    }
}
