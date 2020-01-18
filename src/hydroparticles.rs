use super::units::*;
use ggez::graphics::Rect;
use ggez::nalgebra as na;

use crate::smoothing_kernel;
use crate::smoothing_kernel::Kernel;

pub struct HydroParticles {
    pub positions: Vec<Position>,
    pub velocities: Vec<Velocity>,

    pub boundary_particles: Vec<Position>, // also called "shadow particles", immovable particles used for boundaries

    smoothing_length_sq: f32, // typically expressed as 'h'
    particle_density: f32,    // #particles/m² for resting fluid
    fluid_density: f32,       // kg/m² for the resting fluid (ρ, rho)
    fluid_machnumber_sq: f32, // speed of sound in this fluid squared
    fluid_viscosity: f32,     // the dynamic viscosity of this fluid in Pa*s (μ, mu)

    density_kernel: smoothing_kernel::Poly6,
    pressure_kernel: smoothing_kernel::Spiky,
    viscosity_kernel: smoothing_kernel::Viscosity,

    boundary_force_factor: f32,

    densities: Vec<f32>, // Local densities ρ
    pub forces: Vec<Direction>,
}

impl HydroParticles {
    pub fn new(
        smoothing_factor: f32,
        particle_density: f32, // #particles/m² for resting fluid
        fluid_density: f32,    // kg/m² for the resting fluid
        fluid_machnumber: f32, // speed of sound in this fluid
        fluid_viscosity: f32,  // the dynamic viscosity of this fluid in Pa*s (μ, mu)
    ) -> HydroParticles {
        let smoothing_length = 2.0 * Self::particle_radius_from_particle_density(particle_density) * smoothing_factor;
        HydroParticles {
            positions: Vec::new(),
            velocities: Vec::new(),
            boundary_particles: Vec::new(),

            smoothing_length_sq: smoothing_length * smoothing_length,
            particle_density: particle_density,
            fluid_density: fluid_density,
            fluid_machnumber_sq: fluid_machnumber * fluid_machnumber,
            fluid_viscosity: fluid_viscosity,

            density_kernel: smoothing_kernel::Poly6::new(smoothing_length),
            pressure_kernel: smoothing_kernel::Spiky::new(smoothing_length),
            viscosity_kernel: smoothing_kernel::Viscosity::new(smoothing_length),

            boundary_force_factor: 10.0, // (expected accelleration) / (spacing ratio of boundary / normal particles)

            densities: Vec::new(),
            forces: Vec::new(),
        }
    }

    pub fn particle_mass(&self) -> f32 {
        self.fluid_density / self.particle_density
    }

    fn pressure(&self, local_density: f32) -> f32 {
        // todo: Learn more about other formulations. Tait equation?
        // This one is from http://www8.cs.umu.se/kurser/TDBD24/VT07/lectures/Lecture10.pdf
        self.fluid_machnumber_sq * (local_density - self.fluid_density)
    }

    fn particle_radius_from_particle_density(particle_density: f32) -> f32 {
        // density is per m²
        0.5 / particle_density.sqrt()
    }

    fn num_particles_per_meter(&self) -> f32 {
        self.particle_density.sqrt()
    }

    pub fn suggested_particle_render_radius(&self) -> f32 {
        Self::particle_radius_from_particle_density(self.particle_density)
    }

    pub fn add_fluid_rect(&mut self, fluid_rect: &Rect) {
        // fluid_rect.w * fluid_rect.h / self.particle_density, but discretized per axis
        let num_particles_per_meter = self.num_particles_per_meter();
        let num_particles_x = std::cmp::max(1, (fluid_rect.w * num_particles_per_meter) as usize);
        let num_particles_y = std::cmp::max(1, (fluid_rect.h * num_particles_per_meter) as usize);
        let num_particles = num_particles_x * num_particles_y;

        self.positions.reserve(num_particles);
        self.velocities.resize(self.velocities.len() + num_particles, na::zero());
        self.densities.resize(self.densities.len() + num_particles, na::zero());
        self.forces.resize(self.forces.len() + num_particles, na::zero());

        let bottom_left = Position::new(fluid_rect.x, fluid_rect.y);
        let step_x = fluid_rect.w / (num_particles_x as f32);
        let step_y = fluid_rect.h / (num_particles_y as f32);
        for y in 0..num_particles_y {
            for x in 0..num_particles_x {
                self.positions
                    .push(bottom_left + na::Vector2::new(step_x * (x as f32), step_y * (y as f32)));
            }
        }
    }

    pub fn add_boundary_line(&mut self, start: &Position, end: &Position) {
        let distance = na::distance(start, end);
        let num_particles_per_meter = self.num_particles_per_meter();
        let num_shadow_particles = std::cmp::max(1, (distance * num_particles_per_meter) as usize);
        self.boundary_particles.reserve(num_shadow_particles);
        let step = (end - start) / (num_shadow_particles as f32);

        let mut pos = *start;
        for _ in 0..num_shadow_particles {
            self.boundary_particles.push(pos);
            pos += step;
        }
    }

    fn update_densities(&mut self) {
        assert_eq!(self.positions.len(), self.densities.len());

        for density in self.densities.iter_mut() {
            *density = 1.0e-10; // avoid dealing with 0 density
        }

        let mass = self.particle_mass();

        // Density contributions are symmetric!
        for (i, ri) in self.positions.iter().enumerate() {
            self.densities[i] += self.density_kernel.evaluate(0.0) * mass; // self-contribution
            for (j, rj) in self.positions.iter().enumerate().skip(i + 1) {
                let r_sq = na::distance_squared(ri, rj);
                if r_sq > self.smoothing_length_sq {
                    continue;
                }
                let density_contribution = self.density_kernel.evaluate(r_sq) * mass;
                self.densities[i] += density_contribution;
                self.densities[j] += density_contribution;
            }
            for rj in self.boundary_particles.iter() {
                let r_sq = na::distance_squared(ri, rj);
                if r_sq > self.smoothing_length_sq {
                    continue;
                }
                let density_contribution = self.density_kernel.evaluate(r_sq) * mass;
                self.densities[i] += density_contribution;
            }
        }
    }

    pub fn physics_step(&mut self, dt: f32) {
        assert_eq!(self.positions.len(), self.velocities.len());
        assert_eq!(self.positions.len(), self.forces.len());

        self.update_densities();

        let mass = self.particle_mass();
        let mass_sq = mass * mass;

        let gravity = na::Vector2::new(0.0, -9.81) * 0.01;
        for f in self.forces.iter_mut() {
            *f = gravity * mass;
        }

        // pressure forces
        // It's done in a symmetric way
        // According to https://www8.cs.umu.se/kurser/TDBD24/VT06/lectures/sphsurvivalkit.pdf
        // the "good way" to do symmetric forces in SPH is -m (pi + pj) / (2 * rhoj * rhoi)
        for (i, (ri, rhoi)) in self.positions.iter().zip(self.densities.iter()).enumerate() {
            let pi = self.pressure(*rhoi);

            // no self-contribution since vector to particle is zero (-> no pressure) and velocity difference is zero as well (-> no viscosity)
            for (j, (rj, rhoj)) in self.positions.iter().zip(self.densities.iter()).enumerate().skip(i + 1) {
                let ri_rj = ri - rj;
                let r_sq = ri_rj.norm_squared();
                if r_sq > self.smoothing_length_sq {
                    continue;
                }

                let pj = self.pressure(*rhoj);

                // ... also conveniently gives us the same term for i and j then!
                // (which makes sense, since interaction of two particles should always be symmetric)

                // accelleration from pressure force
                let fpressure_unsmoothed = -mass_sq * (pi + pj) / (2.0 * rhoi * rhoj);
                assert!(!fpressure_unsmoothed.is_nan());
                assert!(!fpressure_unsmoothed.is_infinite());
                let fpressure = fpressure_unsmoothed * self.pressure_kernel.gradient(ri_rj, r_sq);

                // accelleration from viscosity force
                let velocitydiff = self.velocities[j] - self.velocities[i];
                let fviscosity = self.fluid_viscosity * mass_sq * self.viscosity_kernel.laplacian(r_sq) / (rhoi * rhoj) * velocitydiff;

                let ftotal = fpressure + fviscosity;

                self.forces[i] += ftotal;
                self.forces[j] -= ftotal;
            }

            // Boundary forces as described by
            // "SPH particle boundary forces for arbitrary bound-aries" by Monaghan and Kajtar 2009
            // Simple formulation found in http://www.unige.ch/math/folks/sutti/SPH_2019.pdf under 2.3.4 Radial force
            // ("SPH treatment of boundaries and application to moving objects" by Marco Sutti)
            for rj in self.boundary_particles.iter() {
                let ri_rj = ri - rj;
                let r_sq = ri_rj.norm_squared();
                if r_sq > self.smoothing_length_sq {
                    continue;
                }
                self.forces[i] += mass * self.boundary_force_factor * self.density_kernel.evaluate(r_sq) / r_sq * ri_rj;
            }
        }

        for ((pos, v), f) in self.positions.iter_mut().zip(self.velocities.iter_mut()).zip(self.forces.iter()) {
            // todo izip!
            *v += f / mass * dt;
            *pos += *v * dt;
        }
    }
}
