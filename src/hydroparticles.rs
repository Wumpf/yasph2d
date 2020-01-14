use super::units::*;
use ggez::graphics::Rect;
use ggez::nalgebra as na;

use crate::smoothing_kernel::{Kernel, Poly6};

pub struct HydroParticles {
    pub positions: Vec<Position>,
    pub velocities: Vec<Velocity>,

    pub boundary_particles: Vec<Position>, // also called "shadow particles", immovable particles used for boundaries

    smoothing_length: f32, // typically expressed as 'h'
    particle_density: f32, // #particles/m² for resting fluid
    fluid_density: f32,    // kg/m² for the resting fluid

    pressure_kernel: Poly6,

    densities: Vec<f32>,
}

impl HydroParticles {
    pub fn new(
        smoothing_length: f32,
        particle_density: f32, // #particles/m² for resting fluid
        fluid_density: f32,    // kg/m² for the resting fluid
    ) -> HydroParticles {
        HydroParticles {
            positions: Vec::new(),
            velocities: Vec::new(),
            boundary_particles: Vec::new(),
            smoothing_length: smoothing_length,
            particle_density: particle_density,
            fluid_density: fluid_density,
            pressure_kernel: Poly6::new(smoothing_length),
            densities: Vec::new(),
        }
    }

    // fn particle_mass(&self) -> f32 {
    //     self.particle_density / self.fluid_density
    // }

    fn num_particles_per_meter(&self) -> f32 {
        self.particle_density.sqrt()
    }

    pub fn suggested_particle_render_radius(&self) -> f32 {
        0.5 / self.num_particles_per_meter()
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

    // fn update_densities(&mut self) {
    //     assert_eq!(self.positions.len(), self.densities.len());

    //     for density in self.densities.iter_mut() {
    //         self.pressure_kernel.evaluate(1.0);
    //     }
    // }

    pub fn physics_step(&mut self, dt: f32) {
        assert_eq!(self.positions.len(), self.velocities.len());

        let gravity = na::Vector2::new(0.0, -9.81);

        for (pos, v) in self.positions.iter_mut().zip(self.velocities.iter_mut()) {
            *pos += *v * dt;
            *v += gravity * dt;
        }
    }
}
