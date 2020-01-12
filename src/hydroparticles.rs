use super::units::*;
use ggez::nalgebra as na;

pub struct HydroParticles {
    pub positions: Vec<Position>,
    pub boundary_particles: Vec<Position>, // also called "shadow particles", immovable particles used for boundaries
    pub velocities: Vec<Velocity>,
    pub smoothing_length: f32,       // typically expressed as 'h'
}

impl HydroParticles {
    pub fn new(num_x: usize, num_y: usize, smoothing_length: f32) -> HydroParticles {
        let num_particles = num_x * num_y;

        let mut particles = HydroParticles {
            positions: Vec::with_capacity(num_particles),
            boundary_particles: Vec::new(),
            velocities: vec![na::zero(); num_particles],
            smoothing_length: smoothing_length
        };

        let dist = 1.0;
        for y in 0..num_y {
            for x in 0..num_x {
                //let i = x + y * num_x;
                particles.positions.push(na::Point2::new(dist * (x as f32), dist * (y as f32)));
            }
        }
        particles
    }

    pub fn add_boundary_line(&mut self, start: Position, end: Position) {
        let distance = na::distance(&start, &end);
        let num_shadow_particles = std::cmp::max(1, (distance / self.smoothing_length) as usize);
        
        self.boundary_particles.reserve(num_shadow_particles);
        let step = (end - start) / (num_shadow_particles as f32);

        let mut pos = start;

        for _ in 0..num_shadow_particles {
            self.boundary_particles.push(pos);
            pos += step;
        }
    }

    pub fn physics_step(&mut self, dt: Time) {
        let gravity = na::Vector2::new(0.0, -9.81);

        for (pos, v) in self.positions.iter_mut().zip(self.velocities.iter_mut()) {
            *v += gravity * dt;
            *pos += *v * dt;
        }
    }
}
