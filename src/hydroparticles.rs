use super::units::*;
use ggez::nalgebra as na;

pub struct HydroParticles {
    pub positions: Vec<Position>,
    pub velocities: Vec<Velocity>,
}

impl HydroParticles {
    pub fn new(num_x: usize, num_y: usize) -> HydroParticles {
        let num_particles = num_x * num_y;

        let mut particles = HydroParticles {
            positions: Vec::with_capacity(num_particles),
            velocities: vec![na::zero(); num_particles],
        };

        let dist = 10.0;
        for y in 0..num_y {
            for x in 0..num_x {
                //let i = x + y * num_x;
                particles.positions.push(na::Point2::new(dist * (x as f32), dist * (y as f32)));
            }
        }
        particles
    }

    pub fn physics_step(&mut self, dt: f32) {
        let gravity = na::Vector2::new(0.0, 9.81);

        for (pos, v) in self.positions.iter_mut().zip(self.velocities.iter_mut()) {
            *v = *v + gravity * dt;
            *pos = *pos + *v * dt;
        }
    }
}
