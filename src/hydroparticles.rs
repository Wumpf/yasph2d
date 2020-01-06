use ggez::nalgebra as na;

pub struct HydroParticles {
    pub positions: Vec<na::Point2<f32>>,
}

impl HydroParticles {
    pub fn new(num_x: usize, num_y: usize) -> HydroParticles {
        let num_particles = num_x * num_y;
        let mut particles = HydroParticles {
            positions: Vec::with_capacity(num_particles),
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
        for pos in &mut self.positions {
            pos.x += dt * 100.0;
        }
    }
}
