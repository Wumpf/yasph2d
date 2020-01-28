use crate::units::*;
use ggez::graphics::Rect;
use ggez::nalgebra as na;
use rayon::prelude::*;

use super::smoothing_kernel as smoothing_kernel;
use super::smoothing_kernel::*;
use super::solver::Solver;
use super::viscositymodel::ViscosityModel;


// Solver LOOSELY based on Becker & Teschner 2007 WCSPH07
// https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
pub struct WCSPHSolver<TViscosityModel: ViscosityModel> {
    viscosity_model: TViscosityModel,

    density_kernel: smoothing_kernel::Poly6,
    pressure_kernel: smoothing_kernel::Spiky,

    boundary_force_factor: Real,
}
impl<TViscosityModel: ViscosityModel + std::marker::Sync> WCSPHSolver<TViscosityModel> {
    pub fn new(viscosity_model: TViscosityModel, smoothing_length: Real) -> WCSPHSolver<TViscosityModel> {
        WCSPHSolver {
            viscosity_model: viscosity_model,

            density_kernel: smoothing_kernel::Poly6::new(smoothing_length),
            pressure_kernel: smoothing_kernel::Spiky::new(smoothing_length),

            boundary_force_factor: 10.0, // (expected accelleration) / (spacing ratio of boundary / normal particles)
        }
    }

    // Equation of State (EOS)
    fn pressure(fluid_density: Real, local_density: Real) -> Real {
        // Isothermal gas (== Tait equation for water-like fluids with gamma 1)
        //self.fluid_speedofsound_sq * (local_density - self.fluid_density)
        // Tait equation as in Becker & Teschner 2007 WCSPH07
        2.0 * ((local_density / fluid_density).powi(7) - 1.0) // 2.0 is a hack
    }

    fn update_accellerations(&self, particles: &mut HydroParticles, dt: Real) {
        let mass = particles.particle_mass();

        let gravity = Vector::new(0.0, -9.81);

        // pressure & viscosity forces
        // According to https://www8.cs.umu.se/kurser/TDBD24/VT06/lectures/sphsurvivalkit.pdf
        // the "good way" to do symmetric forces in SPH is -m (pi + pj) / (2 * rhoj * rhoi)

        // TODO: This is just insane.
        let positions = &particles.positions;
        let densities = &particles.densities;
        let velocities = &particles.velocities;
        let pressure_kernel = &self.pressure_kernel;
        let density_kernel = &self.density_kernel;
        let smoothing_length_sq = particles.smoothing_length_sq;
        let boundary_particles = &particles.boundary_particles;
        let fluid_density = particles.fluid_density;

        particles
            .accellerations
            .par_iter_mut()
            .zip(positions.par_iter())
            .zip(densities.par_iter())
            .zip(velocities.par_iter())
            .for_each(|(((accelleration, ri), rhoi), vi)| {
                *accelleration = gravity;

                let pi = Self::pressure(fluid_density, *rhoi);

                // no self-contribution since vector to particle is zero (-> no pressure) and velocity difference is zero as well (-> no viscosity)
                for ((rj, rhoj), vj) in positions.iter().zip(densities.iter()).zip(velocities.iter()) {
                    let ri_rj = ri - rj;
                    let r_sq = ri_rj.norm_squared();
                    if r_sq > smoothing_length_sq {
                        continue;
                    }
                    let r = r_sq.sqrt();

                    let pj = Self::pressure(fluid_density, *rhoj);

                    // accelleration from pressure force
                    // As in "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
                    // This is a weakly compressible model (WCSPH)
                    let pressure_unsmoothed = -mass * (pi + pj) / (2.0 * rhoi * rhoj);
                    *accelleration += pressure_unsmoothed * pressure_kernel.gradient(ri_rj, r_sq, r);

                    *accelleration += self.viscosity_model.compute_viscous_accelleration(dt, r_sq, r, mass, *rhoj, vj - vi);
                }

                // Boundary forces as described by
                // "SPH particle boundary forces for arbitrary boundaries" by Monaghan and Kajtar 2009
                // Simple formulation found in http://www.unige.ch/math/folks/sutti/SPH_2019.pdf under 2.3.4 Radial force
                // ("SPH treatment of boundaries and application to moving objects" by Marco Sutti)
                for rj in boundary_particles.iter() {
                    let ri_rj = ri - rj;
                    let r_sq = ri_rj.norm_squared();
                    if r_sq > smoothing_length_sq {
                        continue;
                    }
                    *accelleration += self.boundary_force_factor * density_kernel.evaluate(r_sq, r_sq.sqrt()) / r_sq * ri_rj;
                }
            });
    }
}

impl<TViscosityModel: ViscosityModel + std::marker::Sync> Solver for WCSPHSolver<TViscosityModel> {
    fn simulation_step(&self, particles: &mut HydroParticles, dt: Real) {
        assert_eq!(particles.positions.len(), particles.velocities.len());
        assert_eq!(particles.positions.len(), particles.accellerations.len());

        // leap frog integration scheme with integer steps
        // https://en.wikipedia.org/wiki/Leapfrog_integration
        for ((pos, v), a) in particles
            .positions
            .iter_mut()
            .zip(particles.velocities.iter_mut())
            .zip(particles.accellerations.iter())
        {
            *pos += *v * dt + a * (0.5 * dt * dt);
            // partial update of velocity.
            // what we want is v_new = v_old + 0.5 (a_old + a_new) () t
            // spit it to: v_almostnew = v_old + 0.5 * a_old * t + 0.5 * a_new * t
            *v += 0.5 * dt * a;
        }

        particles.update_densities();
        self.update_accellerations(particles, dt);

        // part 2 of leap frog integration. Finish updating velocity.
        for (v, a) in particles.velocities.iter_mut().zip(particles.accellerations.iter()) {
            *v += 0.5 * dt * a;
        }
    }
}

pub struct HydroParticles {
    pub positions: Vec<Point>,
    pub velocities: Vec<Vector>,
    pub accellerations: Vec<Vector>,

    pub boundary_particles: Vec<Point>, // also called "shadow particles", immovable particles used for boundaries

    smoothing_length_sq: Real, // typically expressed as 'h'
    smoothing_length: Real,
    particle_density: Real, // #particles/m² for resting fluid
    fluid_density: Real,    // kg/m² for the resting fluid (ρ, rho)

    density_kernel: smoothing_kernel::Poly6,

    densities: Vec<Real>, // Local densities ρ
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

            boundary_particles: Vec::new(),

            smoothing_length: smoothing_length,
            smoothing_length_sq: smoothing_length * smoothing_length,
            particle_density: particle_density,
            fluid_density: fluid_density,

            density_kernel: smoothing_kernel::Poly6::new(smoothing_length),

            densities: Vec::new(),
        }
    }

    pub fn smoothing_length(&self) -> Real {
        self.smoothing_length
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

    fn update_densities(&mut self) {
        assert_eq!(self.positions.len(), self.densities.len());

        let mass = self.particle_mass();

        // Density contributions are symmetric, but that is hard to use in a parallel loop.
        let positions = &self.positions;
        let kernel = &self.density_kernel;
        let smoothing_length_sq = self.smoothing_length_sq;
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
