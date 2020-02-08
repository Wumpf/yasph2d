use super::super::fluidparticleworld::FluidParticleWorld;
use super::super::fluidparticleworld::Particles;
use super::super::smoothing_kernel;
use super::super::smoothing_kernel::Kernel;
use super::super::viscositymodel::ViscosityModel;
use super::Solver;
use crate::units::*;
use rayon::prelude::*;

// Solver LOOSELY based on Becker & Teschner 2007 WCSPH07
// https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
pub struct WCSPHSolver<TViscosityModel: ViscosityModel> {
    viscosity_model: TViscosityModel,
    pressure_kernel: smoothing_kernel::Spiky,
    boundary_force_factor: Real,
}
impl<TViscosityModel: ViscosityModel + std::marker::Sync> WCSPHSolver<TViscosityModel> {
    pub fn new(viscosity_model: TViscosityModel, smoothing_length: Real) -> WCSPHSolver<TViscosityModel> {
        WCSPHSolver {
            viscosity_model,
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

    fn update_accellerations(&self, fluid_world: &mut FluidParticleWorld, dt: Real) {
        let mass = fluid_world.particle_mass();

        // pressure & viscosity forces
        // According to https://www8.cs.umu.se/kurser/TDBD24/VT06/lectures/sphsurvivalkit.pdf
        // the "good way" to do symmetric forces in SPH is -m (pi + pj) / (2 * rhoj * rhoi)

        // TODO: This is just insane.
        let positions = &fluid_world.particles.positions;
        let densities = &fluid_world.particles.densities;
        let velocities = &fluid_world.particles.velocities;
        let smoothing_length_sq = fluid_world.smoothing_length() * fluid_world.smoothing_length();
        let boundary_particles = &fluid_world.particles.boundary_particles;
        let fluid_density = fluid_world.fluid_density();
        let gravity = fluid_world.gravity;

        fluid_world
            .particles
            .accellerations
            .par_iter_mut()
            .zip(velocities.par_iter())
            .zip(positions.par_iter().zip(densities.par_iter()))
            .for_each(|((accelleration, &vi), (&ri, &rhoi))| {
                *accelleration = gravity;

                let pi = Self::pressure(fluid_density, rhoi);

                // no self-contribution since vector to particle is zero (-> no pressure) and velocity difference is zero as well (-> no viscosity)
                Particles::foreach_neighbor_particle(
                    positions,
                    smoothing_length_sq,
                    ri,
                    #[inline(always)]
                    |j, r_sq, ri_to_rj| {
                        let rhoj = densities[j];
                        let pj = Self::pressure(fluid_density, rhoj);
                        let r = r_sq.sqrt();

                        // accelleration from pressure force
                        // As in "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
                        // This is a weakly compressible model (WCSPH)
                        let pressure_unsmoothed = -mass * (pi + pj) / (2.0 * rhoi * rhoj);
                        *accelleration += pressure_unsmoothed * self.pressure_kernel.gradient(-ri_to_rj, r_sq, r);

                        *accelleration += self
                            .viscosity_model
                            .compute_viscous_accelleration(dt, r_sq, r, mass, rhoj, velocities[j] - vi);
                    },
                );

                // Boundary forces as described by
                // "SPH particle boundary forces for arbitrary boundaries" by Monaghan and Kajtar 2009
                // Simple formulation found in http://www.unige.ch/math/folks/sutti/SPH_2019.pdf under 2.3.4 Radial force
                // ("SPH treatment of boundaries and application to moving objects" by Marco Sutti)
                Particles::foreach_neighbor_particle_noindex(
                    boundary_particles,
                    smoothing_length_sq,
                    ri,
                    #[inline(always)]
                    |r_sq, ri_to_rj| {
                        *accelleration -= self.boundary_force_factor * self.pressure_kernel.evaluate(r_sq, r_sq.sqrt()) / r_sq * ri_to_rj;
                    },
                );
            });
    }
}

impl<TViscosityModel: ViscosityModel + std::marker::Sync> Solver for WCSPHSolver<TViscosityModel> {
    fn simulation_step(&mut self, fluid_world: &mut FluidParticleWorld, dt: Real) {
        // leap frog integration scheme with integer steps
        // https://en.wikipedia.org/wiki/Leapfrog_integration
        for ((pos, v), a) in fluid_world
            .particles
            .positions
            .iter_mut()
            .zip(fluid_world.particles.velocities.iter_mut())
            .zip(fluid_world.particles.accellerations.iter())
        {
            *pos += *v * dt + a * (0.5 * dt * dt);
            // partial update of velocity.
            // what we want is v_new = v_old + 0.5 (a_old + a_new) () t
            // spit it to: v_almostnew = v_old + 0.5 * a_old * t + 0.5 * a_new * t
            *v += 0.5 * dt * a;
        }

        fluid_world.update_densities();
        self.update_accellerations(fluid_world, dt);

        // part 2 of leap frog integration. Finish updating velocity.
        for (v, a) in fluid_world
            .particles
            .velocities
            .iter_mut()
            .zip(fluid_world.particles.accellerations.iter())
        {
            *v += 0.5 * dt * a;
        }
    }
}
