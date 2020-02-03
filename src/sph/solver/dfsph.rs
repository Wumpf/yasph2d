use super::super::smoothing_kernel;
use super::super::smoothing_kernel::Kernel;
use super::super::viscositymodel::ViscosityModel;
use super::super::FluidParticleWorld;
use super::Solver;
use crate::units::*;
use ggez::nalgebra as na;
use rayon::prelude::*;

// WCSPH implementation as described in
// Divergence-Free SPH for Incompressible and Viscious Fluids
// https://animation.rwth-aachen.de/publication/051/
pub struct DFSPHSolver<TViscosityModel: ViscosityModel> {
    viscosity_model: TViscosityModel,

    density_kernel: smoothing_kernel::Poly6,
    pressure_kernel: smoothing_kernel::Spiky,

    boundary_mass_factor: Real,

    // Recomputed every frame, only here to avoid realloc.
    predicted_velocities: Vec<Vector>,
    // Recomputed every frame, but needs to be up to date at start of simulation step.
    alpha_values: Vec<Real>,
}
impl<TViscosityModel: ViscosityModel + std::marker::Sync> DFSPHSolver<TViscosityModel> {
    pub fn new(viscosity_model: TViscosityModel, smoothing_length: Real) -> DFSPHSolver<TViscosityModel> {
        DFSPHSolver {
            viscosity_model,

            density_kernel: smoothing_kernel::Poly6::new(smoothing_length),
            pressure_kernel: smoothing_kernel::Spiky::new(smoothing_length),

            boundary_mass_factor: 1000.0, // pressure & divergence solver will treat boundary particles in interactions like normal particles with this mass factor

            alpha_values: vec![],
            predicted_velocities: vec![],
        }
    }

    // computes alpha factors.
    // Note that in the paper the alpha factors contained density as well (== density / thing-we-compute-here)
    // However, all uses of the factor in the paper divide density again, so no need for having it in here in the first place!
    // (seemed to make sense for derivation though :))
    fn compute_alpha_factors(alpha_values: &mut Vec<Real>, fluid_world: &FluidParticleWorld) {
        let smoothing_length_sq = fluid_world.smoothing_length() * fluid_world.smoothing_length();
        let particle_mass = fluid_world.particle_mass();
        let boundary_particle_particle_mass = fluid_world.particle_mass();
        alpha_values
            .par_iter_mut()
            .zip(fluid_world.particles.positions.par_iter())
            .for_each(|(alpha_value, ri)| {
                let mut gradient_square_sum = 0.0;
                let mut gradient_sum: Vector = na::zero();

                for rj in fluid_world.particles.positions.iter() {
                    let ri_to_rj = rj - ri;
                    let r_sq = ri_to_rj.norm_squared();
                    if r_sq > smoothing_length_sq {
                        continue;
                    }
                    let grad_ij = fluid_world.density_kernel.gradient(ri_to_rj, r_sq, r_sq.sqrt()) * particle_mass;
                    gradient_sum += grad_ij;
                    gradient_square_sum += grad_ij.norm_squared();
                }
                for rj in fluid_world.particles.boundary_particles.iter() {
                    let ri_to_rj = rj - ri;
                    let r_sq = ri_to_rj.norm_squared();
                    if r_sq > smoothing_length_sq {
                        continue;
                    }
                    let grad_ij = fluid_world.density_kernel.gradient(ri_to_rj, r_sq, r_sq.sqrt()) * boundary_particle_particle_mass;
                    gradient_sum += grad_ij;
                    gradient_square_sum += grad_ij.norm_squared();
                }

                *alpha_value = 1.0 / (gradient_sum.norm_squared() + gradient_square_sum);
            });
    }

    fn correct_density_error() {}
}

impl<TViscosityModel: ViscosityModel + std::marker::Sync> Solver for DFSPHSolver<TViscosityModel> {
    fn simulation_step(&mut self, fluid_world: &mut FluidParticleWorld, dt: Real) {
        // ensure densities and alpha factors were initialized previously ("warmup")
        // Todo: Not happy about the way added particles are handled here. This sort of works for adding, but removing this way is impossible with this design!
        if self.alpha_values.len() != fluid_world.particles.positions.len() {
            self.alpha_values.resize(fluid_world.particles.positions.len(), 0.0 as Real);
            self.predicted_velocities.resize(fluid_world.particles.positions.len(), na::zero());

            // todo: Update only new particles
            fluid_world.update_densities();
            Self::compute_alpha_factors(&mut self.alpha_values, fluid_world);
        }

        // compute non-pressure forces (from scratch)
        let non_pressure_forces = fluid_world.gravity;

        // (optional) adapt timestep using CFL condition
        // todo

        // compute velocity prediction (from scratch, can discard forces now!)
        let force_to_particle_velocity = dt / fluid_world.particle_mass();
        self.predicted_velocities
            .par_iter_mut()
            .zip(fluid_world.particles.velocities.par_iter())
            .for_each(|(predicted_velocity, current_velocity)| {
                *predicted_velocity = non_pressure_forces * force_to_particle_velocity + current_velocity;
            });

        // density correction loop
        Self::correct_density_error();

        // advect particles
        fluid_world
            .particles
            .positions
            .par_iter_mut()
            .zip(self.predicted_velocities.par_iter())
            .for_each(|(position, predicted_velocity)| {
                *position += predicted_velocity * dt;
            });

        // recompute densities
        fluid_world.update_densities();

        // recompute alpha factors
        Self::compute_alpha_factors(&mut self.alpha_values, fluid_world);

        // divergence error loop
        // todo

        // update velocities
        std::mem::swap(&mut fluid_world.particles.velocities, &mut self.predicted_velocities);
    }
}
