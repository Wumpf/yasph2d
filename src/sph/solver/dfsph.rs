use super::super::fluidparticleworld::FluidParticleWorld;
use super::super::fluidparticleworld::Particles;
use super::super::smoothing_kernel;
use super::super::smoothing_kernel::Kernel;
use super::super::viscositymodel::ViscosityModel;
use super::Solver;
use crate::units::*;
use cgmath::prelude::*;
use rayon::prelude::*;

// WCSPH implementation as described in
// Divergence-Free SPH for Incompressible and Viscious Fluids
// https://animation.rwth-aachen.de/publication/051/
pub struct DFSPHSolver<TViscosityModel: ViscosityModel> {
    #[allow(dead_code)]
    viscosity_model: TViscosityModel,

    kernel: smoothing_kernel::CubicSpline,

    #[allow(dead_code)]
    boundary_mass_factor: Real,

    // Recomputed every frame, only here to avoid realloc.
    predicted_velocities: Vec<Vector>,
    // Recomputed every frame, only here to avoid realloc.
    predicted_densities: Vec<Real>,
    // Recomputed every frame, but needs to be up to date at start of simulation step.
    alpha_values: Vec<Real>,
}
impl<TViscosityModel: ViscosityModel + std::marker::Sync> DFSPHSolver<TViscosityModel> {
    #[allow(dead_code)]
    pub fn new(viscosity_model: TViscosityModel, smoothing_length: Real) -> DFSPHSolver<TViscosityModel> {
        DFSPHSolver {
            viscosity_model,

            kernel: smoothing_kernel::CubicSpline::new(smoothing_length),

            boundary_mass_factor: 1000.0, // pressure & divergence solver will treat boundary particles in interactions like normal particles with this mass factor

            alpha_values: vec![],
            predicted_velocities: vec![],
            predicted_densities: vec![],
        }
    }

    // computes alpha factors.
    // Note that in the paper the alpha factors contained density as well (== density / thing-we-compute-here)
    // (Note that the newer Eurographics SPH Tutorial from 2019 https://interactivecomputergraphics.github.io/SPH-Tutorial/pdf/SPH_Tutorial.pdf actually works with density-squared!)
    // However, all uses of the factor in the paper divide density again, so no need for having it in here in the first place!
    // (seemed to make sense for derivation though :))
    fn compute_alpha_factors(alpha_values: &mut Vec<Real>, fluid_world: &FluidParticleWorld, kernel: impl Kernel + std::marker::Sync) {
        const EPSILON: Real = 0e-6;
        let smoothing_length_sq = fluid_world.smoothing_length() * fluid_world.smoothing_length();
        let particle_mass = fluid_world.particle_mass();
        //let boundary_particle_particle_mass = fluid_world.particle_mass();
        alpha_values
            .par_iter_mut()
            .zip(fluid_world.particles.positions.par_iter())
            .for_each(|(alpha_value, &ri)| {
                let mut gradient_square_sum = 0.0;
                let mut gradient_sum: Vector = Zero::zero();

                Particles::foreach_neighbor_particle_noindex(
                    &fluid_world.particles.positions,
                    smoothing_length_sq,
                    ri,
                    #[inline(always)]
                    |r_sq, ri_to_rj| {
                        let grad_ij = kernel.gradient(ri_to_rj, r_sq, r_sq.sqrt()) * particle_mass;
                        gradient_sum += grad_ij;
                        gradient_square_sum += grad_ij.magnitude2();
                    },
                );
                // Particles::foreach_neighbor_particle_noindex(
                //     &fluid_world.particles.boundary_particles,
                //     smoothing_length_sq,
                //     ri,
                //     #[inline(always)]
                //     |r_sq, ri_to_rj| {
                //         let grad_ij = kernel.gradient(ri_to_rj, r_sq, r_sq.sqrt()) * boundary_particle_particle_mass;
                //         gradient_sum += grad_ij;
                //         gradient_square_sum += grad_ij.magnitude2();
                //     },
                // );

                *alpha_value = 1.0 / (gradient_sum.magnitude2() + gradient_square_sum).max(EPSILON);
            });
    }

    fn predict_densities(&mut self, dt: Real, fluid_world: &FluidParticleWorld) {
        let smoothing_length = fluid_world.smoothing_length();
        let particle_mass = fluid_world.particle_mass();
        let predicted_velocities = &self.predicted_velocities;
        let kernel = &self.kernel;

        self.predicted_densities
            .par_iter_mut()
            .zip(fluid_world.particles.densities.par_iter())
            .zip(fluid_world.particles.positions.par_iter().zip(predicted_velocities.par_iter()))
            .for_each(|((predicted_densitiy, &original_density), (&ri, &predicted_vi))| {
                let mut delta = 0.0;

                Particles::foreach_neighbor_particle(
                    &fluid_world.particles.positions,
                    smoothing_length,
                    ri,
                    #[inline(always)]
                    |j, r_sq, ri_to_rj| {
                        let delta_v = predicted_vi - predicted_velocities[j];
                        delta += delta_v.dot(kernel.gradient(ri_to_rj, r_sq, r_sq.sqrt()));
                    },
                );
                // Particles::foreach_neighbor_particle_noindex(
                //     &fluid_world.particles.boundary_particles,
                //     smoothing_length,
                //     ri,
                //     #[inline(always)]
                //     |r_sq, ri_to_rj| {
                //         let delta_v = predicted_vi;
                //         delta += delta_v.dot(&kernel.gradient(ri_to_rj, r_sq, r_sq.sqrt()));
                //     },
                // );
                *predicted_densitiy = original_density + delta * particle_mass * dt;
            });
    }

    fn update_velocity_prediction(&mut self, dt: Real, fluid_world: &FluidParticleWorld) {
        let smoothing_length = fluid_world.smoothing_length();
        let particle_mass = fluid_world.particle_mass();
        let predicted_densities = &self.predicted_densities;
        let alpha_values = &self.alpha_values;
        let reference_density = fluid_world.fluid_density();
        let inv_dt = 1.0 / dt;
        let kernel = &self.kernel;

        self.predicted_velocities.par_iter_mut().enumerate().for_each(|(i, predicted_velocity)| {
            let mut delta: Vector = Zero::zero();
            let ki = (predicted_densities[i] - reference_density) * alpha_values[i];

            Particles::foreach_neighbor_particle(
                &fluid_world.particles.positions,
                smoothing_length,
                fluid_world.particles.positions[i],
                #[inline(always)]
                |j, r_sq, ri_to_rj| {
                    let kj = (predicted_densities[j] - reference_density) * alpha_values[j];
                    // compared to k values in paper already divided with density and multiplied by dt!
                    delta += (ki + kj) * kernel.gradient(ri_to_rj, r_sq, r_sq.sqrt());
                },
            );
            // Particles::foreach_neighbor_particle_noindex(
            //     &fluid_world.particles.boundary_particles,
            //     smoothing_length,
            //     fluid_world.particles.positions[i],
            //     #[inline(always)]
            //     |r_sq, ri_to_rj| {
            //         // compared to k values in paper already divided with density and multiplied by dt!
            //         delta += inv_dt * ki * kernel.gradient(ri_to_rj, r_sq, r_sq.sqrt());
            //     }
            // );

            *predicted_velocity -= delta * inv_dt * particle_mass;
        });
    }

    fn correct_density_error(&mut self, dt: Real, fluid_world: &FluidParticleWorld) {
        // todo: proper iteration
        // todo: warmup
        for _ in 0..2 {
            self.predict_densities(dt, fluid_world);
            self.update_velocity_prediction(dt, fluid_world);
        }
    }
}

impl<TViscosityModel: ViscosityModel + std::marker::Sync> Solver for DFSPHSolver<TViscosityModel> {
    fn simulation_step(&mut self, fluid_world: &mut FluidParticleWorld, dt: Real) {
        // ensure densities and alpha factors were initialized previously ("warmup")
        // Todo: Not happy about the way added particles are handled here. This sort of works for adding, but removing this way is impossible with this design!
        if self.alpha_values.len() != fluid_world.particles.positions.len() {
            self.alpha_values.resize(fluid_world.particles.positions.len(), 0.0 as Real);
            self.predicted_velocities.resize(fluid_world.particles.positions.len(), Zero::zero());
            self.predicted_densities.resize(fluid_world.particles.positions.len(), Zero::zero());

            // todo: Update only new particles
            fluid_world.update_densities(self.kernel);
            Self::compute_alpha_factors(&mut self.alpha_values, fluid_world, self.kernel);
        }

        // compute non-pressure forces (from scratch)
        let non_pressure_forces: Vector = Vector::zero(); //fluid_world.gravity;

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
        self.correct_density_error(dt, fluid_world);

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
        fluid_world.update_densities(self.kernel);

        // recompute alpha factors
        Self::compute_alpha_factors(&mut self.alpha_values, fluid_world, self.kernel);

        // divergence error loop
        // todo

        // update velocities
        std::mem::swap(&mut fluid_world.particles.velocities, &mut self.predicted_velocities);
    }
}
