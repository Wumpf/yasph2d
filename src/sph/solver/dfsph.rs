use super::super::fluidparticleworld::FluidParticleWorld;
use super::super::smoothing_kernel;
use super::super::smoothing_kernel::Kernel;
use super::super::timemanager::TimeManager;
use super::super::viscositymodel::ViscosityModel;
use super::Solver;
use crate::units::*;
use cgmath::prelude::*;
use rayon::prelude::*;

// WCSPH implementation as described in
// Divergence-Free SPH for Incompressible and Viscious Fluids
// https://animation.rwth-aachen.de/publication/051/
pub struct DFSPHSolver<TViscosityModel: ViscosityModel> {
    viscosity_model: TViscosityModel,

    kernel: smoothing_kernel::CubicSpline,

    // Not using a boundary mass/force factor as in our WCSPH, since solver usually makes sure particles don't pass each other.
    //boundary_mass_factor: Real,

    // Max density error. In relative density deviation per second - 0.01 means 1% density deviation per second.
    max_avg_density_error: Real,
    // Maximum number of pressure solver iterations
    max_num_density_correction_iterations: usize,

    // Max divergenc error. In relative density deviation per second - 0.01 means 1% density deviation per second.
    max_divergence_error: Real,
    // Maximum number of divergence minimizer iterations
    max_num_divergence_correction_iterations: usize,

    // Recomputed every frame, but needs to be up to date at start of simulation step.
    alpha_values: Vec<Real>,
}
impl<TViscosityModel: ViscosityModel + std::marker::Sync> DFSPHSolver<TViscosityModel> {
    pub fn new(viscosity_model: TViscosityModel, smoothing_length: Real) -> DFSPHSolver<TViscosityModel> {
        DFSPHSolver {
            viscosity_model,

            kernel: smoothing_kernel::CubicSpline::new(smoothing_length),

            max_avg_density_error: 0.01 / 100.0, // 0.1% deviation per second.
            max_num_density_correction_iterations: 200,

            max_divergence_error: 0.1 / 100.0, // 1.0% deviation per second.
            max_num_divergence_correction_iterations: 400,

            alpha_values: vec![],
        }
    }

    // computes alpha factors.
    // Note that in the paper the alpha factors contained density as well (== density / thing-we-compute-here)
    // (Note that the newer Eurographics SPH Tutorial from 2019 https://interactivecomputergraphics.github.io/SPH-Tutorial/pdf/SPH_Tutorial.pdf actually works with density-squared!)
    // However, all uses of the factor in the paper divide density again, so no need for having it in here in the first place!
    // (seemed to make sense for derivation though :))
    fn compute_alpha_factors(alpha_values: &mut Vec<Real>, fluid_world: &FluidParticleWorld, kernel: impl Kernel + std::marker::Sync) {
        microprofile::scope!("DFSPHSolver", "compute_alpha_factors");
        const EPSILON: Real = 1e-6;
        let particle_mass = fluid_world.properties.particle_mass();
        let particles = &fluid_world.particles;
        alpha_values
            .par_iter_mut()
            .zip(fluid_world.particles.positions.par_iter())
            .enumerate()
            .for_each(|(i, (alpha_value, &ri))| {
                // self contribution is zero since gradient to self is zero
                let mut gradient_square_sum = 0.0;
                let mut gradient_sum = Vector::zero();
                let i = i as u32;
                particles.foreach_neighbor_particle(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.positions[j as usize];
                        let grad_ij = kernel.gradient_from_positions(ri, pos_j) * particle_mass;
                        gradient_sum += grad_ij;
                        gradient_square_sum += grad_ij.magnitude2();
                    },
                );
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.boundary_particles[j as usize];
                        let grad_ij = kernel.gradient_from_positions(ri, pos_j) * particle_mass;
                        gradient_sum += grad_ij;
                        gradient_square_sum += grad_ij.magnitude2();
                    },
                );

                *alpha_value = 1.0 / (gradient_sum.magnitude2() + gradient_square_sum).max(EPSILON);
                // todo?
            });
    }

    fn compute_density_error(&self, dt: Real, fluid_world: &FluidParticleWorld, velocities: &[Vector], density_error: &mut [Real]) {
        microprofile::scope!("DFSPHSolver", "compute_density_error");
        let particle_mass = fluid_world.properties.particle_mass();
        let particles = &fluid_world.particles;
        let reference_density = fluid_world.properties.fluid_density();
        density_error
            .par_iter_mut()
            .zip((&fluid_world.particles.densities, &fluid_world.particles.positions, velocities).into_par_iter())
            .enumerate()
            .for_each(|(i, (density_error_i, (&original_density, &pos_i, &velocity_vi)))| {
                let mut delta = 0.0; // gradient to self is zero.
                let i = i as u32;
                particles.foreach_neighbor_particle(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.positions[j as usize];
                        let delta_v = velocity_vi - velocities[j as usize];
                        delta += delta_v.dot(self.kernel.gradient_from_positions(pos_i, pos_j));
                    },
                );
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.boundary_particles[j as usize];
                        let delta_v = velocity_vi;
                        delta += delta_v.dot(self.kernel.gradient_from_positions(pos_i, pos_j));
                    },
                );
                *density_error_i = original_density + delta * particle_mass * dt;

                // ignore loss of density
                *density_error_i = reference_density.max(*density_error_i) - reference_density;
            });
    }

    fn correct_velocity_with_density_error(&self, dt: Real, fluid_world: &FluidParticleWorld, velocities: &mut [Vector], density_error: &[Real]) {
        microprofile::scope!("DFSPHSolver", "correct_velocity_with_density_error");
        let particle_mass = fluid_world.properties.particle_mass();
        let particles = &fluid_world.particles;
        let inv_dt = 1.0 / dt;
        let kernel = &self.kernel;

        velocities
            .par_iter_mut()
            .zip((&fluid_world.particles.positions, density_error, &self.alpha_values).into_par_iter())
            .enumerate()
            .for_each(|(i, (predicted_velocity, (&ri, density_error_i, alpha_i)))| {
                let mut delta: Vector = Zero::zero(); // gradient to self is zero.
                let ki = density_error_i * alpha_i;

                // compared to k values in paper already divided with density
                // collapsing divition of dt² with multiply later -> divide delta with dt

                let i = i as u32;
                particles.foreach_neighbor_particle(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.positions[j as usize];
                        let kj = density_error[j as usize] * self.alpha_values[j as usize];
                        delta += (ki + kj) * kernel.gradient_from_positions(ri, pos_j);
                    },
                );
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        // compared to k values in paper already divided with density and multiplied with dt²!
                        let pos_j = particles.boundary_particles[j as usize];
                        delta += ki * kernel.gradient_from_positions(ri, pos_j);
                    },
                );

                *predicted_velocity -= inv_dt * delta * particle_mass;
            });
    }

    fn correct_density_error(&mut self, dt: Real, fluid_world: &mut FluidParticleWorld, velocities: &mut [Vector]) {
        // todo: warmup & general use of values from previous frame
        microprofile::scope!("DFSPHSolver", "correct_density_error");

        let mut _density_error = fluid_world.scratch_buffers.get_buffer_real(fluid_world.particles.positions.len());
        let density_error = &mut _density_error.buffer;

        let mut num_iter = 0;
        loop {
            microprofile::scope!("DFSPHSolver", "density_iteration");

            self.compute_density_error(dt, fluid_world, velocities, density_error);
            self.correct_velocity_with_density_error(dt, fluid_world, velocities, density_error);
            num_iter += 1;

            let avg_density_error: Real = density_error.par_iter().sum::<Real>() / density_error.len() as Real;
            let relative_density_error = avg_density_error / fluid_world.properties.fluid_density();
            assert!(avg_density_error.is_finite());

            // error is expressed relative to fluid density and time!
            if relative_density_error * dt < self.max_avg_density_error {
                // println!(
                //     "Density error correction finished after {} steps. Density error was {}, that is {}% relative error per second. Target was {}% per second",
                //     num_iter,
                //     avg_density_error,
                //     relative_density_error * dt * 100.0,
                //     self.max_avg_density_error * 100.0,
                // );
                break;
            }
            if num_iter > self.max_num_density_correction_iterations {
                println!(
                    "Density error correction canceled after {} steps. Density error was {}, that is {}% relative error per second. Target was {}% per second",
                    num_iter,
                    avg_density_error,
                    relative_density_error * dt * 100.0,
                    self.max_avg_density_error * 100.0,
                );
                break;
            }
        }
    }

    // todo: this is almost identical to compute_density_error, use this fact!
    fn compute_density_change(&self, fluid_world: &FluidParticleWorld, velocities: &[Vector], density_change: &mut [Real]) {
        microprofile::scope!("DFSPHSolver", "compute_density_change");
        let particle_mass = fluid_world.properties.particle_mass();
        let particles = &fluid_world.particles;
        density_change
            .par_iter_mut()
            .zip((&fluid_world.particles.positions, velocities).into_par_iter())
            .enumerate()
            .for_each(|(i, (density_change_i, (&ri, &velocity_vi)))| {
                let i = i as u32;

                // particle deficiency?
                if particles.num_total_neighbors(i) < 5 {
                    *density_change_i = 0.0;
                    return;
                }

                let mut delta = 0.0; // gradient to self is zero.
                particles.foreach_neighbor_particle(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.positions[j as usize];
                        let delta_v = velocity_vi - velocities[j as usize];
                        delta += delta_v.dot(self.kernel.gradient_from_positions(ri, pos_j));
                    },
                );
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.boundary_particles[j as usize];
                        let delta_v = velocity_vi;
                        delta += delta_v.dot(self.kernel.gradient_from_positions(ri, pos_j));
                    },
                );
                *density_change_i = delta * particle_mass;
                *density_change_i = density_change_i.max(0.0); // todo: what's the explanation for this?
            });
    }

    // todo: this is almost identical to correct_velocity_with_density_error, use this fact!
    fn correct_velocity_with_divergence_error(&self, fluid_world: &FluidParticleWorld, velocities: &mut [Vector], density_change: &[Real]) {
        microprofile::scope!("DFSPHSolver", "correct_velocity_with_divergence_error");
        let particle_mass = fluid_world.properties.particle_mass();
        let particles = &fluid_world.particles;
        let kernel = &self.kernel;

        velocities
            .par_iter_mut()
            .zip((&fluid_world.particles.positions, density_change, &self.alpha_values).into_par_iter())
            .enumerate()
            .for_each(|(i, (predicted_velocity, (&ri, density_change_i, alpha_i)))| {
                let mut delta: Vector = Zero::zero(); // gradient to self is zero.
                let ki = density_change_i * alpha_i;

                // compared to k values in paper already divided with density
                // collapsing divition of dt with multiply later -> nothing!

                let i = i as u32;
                particles.foreach_neighbor_particle(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.positions[j as usize];
                        let kj = density_change[j as usize] * self.alpha_values[j as usize];
                        delta += (ki + kj) * kernel.gradient_from_positions(ri, pos_j);
                    },
                );
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        let pos_j = particles.boundary_particles[j as usize];
                        delta += ki * kernel.gradient_from_positions(ri, pos_j);
                    },
                );

                *predicted_velocity -= delta * particle_mass;
            });
    }

    fn correct_divergence_error(&mut self, dt: Real, fluid_world: &mut FluidParticleWorld, velocities: &mut [Vector]) {
        // todo: warmup & general use of values from previous frame
        microprofile::scope!("DFSPHSolver", "correct_divergence_error");

        // Relationship between density change and divergence:
        // https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics#Operators
        // Divergence = densitychange / density

        let mut _density_change = fluid_world.scratch_buffers.get_buffer_real(fluid_world.particles.positions.len());
        let density_change = &mut _density_change.buffer;

        let mut num_iter = 0;
        loop {
            microprofile::scope!("DFSPHSolver", "density_change_iteration");

            self.compute_density_change(fluid_world, velocities, density_change);
            self.correct_velocity_with_divergence_error(fluid_world, velocities, density_change);
            num_iter += 1;

            let avg_divergence: Real =
                density_change.par_iter().sum::<Real>() / density_change.len() as Real / fluid_world.properties.fluid_density();
            assert!(avg_divergence.is_finite());

            // error is expressed relative to time
            if avg_divergence * dt < self.max_divergence_error {
                // println!(
                //     "Divergence error correction finished after {} steps. Avg divergence was {}, that is {}% per second. Target was {}% per second",
                //     num_iter,
                //     avg_divergence,
                //     avg_divergence * dt * 100.0,
                //     self.max_divergence_error * 100.0,
                // );
                break;
            }
            if num_iter > self.max_num_divergence_correction_iterations {
                println!(
                    "Divergence error correction canceled after {} steps. Avg divergence was {}, that is {}% per second. Target was {}% per second",
                    num_iter,
                    avg_divergence,
                    avg_divergence * dt * 100.0,
                    self.max_divergence_error * 100.0,
                );
                break;
            }
        }
    }
}

impl<TViscosityModel: ViscosityModel + std::marker::Sync> Solver for DFSPHSolver<TViscosityModel> {
    fn clear_cached_data(&mut self) {
        self.alpha_values.clear();
    }

    fn simulation_step(&mut self, fluid_world: &mut FluidParticleWorld, time_manager: &mut TimeManager) {
        microprofile::scope!("DFSPHSolver", "simulation_step");

        // ensure densities and alpha factors were initialized previously ("warmup")
        // Todo: Not happy about the way added particles are handled here. This sort of works for adding, but removing this way is impossible with this design!
        if self.alpha_values.len() != fluid_world.particles.positions.len() {
            self.alpha_values.resize(fluid_world.particles.positions.len(), 0.0 as Real);

            // todo: Update only new particles.. HOW? better would be to only effectively add later
            fluid_world.update_neighborhood_datastructure(Vec::new(), vec![&mut self.alpha_values]);
            fluid_world.update_densities(self.kernel);
            Self::compute_alpha_factors(&mut self.alpha_values, fluid_world, self.kernel);
        }

        let mut _predicted_velocities = fluid_world.scratch_buffers.get_buffer_vector(fluid_world.particles.positions.len());
        let predicted_velocities = &mut _predicted_velocities.buffer;

        // compute non-pressure forces (from scratch)
        {
            let mut accellerations = fluid_world.scratch_buffers.get_buffer_vector(fluid_world.particles.positions.len());

            {
                microprofile::scope!("DFSPHSolver", "non-pressure forces");

                let non_pressure_forces: Vector = fluid_world.gravity * fluid_world.properties.particle_mass();
                let particle_mass = fluid_world.properties.particle_mass();
                let non_pressure_accelleration = non_pressure_forces / particle_mass;
                let dt = time_manager.timestep();
                let particles = &fluid_world.particles;
                let viscosity_model = &self.viscosity_model;
                accellerations
                    .buffer
                    .par_iter_mut()
                    .zip((&particles.positions, &particles.velocities).into_par_iter())
                    .enumerate()
                    .for_each(|(i, (a, (&ri, &vi)))| {
                        // forces
                        *a = non_pressure_accelleration;

                        // viscosity
                        particles.foreach_neighbor_particle(
                            i as u32,
                            #[inline(always)]
                            |j| {
                                let j = j as usize;
                                let r_sq = ri.distance2(particles.positions[j]);
                                *a += viscosity_model.compute_viscous_accelleration(
                                    dt,
                                    r_sq,
                                    r_sq.sqrt(),
                                    particle_mass,
                                    particles.densities[j],
                                    particles.velocities[j] - vi,
                                );
                            },
                        );
                    });
            }

            // update timestep
            {
                microprofile::scope!("DFSPHSolver", "update timestep");
                let dt = time_manager.timestep();
                let mut max_velocity_sq: Real = 0.0;
                for (v, a) in fluid_world.particles.velocities.iter().zip(accellerations.buffer.iter()) {
                    max_velocity_sq = max_velocity_sq.max((v + a * dt).magnitude2());
                }
                time_manager.update_timestep(fluid_world.properties.particle_radius() * 2.0, max_velocity_sq.sqrt());
            }

            // predict velocity
            {
                microprofile::scope!("DFSPHSolver", "velocity prediction");
                let dt = time_manager.timestep();
                for (predicted_velocity, (v, a)) in predicted_velocities
                    .iter_mut()
                    .zip(fluid_world.particles.velocities.iter().zip(accellerations.buffer.iter()))
                {
                    *predicted_velocity = v + a * dt;
                }
            }
        }
        let dt = time_manager.timestep();

        // density correction loop
        self.correct_density_error(dt, fluid_world, predicted_velocities);

        // advect particles
        {
            microprofile::scope!("DFSPHSolver", "advect");

            fluid_world
                .particles
                .positions
                .par_iter_mut()
                .zip(predicted_velocities.par_iter())
                .for_each(|(position, predicted_velocity)| {
                    *position += predicted_velocity * dt;
                });
            time_manager.update_time();
        }
        // only attribute other than position that we need going forward is predicted velocities!
        fluid_world.update_neighborhood_datastructure(vec![predicted_velocities], Vec::new());

        // todo: fuse density & alpha factor computation?
        // recompute densities
        fluid_world.update_densities(self.kernel);
        // recompute alpha factors
        Self::compute_alpha_factors(&mut self.alpha_values, fluid_world, self.kernel);

        // divergence error loop
        self.correct_divergence_error(dt, fluid_world, predicted_velocities);

        // update velocities
        std::mem::swap(&mut fluid_world.particles.velocities, predicted_velocities);
    }
}
