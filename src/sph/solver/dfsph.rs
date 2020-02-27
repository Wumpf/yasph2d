use super::super::fluidparticleworld::FluidParticleWorld;
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
    viscosity_model: TViscosityModel,

    kernel: smoothing_kernel::CubicSpline,

    // Not using a boundary mass/force factor as in our WCSPH, since solver usually makes sure particles don't pass each other.
    //boundary_mass_factor: Real,

    // Max density error. In relative density deviation per second - 0.01 means 1% density deviation per second.
    max_density_error: Real,
    max_num_density_correction_iterations: usize,

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

            max_density_error: 1.0 / 1000.0 / 100.0, // 1.0% deviation per millisecond.
            max_num_density_correction_iterations: 100,

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
                        let rj = particles.positions[j as usize];
                        let grad_ij = kernel.gradient_from_positions(ri, rj) * particle_mass;
                        gradient_sum += grad_ij;
                        gradient_square_sum += grad_ij.magnitude2();
                    },
                );
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        let rj = particles.boundary_particles[j as usize];
                        let grad_ij = kernel.gradient_from_positions(ri, rj) * particle_mass;
                        gradient_sum += grad_ij;
                        gradient_square_sum += grad_ij.magnitude2();
                    },
                );

                *alpha_value = 1.0 / (gradient_sum.magnitude2() + gradient_square_sum).max(EPSILON);
                // todo?
            });
    }

    fn predict_densities(&mut self, dt: Real, fluid_world: &FluidParticleWorld) {
        microprofile::scope!("DFSPHSolver", "predict_densities");
        let particle_mass = fluid_world.properties.particle_mass();
        let particles = &fluid_world.particles;
        let predicted_velocities = &self.predicted_velocities;
        let kernel = &self.kernel;
        let reference_density = fluid_world.properties.fluid_density();
        self.predicted_densities
            .par_iter_mut()
            .zip(
                (
                    &fluid_world.particles.densities,
                    &fluid_world.particles.positions,
                    &self.predicted_velocities,
                )
                    .into_par_iter(),
            )
            .enumerate()
            .for_each(|(i, (predicted_densitiy, (&original_density, &ri, &predicted_vi)))| {
                let mut delta = 0.0; // gradient to self is zero.
                let i = i as u32;
                particles.foreach_neighbor_particle(
                    i,
                    #[inline(always)]
                    |j| {
                        let rj = particles.positions[j as usize];
                        let delta_v = predicted_vi - predicted_velocities[j as usize];
                        delta += delta_v.dot(kernel.gradient_from_positions(ri, rj));
                    },
                );
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        let rj = particles.boundary_particles[j as usize];
                        let delta_v = predicted_vi;
                        delta += delta_v.dot(kernel.gradient_from_positions(ri, rj));
                    },
                );
                *predicted_densitiy = original_density + delta * particle_mass * dt;

                // ignore loss of density
                *predicted_densitiy = reference_density.max(*predicted_densitiy);
            });
    }

    fn update_velocity_prediction(&mut self, dt: Real, fluid_world: &FluidParticleWorld) {
        microprofile::scope!("DFSPHSolver", "update_velocity_prediction");
        let particle_mass = fluid_world.properties.particle_mass();
        let particles = &fluid_world.particles;
        let predicted_densities = &self.predicted_densities;
        let alpha_values = &self.alpha_values;
        let reference_density = fluid_world.properties.fluid_density();
        let inv_dt = 1.0 / dt;
        let kernel = &self.kernel;

        self.predicted_velocities
            .par_iter_mut()
            .zip((&fluid_world.particles.positions, &self.predicted_densities, &self.alpha_values).into_par_iter())
            .enumerate()
            .for_each(|(i, (predicted_velocity, (&ri, predicted_density_i, alpha_i)))| {
                let mut delta: Vector = Zero::zero(); // gradient to self is zero.
                let ki = (predicted_density_i - reference_density) * alpha_i;

                // compared to k values in paper already divided with density
                // collapsing divition of dt² with multiply later -> divide delta with dt

                let i = i as u32;
                particles.foreach_neighbor_particle(
                    i,
                    #[inline(always)]
                    |j| {
                        let rj = particles.positions[j as usize];
                        let kj = (predicted_densities[j as usize] - reference_density) * alpha_values[j as usize];
                        delta += (ki + kj) * kernel.gradient_from_positions(ri, rj);
                    },
                );
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        // compared to k values in paper already divided with density and multiplied with dt²!
                        let rj = particles.boundary_particles[j as usize];
                        delta += ki * kernel.gradient_from_positions(ri, rj);
                    },
                );

                *predicted_velocity -= inv_dt * delta * particle_mass;
            });
    }

    fn correct_density_error(&mut self, dt: Real, fluid_world: &FluidParticleWorld) {
        // todo: warmup & general use of values from previous frame
        microprofile::scope!("DFSPHSolver", "correct_density_error");

        let mut num_iter = 0;

        loop {
            microprofile::scope!("DFSPHSolver", "density_iteration");

            self.predict_densities(dt, fluid_world);
            self.update_velocity_prediction(dt, fluid_world);
            num_iter += 1;

            let average_density: Real = self.predicted_densities.par_iter().sum::<Real>() / self.predicted_densities.len() as Real;
            assert!(average_density.is_finite());
            let density_error = (average_density - fluid_world.properties.fluid_density()).abs();

            // error is expressed relative to fluid density and time!
            if density_error < self.max_density_error / dt * fluid_world.properties.fluid_density() {
                // println!(
                //     "density error correction succeeded after {} steps. Density error was {}.",
                //     num_iter, density_error
                // );
                break;
            }
            if num_iter > self.max_num_density_correction_iterations {
                println!(
                    "density error canceled after {} steps. Density error was {}, that is {}%.",
                    num_iter,
                    density_error,
                    density_error / fluid_world.properties.fluid_density() / 100.0
                );
                break;
            }
        }
    }
}

impl<TViscosityModel: ViscosityModel + std::marker::Sync> Solver for DFSPHSolver<TViscosityModel> {
    fn clear_cached_data(&mut self) {
        self.alpha_values.clear();
        self.predicted_velocities.clear();
        self.predicted_densities.clear();
    }

    fn simulation_step(&mut self, fluid_world: &mut FluidParticleWorld, dt: Real) {
        microprofile::scope!("DFSPHSolver", "simulation_step");

        // ensure densities and alpha factors were initialized previously ("warmup")
        // Todo: Not happy about the way added particles are handled here. This sort of works for adding, but removing this way is impossible with this design!
        if self.alpha_values.len() != fluid_world.particles.positions.len() {
            self.alpha_values.resize(fluid_world.particles.positions.len(), 0.0 as Real);
            self.predicted_velocities.resize(fluid_world.particles.positions.len(), Zero::zero());
            self.predicted_densities.resize(fluid_world.particles.positions.len(), Zero::zero());

            // todo: Update only new particles.. HOW? better would be to only effectively add later
            fluid_world.update_neighborhood_datastructure(Vec::new(), vec![&mut self.alpha_values]);
            fluid_world.update_densities(self.kernel);
            Self::compute_alpha_factors(&mut self.alpha_values, fluid_world, self.kernel);
        }

        // compute non-pressure forces (from scratch)
        let non_pressure_forces: Vector = fluid_world.gravity * fluid_world.properties.particle_mass();

        // (optional) adapt timestep using CFL condition
        // todo

        // compute velocity prediction
        {
            microprofile::scope!("DFSPHSolver", "velocity prediction");

            let particles = &fluid_world.particles;
            let particle_mass = fluid_world.properties.particle_mass();
            let viscosity_model = &self.viscosity_model;
            let force_to_particle_velocitychange = dt / particle_mass;
            let non_pressure_velocitychange = force_to_particle_velocitychange * non_pressure_forces;
            self.predicted_velocities
                .par_iter_mut()
                .zip((&particles.positions, &particles.velocities).into_par_iter())
                .enumerate()
                .for_each(|(i, (predicted_velocity, (&ri, &vi)))| {
                    // forces
                    *predicted_velocity = non_pressure_velocitychange + vi;

                    // viscosity
                    particles.foreach_neighbor_particle(
                        i as u32,
                        #[inline(always)]
                        |j| {
                            let j = j as usize;
                            let r_sq = ri.distance2(particles.positions[j]);
                            *predicted_velocity += dt
                                * viscosity_model.compute_viscous_accelleration(
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

        // density correction loop
        self.correct_density_error(dt, fluid_world);

        // advect particles
        {
            microprofile::scope!("DFSPHSolver", "advect");

            fluid_world
                .particles
                .positions
                .par_iter_mut()
                .zip(self.predicted_velocities.par_iter())
                .for_each(|(position, predicted_velocity)| {
                    *position += predicted_velocity * dt;
                });
        }
        // only attribute other than position that we need going forward is predicted velocities!
        fluid_world.update_neighborhood_datastructure(vec![&mut self.predicted_velocities], Vec::new());

        // todo: fuse density & alpha factor computation.
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
