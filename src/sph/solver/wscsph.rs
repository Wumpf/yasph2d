use super::super::fluidparticleworld::{ConstantFluidProperties, FluidParticleWorld};
use super::super::smoothing_kernel;
use super::super::smoothing_kernel::Kernel;
use super::super::timemanager::TimeManager;
use super::super::viscositymodel::ViscosityModel;
use super::Solver;
use crate::units::*;
use cgmath::prelude::*;
use rayon::prelude::*;

// Solver based on Becker & Teschner 2007 WCSPH07
// No surface tension implemented
// https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf
pub struct WCSPHSolver<TViscosityModel: ViscosityModel> {
    viscosity_model: TViscosityModel,
    density_kernel: smoothing_kernel::Poly6,
    pressure_kernel: smoothing_kernel::Spiky,
    boundary_force_factor: Real,
    stiffness: Real, // denoted as B. B = density0 * speed_of_sound * speed_of_sound / γ.

    // recomputed every frame, but need previous frame due to leap frog iteration scheme
    accellerations: Vec<Vector>,
}

// γ is hardcoded to 7 as propsed in the paper
const TAIT_EQUATION_GAMMA: i32 = 7;

impl<TViscosityModel: ViscosityModel + std::marker::Sync> WCSPHSolver<TViscosityModel> {
    #[allow(dead_code)]
    pub fn new(viscosity_model: TViscosityModel, fluid_properties: &ConstantFluidProperties) -> WCSPHSolver<TViscosityModel> {
        let mut solver = WCSPHSolver {
            viscosity_model,
            density_kernel: smoothing_kernel::Poly6::new(fluid_properties.smoothing_length()),
            pressure_kernel: smoothing_kernel::Spiky::new(fluid_properties.smoothing_length()),
            boundary_force_factor: 1.0, // (expected accelleration * initial water depth) / (spacing ratio of boundary / normal particles). Arbitrary value right now.
            stiffness: 0.0,             // set in set_compressibility below
            accellerations: Vec::new(),
        };
        // set a good default for compressibility
        solver.set_compressibility(fluid_properties, 0.01, 1.0);
        solver
    }

    // target_density_variation:    density variation, denoted as η in the paper. defaults to 1%==0.01
    // expected_max_flow_speed:     expected speed of the fluid in m/s. possible estimate is sqrt(2 * gravity * falling_height)
    pub fn set_compressibility(&mut self, fluid_properties: &ConstantFluidProperties, target_density_variation: Real, expected_max_flow_speed: Real) {
        // real speed of sound of the fluid is usually higher, but this makes our timesteps way too small
        let speed_of_sound = expected_max_flow_speed / target_density_variation.sqrt();
        self.stiffness = fluid_properties.fluid_density() * speed_of_sound * speed_of_sound / TAIT_EQUATION_GAMMA as Real;
    }

    // Equation of State (EOS)
    fn pressure(stiffness: Real, fluid_density: Real, local_density: Real) -> Real {
        // Tait equation as in Becker & Teschner 2007 WCSPH07
        // The max on pressure ratio is due to pressure clamping to work around particle deficiency problem. Good explanation here:
        // https://github.com/InteractiveComputerGraphics/SPlisHSPlasH/issues/36#issuecomment-495883932
        stiffness * ((local_density / fluid_density).max(1.0).powi(TAIT_EQUATION_GAMMA) - 1.0)
    }

    fn update_accellerations(&mut self, fluid_world: &FluidParticleWorld, dt: Real) {
        microprofile::scope!("WCSPHSolver", "update_accellerations");
        let mass = fluid_world.properties.particle_mass();

        // pressure & viscosity forces
        // According to https://www8.cs.umu.se/kurser/TDBD24/VT06/lectures/sphsurvivalkit.pdf
        // the "good way" to do symmetric forces in SPH is -m (pi + pj) / (2 * rhoj * rhoi)

        let fluid_density = fluid_world.properties.fluid_density();
        let particles = &fluid_world.particles;
        let pressure_kernel = self.pressure_kernel;
        let boundary_force_factor = self.boundary_force_factor;
        let viscosity_model = &self.viscosity_model;
        let gravity = fluid_world.gravity;
        let stiffness = self.stiffness;

        self.accellerations
            .par_iter_mut()
            .zip(
                (
                    &fluid_world.particles.velocities,
                    &fluid_world.particles.positions,
                    &fluid_world.particles.densities,
                )
                    .into_par_iter(),
            )
            .enumerate()
            .for_each(|(i, (accelleration, (&vi, &ri, &rhoi)))| {
                *accelleration = gravity;

                let pi = Self::pressure(stiffness, fluid_density, rhoi);
                let i = i as u32;

                // no self-contribution since vector to particle is zero (-> no pressure) and velocity difference is zero as well (-> no viscosity)
                particles.foreach_neighbor_particle(
                    i,
                    #[inline(always)]
                    |j| {
                        let j = j as usize;
                        let rhoj = particles.densities[j];
                        let pj = Self::pressure(stiffness, fluid_density, rhoj);
                        let ri_to_rj = particles.positions[j] - ri;
                        let r_sq = ri_to_rj.magnitude2();
                        let r = r_sq.sqrt();

                        // accelleration from pressure force
                        // As in "Particle-Based Fluid Simulation for Interactive Applications", Müller et al.
                        // This is a weakly compressible model (WCSPH)
                        let pressure_unsmoothed = -mass * (pi + pj) / (2.0 * rhoi * rhoj);
                        *accelleration += pressure_unsmoothed * pressure_kernel.gradient(ri_to_rj, r_sq, r);

                        *accelleration += viscosity_model.compute_viscous_accelleration(dt, r_sq, r, mass, rhoj, particles.velocities[j] - vi);
                    },
                );

                // Boundary forces as described by
                // "SPH particle boundary forces for arbitrary boundaries" by Monaghan and Kajtar 2009
                // Simple formulation found in http://www.unige.ch/math/folks/sutti/SPH_2019.pdf under 2.3.4 Radial force
                // ("SPH treatment of boundaries and application to moving objects" by Marco Sutti)
                particles.foreach_neighbor_particle_boundary(
                    i,
                    #[inline(always)]
                    |j| {
                        let ri_to_rj = particles.boundary_particles[j as usize] - ri;
                        let r_sq = ri_to_rj.magnitude2();
                        *accelleration -= boundary_force_factor * pressure_kernel.evaluate(r_sq, r_sq.sqrt()) / r_sq * ri_to_rj;
                    },
                );
            });
    }
}

impl<TViscosityModel: ViscosityModel + std::marker::Sync> Solver for WCSPHSolver<TViscosityModel> {
    fn clear_cached_data(&mut self) {
        self.accellerations.clear();
    }

    fn simulation_step(&mut self, fluid_world: &mut FluidParticleWorld, time_manager: &mut TimeManager) {
        microprofile::scope!("WCSPHSolver", "simulation_step");
        self.accellerations.resize(fluid_world.particles.positions.len(), cgmath::Zero::zero());

        // leap frog integration scheme with integer steps
        // https://en.wikipedia.org/wiki/Leapfrog_integration

        let mut dt = time_manager.timestep();

        {
            microprofile::scope!("WCSPHSolver", "leap frog 1");

            // This got actually slower for a parallel for loop when used with 2500 particles (too few? or is rayon doing something silly?)
            // fluid_world.particles.positions.par_iter_mut().zip(fluid_world.particles.velocities.par_iter_mut()).zip(self.accellerations.par_iter()).for_each(|((pos, v), a)| {

            for ((pos, v), a) in fluid_world
                .particles
                .positions
                .iter_mut()
                .zip(fluid_world.particles.velocities.iter_mut())
                .zip(self.accellerations.iter())
            {
                *v += 0.5 * dt * a; // v at t_(i+0.5)
                *pos += *v * dt; // pos at t_(i+1)
            }
        }

        fluid_world.update_neighborhood_datastructure(Vec::new(), Vec::new());
        fluid_world.update_densities(self.density_kernel);
        self.update_accellerations(fluid_world, dt);

        // update timestep
        {
            microprofile::scope!("WCSPHSolver", "update timestep");
            let mut max_velocity_sq: Real = 0.0;
            for (v, a) in fluid_world.particles.velocities.iter().zip(self.accellerations.iter()) {
                max_velocity_sq = max_velocity_sq.max((v + a * dt).magnitude2());
            }
            time_manager.update_timestep(fluid_world.properties.particle_radius() * 2.0, max_velocity_sq.sqrt());
            dt = time_manager.timestep();
        }

        // part 2 of leap frog integration. Finish updating velocity.
        {
            // This got actually slower for a parallel for loop when used with 2500 particles (too few? or is rayon doing something silly?)
            // fluid_world.particles.velocities.par_iter_mut().zip(self.accellerations.par_iter()).for_each(|(v, a)|

            microprofile::scope!("WCSPHSolver", "leap frog 2");
            for (v, a) in fluid_world.particles.velocities.iter_mut().zip(self.accellerations.iter()) {
                *v += 0.5 * dt * a; // v at t_(i+1)
            }
        }
    }
}
