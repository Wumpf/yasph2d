use super::super::hydroparticles::FluidParticleWorld;
use super::super::smoothing_kernel;
use super::super::smoothing_kernel::Kernel;
use super::super::viscositymodel::ViscosityModel;
use super::Solver;
use crate::units::*;
use rayon::prelude::*;

// WCSPH implementation as described in
// Divergence-Free SPH for Incompressible and Viscious Fluids
// https://animation.rwth-aachen.de/publication/051/
pub struct DFSPHSolver<TViscosityModel: ViscosityModel> {
    viscosity_model: TViscosityModel,

    density_kernel: smoothing_kernel::Poly6,
    pressure_kernel: smoothing_kernel::Spiky,

    boundary_force_factor: Real,
}
impl<TViscosityModel: ViscosityModel + std::marker::Sync> DFSPHSolver<TViscosityModel> {
    pub fn new(viscosity_model: TViscosityModel, smoothing_length: Real) -> DFSPHSolver<TViscosityModel> {
        DFSPHSolver {
            viscosity_model,

            density_kernel: smoothing_kernel::Poly6::new(smoothing_length),
            pressure_kernel: smoothing_kernel::Spiky::new(smoothing_length),

            boundary_force_factor: 10.0, // (expected accelleration) / (spacing ratio of boundary / normal particles)
        }
    }
}

impl<TViscosityModel: ViscosityModel + std::marker::Sync> Solver for DFSPHSolver<TViscosityModel> {
    fn simulation_step(&self, particles: &mut FluidParticleWorld, dt: Real) {
        assert_eq!(particles.positions.len(), particles.velocities.len());
        assert_eq!(particles.positions.len(), particles.accellerations.len());

        // ensure densities and alpha factors were initialized previously ("warmup")

        // compute non-pressure forces (from scratch)
        // (optional) adapter timestep
        // compute velocity prediction (from scratch, can discard forces now!)
        // density correction loop
        // advect particles
        // recompute densities
        // recompute alpha factors
        // divergence error loop
        // update velocities
    }
}
