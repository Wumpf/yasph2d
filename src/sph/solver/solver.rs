use crate::units::Real;
use super::super::hydroparticles::HydroParticles;

pub trait Solver {
    // performs a single simulation step.
    // todo: how to handle adaptive time stepping. Probably make up dt and return it.
    fn simulation_step(&self, particles: &mut HydroParticles, dt: Real);
}
