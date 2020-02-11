pub use dfsph::DFSPHSolver;
pub use wscsph::WCSPHSolver;

mod dfsph;
mod wscsph;

// ------------------------------------------------------

use super::fluidparticleworld::FluidParticleWorld;
use crate::units::Real;

pub trait Solver {
    // todo: this is not elegant, should be done automatically
    fn clear_cached_data(&mut self);

    // performs a single simulation step.
    // todo: how to handle adaptive time stepping. Probably make up dt and return it.
    fn simulation_step(&mut self, fluid_world: &mut FluidParticleWorld, dt: Real);
}
