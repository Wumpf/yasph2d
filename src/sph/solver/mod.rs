pub use dfsph::DFSPHSolver;
pub use wscsph::WCSPHSolver;

mod dfsph;
mod wscsph;

// ------------------------------------------------------

use super::fluidparticleworld::FluidParticleWorld;
use super::timemanager::TimeManager;

pub trait Solver {
    // todo: this is not elegant, should be done automatically
    fn clear_cached_data(&mut self);

    // performs a single simulation step.
    fn simulation_step(&mut self, fluid_world: &mut FluidParticleWorld, time_manager: &mut TimeManager);
}
