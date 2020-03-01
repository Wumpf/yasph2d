use crate::units::*;

pub struct AdpativeTimeStep {
    pub timestep_min: Real,
    pub timestep_max: Real,
    pub cfl_factor: Real,
}

pub enum TimeManagerConfiguration {
    FixedTimeStep(Real),
    //    AdaptiveTimeStep(AdpativeTimeStep),
}

// All timing values in seconds
pub struct TimeManager {
    passed_time: Real,
    timestep: Real,
    config: TimeManagerConfiguration,
}

impl TimeManager {
    pub fn new(config: TimeManagerConfiguration) -> TimeManager {
        TimeManager {
            passed_time: 0.0,
            timestep: 0.0, // solver needs to call update_timestep
            config,
        }
    }

    pub fn restart(&mut self) {
        self.passed_time = 0.0;
        self.timestep = 0.0;
    }

    // how much physical time has passed in the simulation
    pub fn passed_time(&self) -> Real {
        self.passed_time
    }

    // how long the last timestep has been
    pub fn timestep(&self) -> Real {
        self.timestep
    }

    pub(super) fn update_timestep(&mut self, _particle_diameter: Real, _max_velocity: Real) {
        self.timestep = match &self.config {
            TimeManagerConfiguration::FixedTimeStep(timestep) => *timestep,
            // Evaluates Courant–Friedrichs–Lewy (CFL) condition for the solver if any, loose timestep recommendation otherwise.
            // TimeManagerConfiguration::AdaptiveTimeStep(adaptive_config) => Real::min(
            //     adaptive_config.timestep_max,
            //     Real::min(
            //         adaptive_config.timestep_min,
            //         adaptive_config.cfl_factor * 0.4 * particle_diameter / max_velocity,
            //     ),
            // ),
        }
    }

    pub(super) fn update_time(&mut self) {
        self.passed_time += self.timestep;
    }
}
