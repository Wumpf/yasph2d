use crate::units::*;

pub enum AdaptiveTimeStepTarget {
    None,

    // Multiple of total simulated time that the simulation tries to reach.
    //
    // Example: If TargetFrameLength is 1.0 and adaptive stepping yielded 0.9, then the next step will be 0.1 in order to reach the stepping target.
    // Otherwise, we may overshoot all the way to t=1.9 if the next step hits the maximum target. This yields to variable intervals between observable simulation states (==frames)
    // -> This is useful if we want to reach exact specific time intervals.
    //
    // Note: This sounds like an arcane concept but I actually found an instance of it in PySPH, called "final time"
    // https://pysph.readthedocs.io/en/latest/reference/solver.html
    TargetFrameLength(Real),
}

pub enum TimeManagerConfiguration {
    FixedTimeStep(Real),

    // Adjusts the timestep dynamically depending on the simulations needs.
    // (short steps if there is fast moving objects, long steps if objects are moving slowly)
    AdaptiveTimeStep {
        // Maximum time the simulation will advance in one step.
        timestep_max: Real,

        // Minimum time the simulation will advance, independent of CFL condition.
        // (necessary to avoid infinite stepping for few fast objects)
        // Timestep will not go under this value, even if AdaptiveTimeStepTarget::TargetFrameLength is used.
        timestep_min: Real,

        // Optionally further governs the timestep.
        timestep_target_frame: AdaptiveTimeStepTarget,

        // Factor for CFL estimation. Values above 1 mean that we will use a larger timestep than CFL condition dictates.
        cfl_factor: Real,
    },
}

// All timing values in seconds
pub struct TimeManager {
    passed_time: Real,
    timestep: Real,
    config: TimeManagerConfiguration,
}

impl TimeManager {
    pub fn new(config: TimeManagerConfiguration) -> TimeManager {
        let mut instance = TimeManager {
            passed_time: 0.0,
            timestep: 0.0, // solver needs to call update_timestep
            config,
        };
        instance.restart();
        instance
    }

    pub fn restart(&mut self) {
        self.passed_time = 0.0;
        self.timestep = match &self.config {
            TimeManagerConfiguration::FixedTimeStep(timestep) => *timestep,
            TimeManagerConfiguration::AdaptiveTimeStep { timestep_min, .. } => *timestep_min,
        }
    }

    // how much physical time has passed in the simulation
    pub fn passed_time(&self) -> Real {
        self.passed_time
    }

    // how long the last timestep has been
    pub fn timestep(&self) -> Real {
        self.timestep
    }

    pub fn config(&self) -> &TimeManagerConfiguration {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut TimeManagerConfiguration {
        &mut self.config
    }

    pub(super) fn update_timestep(&mut self, particle_diameter: Real, max_velocity: Real) {
        self.timestep = match &self.config {
            TimeManagerConfiguration::FixedTimeStep(timestep) => *timestep,

            TimeManagerConfiguration::AdaptiveTimeStep {
                timestep_max,
                timestep_min,
                timestep_target_frame,
                cfl_factor,
            } => {
                const VELOCITY_EPSILON: Real = 0.00001;
                // Evaluates Courant–Friedrichs–Lewy (CFL) condition
                let time_cfl = cfl_factor * 0.4 * particle_diameter / (max_velocity + VELOCITY_EPSILON);
                // Smaller timestep is always fine, but don't jerk it up. Doing so causes timestep oscillation and resulting instability
                // Supposedly this happens in impact situations: Particle's high velocity is reversed, so for a short moment around 0 -> high timstep -> explode back.
                let upper_bound = timestep_max.min(self.timestep * 2.0);
                let lower_bound = if let AdaptiveTimeStepTarget::TargetFrameLength(timestep_target) = timestep_target_frame {
                    let time_to_target = self.passed_time - (self.passed_time / timestep_target).floor() * timestep_target;
                    timestep_min.min(time_to_target)
                } else {
                    *timestep_min
                };
                lower_bound.max(upper_bound.min(time_cfl))
            }
        }
    }

    // updates time with the current timestep
    pub(super) fn update_time(&mut self) {
        self.passed_time += self.timestep;
    }
}
