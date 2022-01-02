use std::{
    collections::VecDeque,
    ops::Mul,
    time::{Duration, Instant},
};

use crate::units::*;

#[derive(Clone)]
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
    TargetFrameLength(Duration),
}

#[derive(Clone)]
pub enum SimulationStepConfig {
    FixedTimeStep(Duration),

    // Adjusts the timestep dynamically depending on the simulations needs.
    // (short steps if there is fast moving objects, long steps if objects are moving slowly)
    AdaptiveTimeStep {
        // Maximum time the simulation will advance in one step.
        timestep_max: Duration,

        // Minimum time the simulation will advance, independent of CFL condition.
        // (necessary to avoid infinite stepping for few fast objects)
        // Timestep will not go under this value, even if AdaptiveTimeStepTarget::TargetFrameLength is used.
        timestep_min: Duration,

        // Optionally further governs the timestep.
        timestep_target_frame: AdaptiveTimeStepTarget,

        // Factor for CFL estimation. Values above 1 mean that we will use a larger timestep than CFL condition dictates.
        cfl_factor: Real,
    },
}

// The time manager keeps track of render & simulation timing and time statistics.
// It's set up in a way that makes "normal realtime rendering" easy, but has hooks to allow special handling
// (simulation jump, simulation pause, recording)
//
// There is three dependent clocks one needs to keep in mind
// * wall clock time
//      that's the watch on your wrist, independent of whatever is goingon in the application
// * render time
//      same as on your watch if you're not recording or fast forwarding to a specific time
// * simulation time
//      tries to keep up with render time but potentially in different steps and may start to drop steps
pub struct TimeManager {
    // wall clock time measures
    timestamp_last_frame: Instant,
    duration_last_frame: Duration,
    frame_duration_history: VecDeque<Duration>,

    // render time measures
    total_rendered_time: Duration,
    current_frame_delta: Duration,
    num_frames_rendered: u32,

    // simulation time
    simulation_step_config: SimulationStepConfig,
    simulation_step: Duration,
    num_simulation_steps: u32,
    num_simulation_steps_this_frame: u32,
    simulated_time_this_frame: Duration,
    total_simulated_time: Duration,
    accepted_simulation_to_render_lag: Duration, // time lost that we don't plan on catching up anymore
}

#[derive(PartialEq, Eq)]
pub enum SimulationStepResult {
    PerformStepAndCallAgain,

    CaughtUpWithRenderTime,
    DroppingSimulationSteps,
}

const FRAME_DURATION_HISTORY_LENGTH: usize = 50;

impl TimeManager {
    pub fn new(simulation_step_config: SimulationStepConfig) -> TimeManager {
        let initial_step = match simulation_step_config {
            SimulationStepConfig::FixedTimeStep(step) => step,
            SimulationStepConfig::AdaptiveTimeStep { timestep_min, .. } => timestep_min,
        };

        TimeManager {
            timestamp_last_frame: Instant::now(),
            duration_last_frame: Duration::ZERO,
            frame_duration_history: VecDeque::with_capacity(FRAME_DURATION_HISTORY_LENGTH),

            total_rendered_time: Duration::ZERO,
            current_frame_delta: Duration::ZERO,
            num_frames_rendered: 0,

            simulation_step_config,
            simulation_step: initial_step,
            num_simulation_steps: 0,
            num_simulation_steps_this_frame: 0,
            simulated_time_this_frame: Duration::ZERO,
            total_simulated_time: Duration::ZERO,
            accepted_simulation_to_render_lag: Duration::ZERO,
        }
    }

    pub fn restart(&mut self) {
        *self = Self::new(self.simulation_step_config.clone());
    }

    // current/last simulation delta, updated in update_simulation_step
    pub fn simulation_step(&self) -> Duration {
        self.simulation_step
    }

    pub fn frame_delta(&self) -> Duration {
        self.current_frame_delta
    }

    // Duration of the previous frame. (this is not necessarily equal to the frame time delta!)
    pub fn duration_last_frame(&self) -> Duration {
        self.duration_last_frame
    }

    pub fn duration_last_frame_history(&self) -> &VecDeque<Duration> {
        &self.frame_duration_history
    }

    // Total render time, including current frame. (equal to real time if not configured otherwise!)
    pub fn total_render_time(&self) -> Duration {
        self.total_rendered_time
    }

    // Total time simulated
    pub fn total_simulated_time(&self) -> Duration {
        self.total_simulated_time
    }

    pub fn num_simulation_steps_performed_for_current_frame(&self) -> u32 {
        self.num_simulation_steps_this_frame
    }

    pub fn num_simulation_steps_performed(&self) -> u32 {
        self.num_simulation_steps
    }

    pub fn num_frames_rendered(&self) -> u32 {
        self.num_frames_rendered
    }

    pub fn config(&self) -> &SimulationStepConfig {
        &self.simulation_step_config
    }

    pub fn config_mut(&mut self) -> &mut SimulationStepConfig {
        &mut self.simulation_step_config
    }

    // Forces a given frame delta (timestep on the rendering timeline)
    // Usually the frame delta is just the time between the last two on_frame_submitted calls, but this overwrites it.
    // Useful to jump to a specific time (recording, or fast forwarding the simulation).
    pub fn force_frame_delta(&mut self, delta: Duration) {
        self.total_rendered_time -= self.current_frame_delta;
        self.current_frame_delta = delta;
        self.total_rendered_time += self.current_frame_delta;
    }

    pub fn on_frame_submitted(&mut self, wallclock_to_rendertime_scale: f32) {
        self.duration_last_frame = self.timestamp_last_frame.elapsed();
        if self.frame_duration_history.len() == FRAME_DURATION_HISTORY_LENGTH {
            self.frame_duration_history.pop_front();
        }
        self.frame_duration_history.push_back(self.duration_last_frame);
        self.current_frame_delta = self.duration_last_frame.mul_f32(wallclock_to_rendertime_scale);
        self.total_rendered_time += self.current_frame_delta;

        self.timestamp_last_frame = std::time::Instant::now();
        self.num_simulation_steps_this_frame = 0;
        self.simulated_time_this_frame = Duration::ZERO;
        self.num_frames_rendered += 1;
    }

    pub fn skip_simulation_frame(&mut self) {
        self.accepted_simulation_to_render_lag += self.current_frame_delta;
    }

    pub fn simulation_frame_loop(&mut self, max_simulated_time_per_frame: Duration) -> SimulationStepResult {
        // The rendered time we expect will be reached when this frame is shown on the screen.
        // This does not take the screen sync into account, but should.
        let predicted_rendered_time = self.total_rendered_time + self.current_frame_delta;

        // simulation time shouldn't advance faster than render time
        let residual_time = predicted_rendered_time
            .checked_sub(self.total_simulated_time + self.accepted_simulation_to_render_lag)
            .unwrap_or(Duration::ZERO);
        if residual_time < self.simulation_step {
            // println!(
            //     "realtime {}, fps {}",
            //     self.num_simulation_steps_this_frame,
            //     1.0 / self.frame_delta().as_secs_f32()
            // );
            return SimulationStepResult::CaughtUpWithRenderTime;
        }

        // Did we hit a maximum of simulation steps and want to introduce lag instead?
        if self.simulated_time_this_frame > max_simulated_time_per_frame {
            // We heuristically don't drop all lost simulation frames. This avoids oscillating between realtime and offline
            // which may be caused by our frame deltas being influenced by work from a couple of cpu frames ago (due gpu/cpu sync)
            // This is especially important for gpu driven simulations where don't get direct feedback on doing more or less simulation steps.
            self.accepted_simulation_to_render_lag += residual_time.mul_f32(0.9);
            // println!(
            //     "lagtime {}, fps {}",
            //     self.num_simulation_steps_this_frame,
            //     1.0 / self.frame_delta().as_secs_f32()
            // );
            return SimulationStepResult::DroppingSimulationSteps;
        }

        self.num_simulation_steps_this_frame += 1;
        self.num_simulation_steps += 1;
        self.total_simulated_time += self.simulation_step;
        self.simulated_time_this_frame += self.simulation_step;
        SimulationStepResult::PerformStepAndCallAgain
    }

    // Updates the simulation timestep length depending on the configuration
    pub(super) fn update_simulation_step(&mut self, particle_diameter: Real, max_velocity: Real) -> Duration {
        self.simulation_step = match &self.simulation_step_config {
            SimulationStepConfig::FixedTimeStep(timestep) => *timestep,

            SimulationStepConfig::AdaptiveTimeStep {
                timestep_max,
                timestep_min,
                timestep_target_frame,
                cfl_factor,
            } => {
                const VELOCITY_EPSILON: Real = 0.00001;
                // Evaluates Courant–Friedrichs–Lewy (CFL) condition
                let time_cfl = Duration::from_secs_f32(cfl_factor * 0.4 * particle_diameter / (max_velocity + VELOCITY_EPSILON));
                // Smaller timestep is always fine, but don't jerk it up. Doing so causes timestep oscillation and resulting instability
                // Supposedly this happens in impact situations: Particle's high velocity is reversed, so for a short moment around 0 -> high timstep -> explode back.
                let upper_bound = *timestep_max.min(&self.simulation_step.mul(2));
                let lower_bound = if let AdaptiveTimeStepTarget::TargetFrameLength(timestep_target) = timestep_target_frame {
                    let time_to_target =
                        self.total_simulated_time - timestep_target.mul((self.total_simulated_time.as_nanos() / timestep_target.as_nanos()) as u32);
                    *timestep_min.min(&time_to_target)
                } else {
                    *timestep_min
                };
                lower_bound.max(upper_bound.min(time_cfl))
            }
        };
        self.simulation_step
    }
}
