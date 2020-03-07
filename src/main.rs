use cgmath::prelude::*;
use ggez::event::{self, EventHandler, KeyCode, KeyMods};
use ggez::graphics::Rect;
use ggez::{conf, graphics, timer, Context, GameResult};
use microprofile;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

mod camera;

use camera::*;
use yasph2d::sph;
use yasph2d::units::*;

fn main() -> GameResult {
    let context_builder = ggez::ContextBuilder::new("YaSPH2D", "AndreasR")
        .window_setup(
            conf::WindowSetup::default()
                .title("YaSPH2D")
                //.samples(conf::NumSamples::Eight) // https://github.com/ggez/ggez/issues/751
                .vsync(false),
        )
        .window_mode(conf::WindowMode::default().dimensions(1920.0, 1080.0));
    let (ctx, event_loop) = &mut context_builder.build()?;
    let state = &mut MainState::new(ctx);

    microprofile::init!();
    microprofile::set_enable_all_groups!(true);

    event::run(ctx, event_loop, state)
}

#[derive(PartialEq)]
enum UpdateMode {
    RealTime,
    Recording,
}
#[derive(PartialEq)]
enum Solver {
    WSCSPH,
    DFSPH,
}

struct MainState {
    update_mode: UpdateMode,
    //solver: Solver,
    fluid_world: sph::FluidParticleWorld,
    time_manager: sph::TimeManager,
    sph_solver: Box<dyn sph::Solver>,

    camera: Camera,
    particle_mesh: graphics::Mesh,

    simulation_step_duration_history: VecDeque<Duration>,
    simulation_processing_time_frame: Duration,
    simulationstep_count_frame: u32,

    simulation_starttime: Instant,
    simulation_processing_time_total: Duration,
    simulation_to_realtime_offset: f32, // Starts out with 0 and grows if we spend too much time on processing the simulation

    frame_counter: usize,
}

const SIMULATION_STEP_HISTORY_LENGTH: usize = 80;

// Application tries to hit this framerate. If it simulates faster, simulation sleeps. If recording this simply *is* the framerate.
// Todo: Be awesome and make this dependent on what the screen can do.
const TARGET_FPS: Real = 60.0;
// Simulation time will try to stay sync with real time unless framerate drops below this. If it simulates slower, simulation time slows down.
// Note that this measure avoid the "well of despair" (as dubbed by https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/guide/Manual/BestPractices.html)
// where we need to do more physics step to catch up, but by doing so take even longer to catch up.
const TARGET_FPS_MIN: Real = 10.0;
const TARGET_MAX_PROCESSING_TIME: Real = 1.0 / TARGET_FPS_MIN;

// Desired relationship between time in reality and time in simulation. In other word, "speed factor"
// (that is, if we simulation processing time is low enough, otherwise simulation will slow down regardless)
const REALTIME_TO_SIMTIME_SCALE: f32 = 1.0;

const TARGET_FRAME_SIMDURATION: Real = REALTIME_TO_SIMTIME_SCALE / TARGET_FPS;

fn clamp(v: f32, min: f32, max: f32) -> f32 {
    if v < min {
        min
    } else if v > max {
        max
    } else {
        v
    }
}

fn heatmap_color(t: f32) -> graphics::Color {
    graphics::Color {
        r: clamp(t * 3.0, 0.0, 1.0),
        g: clamp(t * 3.0 - 1.0, 0.0, 1.0),
        b: clamp(t * 3.0 - 2.0, 0.0, 1.0),
        a: 1.0,
    }
}

impl MainState {
    pub fn new(ctx: &mut Context) -> MainState {
        let mut fluid_world = sph::FluidParticleWorld::new(
            2.0,    // smoothing factor
            5000.0, // #particles/m²
            100.0,  // density of water (? this is 2d, not 3d where it's 1000 kg/m³)
        );
        Self::reset_fluid(&mut fluid_world);
        let solver = Solver::DFSPH; // Solver::WSCSPH;

        let xsph = sph::XSPHViscosityModel::new(fluid_world.properties.smoothing_length());
        //xsph.epsilon = 0.1;
        let mut physicalviscosity = sph::PhysicalViscosityModel::new(fluid_world.properties.smoothing_length());
        physicalviscosity.fluid_viscosity = 0.01;

        let sph_solver: Box<dyn sph::Solver> = match solver {
            Solver::WSCSPH => Box::new(sph::WCSPHSolver::new(xsph, &fluid_world.properties)),
            Solver::DFSPH => Box::new(sph::DFSPHSolver::new(xsph, fluid_world.properties.smoothing_length())),
        };

        let particle_radius = fluid_world.properties.particle_radius();
        let particle_mesh = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            RenderPoint::origin(),
            particle_radius as f32,
            0.0003,
            graphics::WHITE,
        )
        .unwrap();

        let cfl_factor = match solver {
            Solver::WSCSPH => 0.2,
            Solver::DFSPH => 1.0,
        };

        let time_manager = sph::TimeManager::new(
            //sph::TimeManagerConfiguration::FixedTimeStep(TARGET_FRAME_SIMDURATION / 20.0));
            sph::TimeManagerConfiguration::AdaptiveTimeStep {
                timestep_max: TARGET_FRAME_SIMDURATION / 4.0,
                timestep_min: REALTIME_TO_SIMTIME_SCALE / (400.0 * 60.0), // Don't do steps that results in more than a 400 steps for an image on a classic 60Hz display
                timestep_target_frame: sph::AdaptiveTimeStepTarget::None,
                cfl_factor,
            },
        );

        MainState {
            update_mode: UpdateMode::RealTime,
            fluid_world,
            time_manager,
            sph_solver,

            camera: Camera::center_around_world_rect(graphics::screen_coordinates(ctx), Rect::new(-0.1, -0.1, 2.1, 1.6)),
            particle_mesh,

            simulation_step_duration_history: VecDeque::with_capacity(SIMULATION_STEP_HISTORY_LENGTH),
            simulation_processing_time_frame: Default::default(),
            simulationstep_count_frame: 0,

            simulation_starttime: Instant::now(),
            simulation_processing_time_total: Default::default(),
            simulation_to_realtime_offset: Default::default(),

            frame_counter: 0,
        }
    }

    fn reset_fluid(fluid_world: &mut sph::FluidParticleWorld) {
        fluid_world.remove_all_fluid_particles();
        fluid_world.remove_all_boundary_particles();

        fluid_world.add_fluid_rect(&Rect::new(0.1, 0.7, 0.5, 1.0), 0.05);
        fluid_world.add_boundary_thick_line(Point::new(0.0, 0.0), Point::new(2.0, 0.0), 2);
        fluid_world.add_boundary_thick_line(Point::new(0.0, 0.0), Point::new(0.0, 2.5), 2);
        fluid_world.add_boundary_thick_line(Point::new(2.0, 0.0), Point::new(2.0, 2.5), 2);

        fluid_world.add_boundary_line(Point::new(0.0, 0.6), Point::new(1.75, 0.5));

        // close of the container - stop gap solution for issues with endlessly falling particles
        // (mostly a problem for adaptive timestep but potentially also for neighborhood search)
        fluid_world.add_boundary_thick_line(Point::new(0.0, 2.5), Point::new(2.0, 2.5), 2);
    }

    fn draw_text(&mut self, ctx: &mut Context) -> GameResult {
        microprofile::scope!("MainState", "text");

        let fps = timer::fps(ctx);
        let average_simulation_step_duration =
            self.simulation_step_duration_history.iter().sum::<Duration>() / self.simulation_step_duration_history.len() as u32;

        let simulation_info_text = format!(
            "Frame Processing: {:3.2}ms ({:4} steps)\nSingle Step (averaged over {}): {:.2}ms, last timestep length {:.4}ms\nTotal Simulated {:.2}s\nTotal Processing {:.2}s",
            self.simulation_processing_time_frame.as_secs_f64() * 1000.0,
            self.simulationstep_count_frame,
            self.simulation_step_duration_history.len(),
            average_simulation_step_duration.as_secs_f64() * 1000.0,
            self.time_manager.timestep() * 1000.0,
            self.time_manager.passed_time(),
            self.simulation_processing_time_total.as_secs_f64(),
        );

        let fps_display = graphics::Text::new(match self.update_mode {
            UpdateMode::RealTime => format!(
                "{:3.2}ms, FPS: {:3.2}\ntime since sim start {:.2}s\n\n{}",
                1000.0 / fps,
                fps,
                (Instant::now() - self.simulation_starttime).as_secs_f64(),
                simulation_info_text,
            ),

            UpdateMode::Recording => format!("RECORDING\n{}", simulation_info_text,),
        });
        graphics::draw(ctx, &fps_display, (RenderPoint::new(10.0, 10.0), graphics::WHITE))?;
        if self.simulation_processing_time_frame.as_secs_f32() > TARGET_MAX_PROCESSING_TIME && self.update_mode == UpdateMode::RealTime {
            graphics::draw(
                ctx,
                &graphics::Text::new("REALTIME OFF - simulation time can not keep up with real time"),
                (RenderPoint::new(10.0, 150.0), graphics::Color::new(1.0, 0.2, 0.2, 1.0)),
            )?;
        }

        Ok(())
    }

    fn draw_fluid(&mut self, ctx: &mut Context) -> GameResult {
        microprofile::scope!("MainState", "draw fluid");

        let boundary_color = graphics::Color {
            r: 0.2,
            g: 0.2,
            b: 0.2,
            a: 1.0,
        };
        for (p, a) in self
            .fluid_world
            .particles
            .positions
            .iter()
            .zip(self.fluid_world.particles.velocities.iter())
        {
            let c = heatmap_color((a.magnitude() * 0.1) as f32);
            let rp: RenderPoint = RenderPoint::new(p.x, p.y);
            graphics::draw(ctx, &self.particle_mesh, ggez::graphics::DrawParam::default().dest(rp).color(c))?;
        }
        for p in self.fluid_world.particles.boundary_particles.iter() {
            let rp: RenderPoint = RenderPoint::new(p.x, p.y);
            graphics::draw(
                ctx,
                &self.particle_mesh,
                ggez::graphics::DrawParam::default().dest(rp).color(boundary_color),
            )?;
        }

        graphics::pop_transform(ctx);
        graphics::apply_transformations(ctx)?;
        Ok(())
    }

    fn single_sim_step(&mut self) {
        let time_before = Instant::now();
        self.sph_solver.simulation_step(&mut self.fluid_world, &mut self.time_manager);
        let time_after = Instant::now();

        let step_processing_time = time_after - time_before;
        self.simulation_processing_time_frame += step_processing_time;
        self.simulation_processing_time_total += step_processing_time;
        self.simulationstep_count_frame += 1;

        if self.simulation_step_duration_history.len() == SIMULATION_STEP_HISTORY_LENGTH {
            self.simulation_step_duration_history.pop_front();
        }
        self.simulation_step_duration_history.push_back(step_processing_time);
    }

    fn reset_simulation(&mut self) {
        self.sph_solver.clear_cached_data(); // todo: this is super meh
        self.simulation_starttime = Instant::now();
        self.simulation_to_realtime_offset = 0.0;
        self.simulation_processing_time_total = Default::default();

        self.frame_counter = 0;
        self.time_manager.restart();
        Self::reset_fluid(&mut self.fluid_world);
    }
}

impl EventHandler for MainState {
    fn key_down_event(&mut self, ctx: &mut Context, keycode: KeyCode, _keymods: KeyMods, repeat: bool) {
        match keycode {
            KeyCode::Escape => {
                ggez::event::quit(ctx);
            }
            KeyCode::Space => {
                self.reset_simulation();
            }
            KeyCode::R => {
                if !repeat {
                    self.update_mode = if self.update_mode == UpdateMode::RealTime {
                        UpdateMode::Recording
                    } else {
                        UpdateMode::RealTime
                    };
                    self.reset_simulation();
                }
            }
            _ => {
                self.update_mode = UpdateMode::RealTime;
            }
        }
    }

    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        microprofile::scope!("MainState", "update");

        self.simulationstep_count_frame = 0;
        self.simulation_processing_time_frame = Duration::from_secs(0);

        match self.update_mode {
            UpdateMode::RealTime => {
                // Note that we _could_ influence the simulation timestep target every frame depending on the delta frame time.
                // However, that would make our simulation dependend on external, non-deterministic factors and we don't want that.
                if let sph::TimeManagerConfiguration::AdaptiveTimeStep { timestep_target_frame, .. } = self.time_manager.config_mut() {
                    *timestep_target_frame = sph::AdaptiveTimeStepTarget::None;
                }

                let target_simulation_time =
                    (Instant::now() - self.simulation_starttime).as_secs_f32() * REALTIME_TO_SIMTIME_SCALE - self.simulation_to_realtime_offset;
                while self.time_manager.passed_time() < target_simulation_time {
                    //if self.time_manager.passed_time() > 2.0 {
                    //    break;
                    //}

                    // If we can't process fast enough, we give up and accept that there is an offset between realtime and simulation time.
                    if self.simulation_processing_time_frame.as_secs_f32() > TARGET_MAX_PROCESSING_TIME {
                        self.simulation_to_realtime_offset += target_simulation_time - self.time_manager.passed_time();
                        break;
                    }

                    self.single_sim_step();
                }
            }
            UpdateMode::Recording => {
                // When doing recording, we want to hit the exact frame times.
                let epsilon;
                if let sph::TimeManagerConfiguration::AdaptiveTimeStep {
                    timestep_min,
                    timestep_target_frame,
                    ..
                } = self.time_manager.config_mut()
                {
                    *timestep_target_frame = sph::AdaptiveTimeStepTarget::TargetFrameLength(TARGET_FRAME_SIMDURATION);
                    epsilon = *timestep_min * 0.5;
                } else {
                    epsilon = 1.0e-9;
                }
                let target_simulation_time = self.time_manager.passed_time() + TARGET_FRAME_SIMDURATION - epsilon;
                while self.time_manager.passed_time() < target_simulation_time {
                    self.single_sim_step();
                }
            }
        }

        microprofile::flip!();
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        microprofile::scope!("MainState", "draw");

        graphics::clear(ctx, [0.4, 0.4, 0.45, 1.0].into());
        graphics::push_transform(ctx, Some(self.camera.transformation_matrix()));
        graphics::apply_transformations(ctx)?;

        self.draw_fluid(ctx)?;
        self.draw_text(ctx)?;

        {
            microprofile::scope!("MainState", "present");
            graphics::present(ctx)?;
        }

        if self.update_mode == UpdateMode::Recording {
            microprofile::scope!("MainState", "screenshot");
            ggez::filesystem::create_dir(ctx, "/recording")?;
            let img;
            {
                microprofile::scope!("MainState", "gpu transfer");
                img = graphics::screenshot(ctx).expect("Could not take screenshot");
            }
            {
                // todo: this png encoder/writer here is slow. have something faster.
                microprofile::scope!("MainState", "save as png");
                img.encode(ctx, graphics::ImageFormat::Png, format!("/recording/{}.png", self.frame_counter))
                    .expect("Could not save screenshot");
            }
        }
        self.frame_counter += 1;

        Ok(())
    }
}
