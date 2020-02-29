use cgmath::prelude::*;
use ggez::event::{self, EventHandler, KeyCode, KeyMods};
use ggez::graphics::Rect;
use ggez::{conf, graphics, timer, Context, GameResult};
use microprofile;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

mod camera;

use camera::*;
use yasph2d::sph::*;
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

struct MainState {
    update_mode: UpdateMode,

    fluid_world: FluidParticleWorld,
    sph_solver: Box<dyn Solver>,

    camera: Camera,
    particle_mesh: graphics::Mesh,

    simulation_step_duration_history: VecDeque<Duration>,
    simulation_processing_time_frame: Duration,
    simulationstep_count_frame: u32,

    simulation_starttime: Instant,
    simulation_realtime_total: Duration,
    simulation_processing_time_total: Duration,

    frame_counter: usize,
}

const SIMULATION_STEP_HISTORY_LENGTH: usize = 80;

const REALTIME_TO_SIMTIME: f32 = 1.0;
const NUM_DESIRED_SIM_UPDATES_PER_SECOND: u32 = 60 * 20;
const SIM_TIME_STEP: Real = REALTIME_TO_SIMTIME / (NUM_DESIRED_SIM_UPDATES_PER_SECOND as Real);
// Number of seconds simulation is allowed to process before slowing down physics time.
const MAX_ALLOWED_SIM_PROCESSING_TIME: Real = 1.0 / 15.0; // if render takes 0 time this would result in min 15fps (since it doesn't, real min fps is lower)

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
        let mut fluid_world = FluidParticleWorld::new(
            2.0,    // smoothing factor
            5000.0, // #particles/m²
            100.0,  // density of water (? this is 2d, not 3d where it's 1000 kg/m³)
        );
        Self::reset_fluid(&mut fluid_world);
        let xsph = XSPHViscosityModel::new(fluid_world.properties.smoothing_length());
        //xsph.epsilon = 0.1;
        let mut physicalviscosity = PhysicalViscosityModel::new(fluid_world.properties.smoothing_length());
        physicalviscosity.fluid_viscosity = 0.01;

        //let sph_solver = WCSPHSolver::new(xsph, fluid_world.properties.smoothing_length());
        let sph_solver = DFSPHSolver::new(xsph, fluid_world.properties.smoothing_length());

        let particle_radius = fluid_world.properties.suggested_particle_render_radius();
        let particle_mesh = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            RenderPoint::origin(),
            particle_radius as f32,
            0.0003,
            graphics::WHITE,
        )
        .unwrap();

        MainState {
            update_mode: UpdateMode::RealTime,
            fluid_world,
            sph_solver: Box::new(sph_solver),

            camera: Camera::center_around_world_rect(graphics::screen_coordinates(ctx), Rect::new(-0.1, -0.1, 2.1, 1.6)),
            particle_mesh,

            simulation_step_duration_history: VecDeque::with_capacity(SIMULATION_STEP_HISTORY_LENGTH),
            simulation_processing_time_frame: Default::default(),
            simulationstep_count_frame: 0,

            simulation_starttime: Instant::now(),
            simulation_realtime_total: Default::default(),
            simulation_processing_time_total: Default::default(),

            frame_counter: 0,
        }
    }

    fn reset_fluid(fluid_world: &mut FluidParticleWorld) {
        fluid_world.remove_all_fluid_particles();
        fluid_world.remove_all_boundary_particles();

        fluid_world.add_fluid_rect(&Rect::new(0.1, 0.7, 0.5, 1.0), 0.05);
        fluid_world.add_boundary_line(Point::new(0.0, 0.0), Point::new(2.0, 0.0));
        fluid_world.add_boundary_line(Point::new(0.0, 0.0), Point::new(0.0, 2.5));
        fluid_world.add_boundary_line(Point::new(2.0, 0.0), Point::new(2.0, 2.5));

        fluid_world.add_boundary_line(Point::new(0.0, 0.6), Point::new(1.75, 0.5));
    }

    fn draw_text(&mut self, ctx: &mut Context) -> GameResult {
        microprofile::scope!("MainState", "text");

        let fps = timer::fps(ctx);
        let average_simulation_step_duration =
            self.simulation_step_duration_history.iter().sum::<Duration>() / self.simulation_step_duration_history.len() as u32;

        let simulation_info_text = format!(
            "Sim duration: {:3.2}ms ({:4} steps)| Single Step (averaged over {}): {:.2}ms\nTotal Simulated Time {:.2}\nTotal Processing Time {:.2}",
            self.simulation_processing_time_frame.as_secs_f64() * 1000.0,
            self.simulationstep_count_frame,
            self.simulation_step_duration_history.len(),
            average_simulation_step_duration.as_secs_f64() * 1000.0,
            self.simulation_realtime_total.as_secs_f64(),
            self.simulation_processing_time_total.as_secs_f64(),
        );

        let fps_display = graphics::Text::new(match self.update_mode {
            UpdateMode::RealTime => format!(
                "{:3.2}ms, FPS: {:3.2}, passed time {:.2}s\n{}",
                1000.0 / fps,
                fps,
                (Instant::now() - self.simulation_starttime).as_secs_f64(),
                simulation_info_text,
            ),

            UpdateMode::Recording => format!("RECORDING\n{}", simulation_info_text,),
        });
        graphics::draw(ctx, &fps_display, (RenderPoint::new(10.0, 10.0), graphics::WHITE))?;

        if self.update_mode == UpdateMode::RealTime && self.simulation_processing_time_frame.as_secs_f32() > MAX_ALLOWED_SIM_PROCESSING_TIME {
            graphics::draw(
                ctx,
                &graphics::Text::new("REALTIME OFF (max sim processing time hit)"),
                (RenderPoint::new(10.0, 70.0), graphics::Color::new(1.0, 0.2, 0.2, 1.0)),
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

    fn single_sim_step(&mut self, current_time: &mut Instant) {
        let time_step_start = *current_time;

        self.sph_solver.simulation_step(&mut self.fluid_world, SIM_TIME_STEP);
        *current_time = Instant::now();

        let step_time = *current_time - time_step_start;
        self.simulation_processing_time_frame += step_time;
        self.simulation_processing_time_total += step_time;
        self.simulation_realtime_total += Duration::from_secs_f64(SIM_TIME_STEP as f64);
        self.simulationstep_count_frame += 1;

        if self.simulation_step_duration_history.len() == SIMULATION_STEP_HISTORY_LENGTH {
            self.simulation_step_duration_history.pop_front();
        }
        self.simulation_step_duration_history.push_back(step_time);
    }

    fn reset_simulation(&mut self, ctx: &mut Context) {
        self.sph_solver.clear_cached_data(); // todo: this is super meh
        self.simulation_starttime = Instant::now();
        self.simulation_realtime_total = Default::default();
        self.simulation_processing_time_total = Default::default();
        self.frame_counter = 0;
        Self::reset_fluid(&mut self.fluid_world);

        // reset residual timer
        // todo: patch ggez to handle this better
        let pseudo_target_fps = (1.0 / timer::remaining_update_time(ctx).as_secs_f64()) as u32;
        if pseudo_target_fps > 0 {
            while timer::check_update_time(ctx, pseudo_target_fps) {}
        }
    }
}

impl EventHandler for MainState {
    fn key_down_event(&mut self, ctx: &mut Context, keycode: KeyCode, _keymods: KeyMods, repeat: bool) {
        match keycode {
            KeyCode::Escape => {
                ggez::event::quit(ctx);
            }
            KeyCode::Space => {
                self.reset_simulation(ctx);
            }
            KeyCode::R => {
                if !repeat {
                    self.update_mode = if self.update_mode == UpdateMode::RealTime {
                        UpdateMode::Recording
                    } else {
                        UpdateMode::RealTime
                    };
                    self.reset_simulation(ctx);
                }
            }
            _ => {
                self.update_mode = UpdateMode::RealTime;
            }
        }
    }

    fn update(&mut self, ctx: &mut Context) -> GameResult {
        microprofile::scope!("MainState", "update");

        self.simulationstep_count_frame = 0;
        self.simulation_processing_time_frame = Duration::from_secs(0);

        match self.update_mode {
            UpdateMode::RealTime => {
                let mut current_time = Instant::now();
                while timer::check_update_time(ctx, NUM_DESIRED_SIM_UPDATES_PER_SECOND) {
                    // if self.simulation_realtime_total > Duration::from_secs(2) {
                    //     break;
                    // }

                    self.single_sim_step(&mut current_time);

                    // If we can't process fast enough, consume the remaining residual time.
                    // I.e. we give up on getting physics-time on par to real time.
                    // If we would just break here, check_update_time will keep on trying to catch up!
                    if self.simulation_processing_time_frame.as_secs_f32() > MAX_ALLOWED_SIM_PROCESSING_TIME {
                        while timer::check_update_time(ctx, NUM_DESIRED_SIM_UPDATES_PER_SECOND) {}
                        break;
                    }
                }
            }

            UpdateMode::Recording => {
                const RECORDING_FPS: u32 = 60;
                let mut current_time = Instant::now();
                for _ in 0..(NUM_DESIRED_SIM_UPDATES_PER_SECOND / RECORDING_FPS) {
                    self.single_sim_step(&mut current_time);
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
            let img = graphics::screenshot(ctx).expect("Could not take screenshot");
            img.encode(ctx, graphics::ImageFormat::Png, format!("/recording/{}.png", self.frame_counter)).expect("Could not save screenshot");
        }
        self.frame_counter += 1;

        Ok(())
    }
}
