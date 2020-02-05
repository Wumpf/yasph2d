use ggez::event::{self, EventHandler};
use ggez::graphics::Rect;
use ggez::nalgebra as na;
use ggez::{conf, graphics, timer, Context, GameResult};

use na::clamp;

use std::time::{Duration, Instant};

mod camera;
mod sph;
mod units;

use camera::*;
use sph::*;
use units::*;

use std::collections::VecDeque;

fn main() -> GameResult {
    let context_builder = ggez::ContextBuilder::new("2d sph", "AndreasR")
        .window_setup(conf::WindowSetup::default().title("2d sph").samples(conf::NumSamples::Eight))
        .window_mode(conf::WindowMode::default().dimensions(1920.0, 1080.0));
    let (ctx, event_loop) = &mut context_builder.build()?;
    let state = &mut MainState::new(ctx);
    event::run(ctx, event_loop, state)
}

struct MainState {
    fluid_world: FluidParticleWorld,
    sph_solver: Box<dyn Solver>,

    camera: Camera,
    simulation_step_duration_history: VecDeque<Duration>,
    last_frame_simulation_duration: Duration,
    simulationstep_count: u32,
}

const SIMULATION_STEP_HISTORY_LENGTH: usize = 200;

impl MainState {
    pub fn new(ctx: &mut Context) -> MainState {
        let mut fluid_world = FluidParticleWorld::new(
            1.2,    // smoothing factor
            2500.0, // #particles/m²
            100.0,  // density of water (? this is 2d, not 3d where it's 1000 kg/m³)... want this to be 100, but lowered for stability
        );
        fluid_world.add_fluid_rect(&Rect::new(0.1, 0.1, 0.5, 0.8), 0.05);
        fluid_world.add_boundary_line(Point::new(0.0, 0.0), Point::new(1.5, 0.0));
        fluid_world.add_boundary_line(Point::new(0.0, 0.0), Point::new(0.0, 1.5));
        fluid_world.add_boundary_line(Point::new(1.5, 0.0), Point::new(1.5, 1.5));

        let mut xsph = XSPHViscosityModel::new(fluid_world.smoothing_length());
        xsph.epsilon = 0.1;
        let mut physicalviscosity = PhysicalViscosityModel::new(fluid_world.smoothing_length());
        physicalviscosity.fluid_viscosity = 0.01;
        let sph_solver = WCSPHSolver::new(xsph, fluid_world.smoothing_length());
        //let sph_solver = DFSPHSolver::new(xsph, fluid_world.smoothing_length());

        MainState {
            fluid_world,
            sph_solver: Box::new(sph_solver),
            camera: Camera::center_around_world_rect(graphics::screen_coordinates(ctx), Rect::new(-0.1, -0.1, 1.7, 1.6)),
            simulation_step_duration_history: VecDeque::with_capacity(SIMULATION_STEP_HISTORY_LENGTH),
            last_frame_simulation_duration: Default::default(),
            simulationstep_count: 0,
        }
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

impl EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        const DESIRED_UPDATES_PER_SECOND: u32 = 60 * 20;
        const TIME_STEP: Real = 1.0 / (DESIRED_UPDATES_PER_SECOND as Real);

        self.simulationstep_count = 0;

        let time_sim_start = std::time::Instant::now();
        while timer::check_update_time(ctx, DESIRED_UPDATES_PER_SECOND) {
            let time_step_start = std::time::Instant::now();

            if timer::ticks(ctx) < 80 {
                // warmup frames to avoid visible stuttering on startup. TODO: Why do we need them and why os many?
                self.sph_solver.simulation_step(&mut self.fluid_world, 0.000_000_000_1);
            } else {
                self.sph_solver.simulation_step(&mut self.fluid_world, TIME_STEP);
            }
            self.simulationstep_count += 1;
            
            if self.simulation_step_duration_history.len() == SIMULATION_STEP_HISTORY_LENGTH {
                self.simulation_step_duration_history.pop_front();
            }
            self.simulation_step_duration_history.push_back(Instant::now() - time_step_start);
        }
        self.last_frame_simulation_duration = Instant::now() - time_sim_start;
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        graphics::clear(ctx, [0.4, 0.4, 0.45, 1.0].into());
        graphics::push_transform(ctx, Some(self.camera.transformation_matrix()));
        graphics::apply_transformations(ctx)?;

        let particle_radius = self.fluid_world.suggested_particle_render_radius();
        let particle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            na::Point2::new(0.0, 0.0),
            particle_radius as f32,
            0.0003,
            graphics::WHITE,
        )?;
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
            .zip(self.fluid_world.particles.accellerations.iter())
        {
            let c = heatmap_color((a.norm() * 0.01) as f32);
            let rp: RenderPoint = na::convert(*p);
            graphics::draw(ctx, &particle, ggez::graphics::DrawParam::default().dest(rp).color(c))?;
        }
        for p in self.fluid_world.particles.boundary_particles.iter() {
            let rp: RenderPoint = na::convert(*p);
            graphics::draw(ctx, &particle, ggez::graphics::DrawParam::default().dest(rp).color(boundary_color))?;
        }

        graphics::pop_transform(ctx);
        graphics::apply_transformations(ctx)?;

        {
            let fps = timer::fps(ctx);
            let average_simulation_step_duration = self.simulation_step_duration_history.iter().sum::<Duration>() / self.simulation_step_duration_history.len() as u32;

            let fps_display = graphics::Text::new(format!(
                "{:.2}ms, FPS: {:.2}\nSim duration: {:.2}ms ({} steps)| Single Step (averaged): {:.2}ms",
                1000.0 / fps,
                fps,
                self.last_frame_simulation_duration.as_secs_f64() * 1000.0,
                self.simulationstep_count,
                average_simulation_step_duration.as_secs_f64() * 1000.0,
            ));
            graphics::draw(ctx, &fps_display, (na::Point2::new(10.0, 10.0), graphics::WHITE))?;
        }

        graphics::present(ctx)?;
        Ok(())
    }
}
