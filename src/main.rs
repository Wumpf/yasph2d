use ggez::event::{self, EventHandler};
use ggez::graphics::Rect;
use ggez::nalgebra as na;
use ggez::{conf, graphics, timer, Context, GameResult};

use na::clamp;

use std::time::{Duration, Instant};

mod camera;
mod hydroparticles;
mod smoothing_kernel;
mod units;

use camera::*;
use hydroparticles::*;
use units::*;

fn main() -> GameResult {
    let context_builder = ggez::ContextBuilder::new("2d sph", "AndreasR")
        .window_setup(conf::WindowSetup::default().title("2d sph").samples(conf::NumSamples::Eight))
        .window_mode(conf::WindowMode::default().dimensions(1920.0, 1080.0));
    let (ctx, event_loop) = &mut context_builder.build()?;
    let state = &mut MainState::new(ctx);
    event::run(ctx, event_loop, state)
}

struct MainState {
    particles: HydroParticles,
    camera: Camera,
    last_simulationstep_duration: Duration,
    last_simulationstep_count: u32,
}

impl MainState {
    pub fn new(ctx: &mut Context) -> MainState {
        let mut particles = HydroParticles::new(
            1.2,    // smoothing factor
            2000.0, // #particles/m²
            100.0,  // density of water (? this is 2d, not 3d where it's 1000 kg/m³)
            0.5,    //1500.0, // speed of sound in water in m/s
            0.5,    //1.0016 / 1000.0, // viscosity of water at 20 degrees in Pa*s
        );
        particles.add_fluid_rect(&Rect::new(0.2, 0.4, 0.6, 0.6));
        particles.add_boundary_line(&Position::new(0.0, 0.0), &Position::new(1.0, 0.0));
        particles.add_boundary_line(&Position::new(0.0, 0.0), &Position::new(0.0, 0.75));
        particles.add_boundary_line(&Position::new(1.0, 0.0), &Position::new(1.0, 0.75));

        MainState {
            particles: particles,
            camera: Camera::center_around_world_rect(graphics::screen_coordinates(ctx), Rect::new(-0.1, -0.1, 1.1, 1.1)),
            last_simulationstep_duration: Default::default(),
            last_simulationstep_count: 0,
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
        const DESIRED_UPDATES_PER_SECOND: u32 = 240;
        const TIME_STEP: f32 = 1.0 / (DESIRED_UPDATES_PER_SECOND as f32);

        self.last_simulationstep_count = 0;
        while timer::check_update_time(ctx, DESIRED_UPDATES_PER_SECOND) {
            let time_start = std::time::Instant::now();
            self.particles.physics_step(TIME_STEP);
            self.last_simulationstep_duration = Instant::now() - time_start;
            self.last_simulationstep_count += 1;
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        graphics::clear(ctx, [0.4, 0.4, 0.45, 1.0].into());

        let fps = timer::fps(ctx);

        let fps_display = graphics::Text::new(format!(
            "{:.2}ms, FPS: {:.2}\nSimStep duration {:.2}ms ({} per frame)",
            1000.0 / fps,
            fps,
            self.last_simulationstep_duration.as_secs_f64() * 1000.0,
            self.last_simulationstep_count
        ));
        graphics::draw(ctx, &fps_display, (na::Point2::new(10.0, 10.0), graphics::WHITE))?;

        graphics::push_transform(ctx, Some(self.camera.transformation_matrix()));
        graphics::apply_transformations(ctx)?;

        let particle_radius = self.particles.suggested_particle_render_radius();
        let particle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            na::Point2::new(0.0, 0.0),
            particle_radius,
            0.0003,
            graphics::WHITE,
        )?;
        let boundary_color = graphics::Color {
            r: 0.2,
            g: 0.2,
            b: 0.2,
            a: 1.0,
        };
        for (p, f) in self.particles.positions.iter().zip(self.particles.forces.iter()) {
            let c = heatmap_color(f.norm() / self.particles.particle_mass());
            graphics::draw(ctx, &particle, ggez::graphics::DrawParam::default().dest(*p).color(c))?;
        }
        for p in self.particles.boundary_particles.iter() {
            graphics::draw(ctx, &particle, ggez::graphics::DrawParam::default().dest(*p).color(boundary_color))?;
        }

        graphics::pop_transform(ctx);
        graphics::apply_transformations(ctx)?;

        graphics::present(ctx)?;
        Ok(())
    }
}
