use ggez::event::{self, EventHandler};
use ggez::nalgebra as na;
use ggez::{conf, graphics, timer, Context, GameResult};

mod camera;
mod hydroparticles;
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
    particles: HydroParticles, // box?
    camera: Camera,
}

impl MainState {
    pub fn new(ctx: &mut Context) -> MainState {
        MainState {
            particles: HydroParticles::new(40, 20),
            camera: Camera {
                screen: graphics::screen_coordinates(ctx),
                pixel_per_world_unit: 10.0,
                position: Position::origin(),
            },
        }
    }
}

impl EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        const DESIRED_UPDATES_PER_SECOND: u32 = 60;
        const TIME_STEP: f32 = 1.0 / (DESIRED_UPDATES_PER_SECOND as f32);

        while timer::check_update_time(ctx, DESIRED_UPDATES_PER_SECOND) {
            self.particles.physics_step(TIME_STEP);
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        graphics::clear(ctx, [0.1, 0.2, 0.3, 1.0].into());

        let fps = timer::fps(ctx);
        let fps_display = graphics::Text::new(format!("{:.2}ms, FPS: {:.2}", 1000.0 / fps, fps));
        graphics::draw(ctx, &fps_display, (na::Point2::new(10.0, 10.0), graphics::WHITE))?;

        let circle = graphics::Mesh::new_circle(ctx, graphics::DrawMode::fill(), na::Point2::new(0.0, 0.0), 0.5, 0.1, graphics::WHITE)?;
        for p in self.particles.positions.iter() {
            graphics::draw(
                ctx,
                &circle,
                ggez::graphics::DrawParam::default()
                    .scale(self.camera.world_unit_scale())
                    .dest(self.camera.world_to_screen_coords(*p)),
            )?;
        }

        graphics::present(ctx)?;
        Ok(())
    }
}
