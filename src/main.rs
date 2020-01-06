use ggez::event::{self, EventHandler};
use ggez::nalgebra as na;
use ggez::{conf, graphics, timer, Context, GameResult};

fn main() -> GameResult {
    let context_builder = ggez::ContextBuilder::new("2d sph", "AndreasR")
        .window_setup(conf::WindowSetup::default()
            .title("2d sph")
            .samples(conf::NumSamples::Eight))
        .window_mode(conf::WindowMode::default().dimensions(1920.0, 1080.0));
    let (ctx, event_loop) = &mut context_builder.build()?;
    let state = &mut MainState::new(ctx);
    event::run(ctx, event_loop, state)
}

struct HydroParticles {
    positions: Vec<na::Point2<f32>>,
}

struct MainState {
    particles: HydroParticles, // box?
}

impl MainState {
    pub fn new(_ctx: &mut Context) -> MainState {
        let num_x = 40;
        let num_y = 20;
        let num_particles = num_x * num_y;

        let mut state = MainState {
            particles: HydroParticles {
                positions: Vec::with_capacity(num_particles),
            },
        };

        let dist = 10.0;
        for y in 0..num_y {
            for x in 0..num_x {
                //let i = x + y * num_x;
                state.particles.positions.push(na::Point2::new(dist * (x as f32), dist * (y as f32)));
            }
        }

        state
    }

    pub fn physics_step(&mut self, dt: f32) {
        for pos in &mut self.particles.positions {
            pos.x += dt * 100.0;
        }
    }
}

impl EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult {
        const DESIRED_UPDATES_PER_SECOND: u32 = 60;
        const TIME_STEP: f32 = 1.0 / (DESIRED_UPDATES_PER_SECOND as f32);

        while timer::check_update_time(ctx, DESIRED_UPDATES_PER_SECOND) {
            self.physics_step(TIME_STEP);
        }
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        graphics::clear(ctx, [0.1, 0.2, 0.3, 1.0].into());

        let fps = timer::fps(ctx);
        let fps_display = graphics::Text::new(format!("{:.2}ms, FPS: {:.2}", 1000.0 / fps, fps));
        graphics::draw(
            ctx,
            &fps_display,
            (na::Point2::new(10.0, 10.0), graphics::WHITE),
        )?;

        let circle = graphics::Mesh::new_circle(ctx, graphics::DrawMode::fill(), na::Point2::new(0.0, 0.0), 5.0, 1.0, graphics::WHITE)?;
        for p in self.particles.positions.iter() {
            graphics::draw(ctx, &circle, ggez::graphics::DrawParam::default().dest(*p))?;
        }

        graphics::present(ctx)?;
        Ok(())
    }
}
