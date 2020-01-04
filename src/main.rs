use ggez::event::{self, EventHandler};
use ggez::nalgebra as na;
use ggez::{conf, graphics, Context, GameResult};

fn main() -> GameResult {
    let context_builder = ggez::ContextBuilder::new("2d sph", "AndreasR")
        .window_setup(conf::WindowSetup::default().title("2d sph"))
        .window_mode(conf::WindowMode::default().dimensions(1920.0, 1080.0));
    let (ctx, event_loop) = &mut context_builder.build()?;
    let state = &mut MainState::new(ctx);
    event::run(ctx, event_loop, state)
}

struct MainState {
    pos_x: f32,
}

impl MainState {
    pub fn new(_ctx: &mut Context) -> MainState {
        MainState { pos_x: 0.0 }
    }
}

impl EventHandler for MainState {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        self.pos_x = self.pos_x % 800.0 + 1.0;
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        graphics::clear(ctx, [0.1, 0.2, 0.3, 1.0].into());

        let circle = graphics::Mesh::new_circle(
            ctx,
            graphics::DrawMode::fill(),
            na::Point2::new(0.0, 0.0),
            100.0,
            2.0,
            graphics::WHITE,
        )?;
        graphics::draw(ctx, &circle, (na::Point2::new(self.pos_x, 380.0),))?;

        graphics::present(ctx)?;
        Ok(())
    }
}
