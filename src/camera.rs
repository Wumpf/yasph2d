use super::units::*;
use ggez::graphics::Rect;

// A 2D camera.
// Maps 2D world coordinates/sizes to screen coordinates/sizes.
//
// 2D World: ↑ y → x, origin bottom left
pub struct Camera {
    pub screen: Rect,              // Screen rectangle, as in https://docs.rs/ggez/0.5.1/ggez/graphics/fn.screen_coordinates.html
    pub pixel_per_world_unit: f32, // Scaling/Zoom factor of the camera ()
    pub position: Position,        // The position of this camera in world space, i.e. the middle of the view.
}

impl Camera {
    pub fn world_unit_scale(&self) -> Size {
        Size::new(self.pixel_per_world_unit,self.pixel_per_world_unit)
    }

    pub fn world_to_screen_coords(&self, world_pos: Position) -> Position {
        let from_camera = world_pos - self.position;
        let view_scale = from_camera * self.pixel_per_world_unit;

        Position::new(
            self.screen.x + view_scale.x + self.screen.w * 0.5,
            self.screen.y + self.screen.h.signum() * (view_scale.y + self.screen.h * 0.5),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_screen_conversion() {
        {
            let camera = Camera {
                screen: Rect::new(0.0, 0.0, 200.0, -100.0),
                pixel_per_world_unit: 10.0,
                position: Position::origin(),
            };

            assert_eq!(camera.world_to_screen_coords(Position::origin()), Position::new(100.0, 50.0));
            assert_eq!(camera.world_to_screen_coords(Position::new(1.0, 1.0)), Position::new(110.0, 40.0));
            assert_eq!(camera.world_to_screen_coords(Position::new(-1.0, -1.0)), Position::new(90.0, 60.0));
        }

        {
            let camera = Camera {
                screen: Rect::new(0.0, 0.0, 200.0, -100.0),
                pixel_per_world_unit: 10.0,
                position: Position::new(1.0, 1.0),
            };
            assert_eq!(camera.world_to_screen_coords(Position::origin()), Position::new(90.0, 60.0));
            assert_eq!(camera.world_to_screen_coords(Position::new(1.0, 1.0)), Position::new(100.0, 50.0));
            assert_eq!(camera.world_to_screen_coords(Position::new(-1.0, -1.0)), Position::new(80.0, 70.0));
        }
    }
}
