use ggez::nalgebra as na;

type Scalar = f32;

pub type Position = na::Point2<Scalar>;
pub type Direction = na::Vector2<Scalar>;
pub type Velocity = Direction;
pub type Size = Direction;
