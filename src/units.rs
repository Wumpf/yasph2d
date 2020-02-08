use ggez::nalgebra as na;

// For simulating
pub type Real = f32;
pub type Point = na::Point2<Real>;
pub type Vector = na::Vector2<Real>;

// For rendering
pub type RenderPoint = cgmath::Point2<f32>;
pub type RenderSize = cgmath::Vector2<f32>;
