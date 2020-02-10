pub use self::cubic::CubicSpline;
/// Smoothing Kernels.
pub use self::kernel::Kernel;
pub use self::poly6::Poly6;
pub use self::spiky::Spiky;
pub use self::viscosity::Viscosity;

#[macro_use]
mod kernel;
mod cubic;
mod poly6;
mod spiky;
mod viscosity;
