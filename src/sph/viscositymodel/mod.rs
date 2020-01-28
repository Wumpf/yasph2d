pub use viscositymodel::ViscosityModel;
pub use xsph::XSPHViscosityModel;
pub use physical::PhysicalViscosityModel;

mod xsph;
mod physical;
mod viscositymodel;