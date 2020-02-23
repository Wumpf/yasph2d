use crate::units::*;
use cgmath::prelude::*;
use ggez::graphics::Rect;
use rand::prelude::*;
use rayon::prelude::*;

use super::neighborhood_search::NeighborhoodSearch;
use super::smoothing_kernel::Kernel;

pub struct Particles {
    pub positions: Vec<Point>,
    pub velocities: Vec<Vector>,

    // Local densities ρ
    // typically recomputed every frame
    pub densities: Vec<Real>,

    // also called "shadow particles", immovable particles used for boundaries
    pub boundary_particles: Vec<Point>,

    neighborhood: NeighborhoodSearch,
}

impl Particles {
    const OVERLAP_THRESHOLD: Real = 0.00001;

    pub(super) fn foreach_neighbor_particle(&self, smoothing_length_sq: Real, ri: Point, f: impl FnMut(usize, Real, Vector) -> ()) {
        Self::foreach_neighbor_particle_internal(&self.neighborhood, &self.positions, smoothing_length_sq, ri, f)
    }

    fn foreach_neighbor_particle_internal(
        neighborhood: &NeighborhoodSearch,
        positions: &[Point],
        smoothing_length_sq: Real,
        ri: Point,
        mut f: impl FnMut(usize, Real, Vector) -> (),
    ) {
        neighborhood.foreach_potential_neighbor(
            ri,
            #[inline]
            |j| {
                let rj = positions[j as usize];
                let ri_to_rj = rj - ri;
                let r_sq = ri_to_rj.magnitude2();
                if r_sq > smoothing_length_sq || r_sq < Self::OVERLAP_THRESHOLD {
                    // Skips self and and degenerated overlaps
                    return;
                }
                f(j as usize, r_sq, ri_to_rj);
            },
        );
    }

    pub(super) fn foreach_neighbor_particle_boundary(&self, smoothing_length_sq: Real, ri: Point, f: impl FnMut(Real, Vector) -> ()) {
        Self::foreach_neighbor_particle_boundary_internal(&self.neighborhood, &self.boundary_particles, smoothing_length_sq, ri, f)
    }

    fn foreach_neighbor_particle_boundary_internal(
        neighborhood: &NeighborhoodSearch,
        positions: &[Point],
        smoothing_length_sq: Real,
        ri: Point,
        mut f: impl FnMut(Real, Vector) -> (),
    ) {
        neighborhood.foreach_potential_boundary_neighbor(
            ri,
            #[inline]
            |j| {
                let rj = positions[j as usize];
                let ri_to_rj = rj - ri;
                let r_sq = ri_to_rj.magnitude2();
                if r_sq > smoothing_length_sq {
                    // Skips self and and degenerated overlaps
                    return;
                }
                f(r_sq, ri_to_rj);
            },
        );
    }
}

pub struct ConstantFluidProperties {
    smoothing_length: Real, // typically expressed as 'h'
    particle_density: Real, // #particles/m² for resting fluid
    fluid_density: Real,    // kg/m² for the resting fluid (ρ, rho)
}

impl ConstantFluidProperties {
    fn new(
        smoothing_factor: Real,
        particle_density: Real, // #particles/m² for resting fluid
        fluid_density: Real,    // kg/m² for the resting fluid) {
    ) -> ConstantFluidProperties { 
        let smoothing_length = 2.0 * Self::particle_radius_from_particle_density(particle_density) * smoothing_factor;
        ConstantFluidProperties {
            smoothing_length,
            particle_density,
            fluid_density,
        }
    }

    pub fn smoothing_length(&self) -> Real {
        self.smoothing_length
    }

    pub fn fluid_density(&self) -> Real {
        self.fluid_density
    }

    pub fn particle_mass(&self) -> Real {
        self.fluid_density / self.particle_density
    }

    fn particle_radius_from_particle_density(particle_density: Real) -> Real {
        // density is per m²
        0.5 / particle_density.sqrt()
    }

    fn num_particles_per_meter(&self) -> Real {
        self.particle_density.sqrt()
    }

    pub fn suggested_particle_render_radius(&self) -> Real {
        Self::particle_radius_from_particle_density(self.particle_density)
    }
}

pub struct FluidParticleWorld {
    pub particles: Particles,
    pub properties: ConstantFluidProperties,

    pub gravity: Vector, // global gravity force in m/s² (== N/kg)

    boundary_changed: bool,
}
impl FluidParticleWorld {
    pub fn new(
        smoothing_factor: Real,
        particle_density: Real, // #particles/m² for resting fluid
        fluid_density: Real,    // kg/m² for the resting fluid
    ) -> FluidParticleWorld {
        let properties = ConstantFluidProperties::new(smoothing_factor, particle_density, fluid_density);
        FluidParticleWorld {
            particles: Particles {
                positions: Vec::new(),
                velocities: Vec::new(),
                densities: Vec::new(),

                boundary_particles: Vec::new(),

                neighborhood: NeighborhoodSearch::new(properties.smoothing_length()),
            },
            properties,

            gravity: Vector::new(0.0, -9.81),

            boundary_changed: true,
        }
    }

    pub fn remove_all_fluid_particles(&mut self) {
        self.particles.positions.clear();
        self.particles.densities.clear();
        self.particles.velocities.clear();
    }

    pub fn remove_all_boundary_particles(&mut self) {
        self.particles.boundary_particles.clear();
        self.particles.densities.clear();
        self.particles.velocities.clear();
    }

    /// - `jitter`: Amount of jitter. 0 for perfect lattice. >1 and particles are no longer in a strict lattice.
    pub fn add_fluid_rect(&mut self, fluid_rect: &Rect, jitter_amount: Real) {
        // fluid_rect.w * fluid_rect.h / self.particle_density, but discretized per axis
        let num_particles_per_meter = self.properties.num_particles_per_meter();
        let num_particles_x = std::cmp::max(1, (fluid_rect.w as Real * num_particles_per_meter) as usize);
        let num_particles_y = std::cmp::max(1, (fluid_rect.h as Real * num_particles_per_meter) as usize);
        let num_particles = num_particles_x * num_particles_y;

        self.particles.positions.reserve(num_particles);
        self.particles
            .velocities
            .resize(self.particles.velocities.len() + num_particles, Zero::zero());
        self.particles
            .densities
            .resize(self.particles.densities.len() + num_particles, Zero::zero());

        let mut rng: rand::rngs::SmallRng = rand::SeedableRng::seed_from_u64(self.particles.positions.len() as u64);

        let bottom_left = Point::new(fluid_rect.x as Real, fluid_rect.y as Real);
        let step = (fluid_rect.w as Real / (num_particles_x as Real)).min(fluid_rect.h as Real / (num_particles_y as Real));
        let jitter_factor = step * jitter_amount;
        for y in 0..num_particles_y {
            for x in 0..num_particles_x {
                let jitter = (rng.gen::<Vector>() * 0.5 + Vector::new(0.5, 0.5)) * jitter_factor;
                self.particles
                    .positions
                    .push(bottom_left + jitter + Vector::new(step * (x as Real), step * (y as Real)));
            }
        }
    }

    pub fn add_boundary_line(&mut self, start: Point, end: Point) {
        let distance = start.distance2(end);
        let num_particles_per_meter = self.properties.num_particles_per_meter();
        let num_shadow_particles = std::cmp::max(1, (distance * num_particles_per_meter) as usize);
        self.particles.boundary_particles.reserve(num_shadow_particles);
        let step = (end - start) / (num_shadow_particles as Real);

        let mut pos = start;
        for _ in 0..num_shadow_particles {
            self.particles.boundary_particles.push(pos);
            pos += step;
        }

        self.boundary_changed = true;
    }

    pub(super) fn update_densities(&mut self, kernel: impl Kernel + std::marker::Sync) {
        microprofile::scope!("FluidParticleWorld", "update_densities");
        assert_eq!(self.particles.positions.len(), self.particles.densities.len());

        let mass = self.properties.particle_mass();

        // Density contributions are symmetric, but that is hard to use in a parallel loop.
        let neighborhood = &self.particles.neighborhood;
        let smoothing_length_sq = self.properties.smoothing_length * self.properties.smoothing_length;
        let boundary_particles = &self.particles.boundary_particles;
        let positions = &self.particles.positions;

        self.particles
            .densities
            .par_iter_mut()
            .zip(positions.par_iter())
            .for_each(|(density, ri)| {
                *density = kernel.evaluate(0.0, 0.0) * mass; // self-contribution

                Particles::foreach_neighbor_particle_internal(
                    &neighborhood,
                    &positions,
                    smoothing_length_sq,
                    *ri,
                    #[inline(always)]
                    |_, r_sq, _| {
                        let density_contribution = kernel.evaluate(r_sq, r_sq.sqrt()) * mass;
                        *density += density_contribution;
                    },
                );
                Particles::foreach_neighbor_particle_boundary_internal(
                    &neighborhood,
                    &boundary_particles,
                    smoothing_length_sq,
                    *ri,
                    #[inline(always)]
                    |r_sq, _| {
                        let density_contribution = kernel.evaluate(r_sq, r_sq.sqrt()) * mass;
                        *density += density_contribution;
                    },
                );
            });
    }

    pub(super) fn update_neighborhood_datastructure(&mut self) {
        microprofile::scope!("FluidParticleWorld", "update_neighborhood_datastructure");
        self.particles.neighborhood.update(&self.particles.positions);

        if self.boundary_changed {
            self.particles.neighborhood.update_boundary(&self.particles.boundary_particles);
            self.boundary_changed = false;
        }
    }
}
