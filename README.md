YaSPH2D
===========================
"Yet another SPH" .. thing.. in 2D!\
Playing around with Smoothed Particle Hydrodynamics simulation in Rust. All 2d for now.

Implements solvers using
* Weakly Compressible SPH (WCSPH)
* DFSPH
  * [Bender & Koschier 2015, Divergence-Free Smoothed Particle Hydrodynamicss](https://animation.rwth-aachen.de/publication/054/)  
  * [Bender & Koschier 2017, Divergence-Free SPH for Incompressible and Viscous Fluids](https://animation.rwth-aachen.de/publication/051/)

Nearest neighbor search using ideas from [Compressed Neighbour Lists for SPH, Stefan Band et al.](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13890). Actual compression is WIP (see #3)

Some more links to resources in the code.

To find even more resources about fluid simulation in general check out [my gist on CFD](https://gist.github.com/Wumpf/b3e953984de8b0efdf2c65e827a1ccc3) where I continously gather links and short descriptions on various concepts.