YaSPH2D
===========================
"Yet another SPH" .. thing.. in 2D!\
Playing around with Smoothed Particle Hydrodynamics simulation in Rust. All 2d for now.

Has something of a library, but right now the separation with the test application is too loose for that (also lack of docs and tests to call it a real library ;-)). Not sure where it will end up, maybe it's gonna stay this way, maybe I will end up making game, maybe I focus on non-interactive simulations instead.
Not even sure yet if I ditch the 2d thing and make it 3d.
(Note: Mixing is possible but seems like a bad idea since some parameters go all over the place and it makes it really hard to do the _actually cool_ optimizations ;-))

Implements solvers using
* Weakly Compressible SPH (WCSPH)
* DFSPH (wip, see #1)
  * [Bender & Koschier 2015, Divergence-Free Smoothed Particle Hydrodynamicss](https://animation.rwth-aachen.de/publication/054/)  
  * [Bender & Koschier 2017, Divergence-Free SPH for Incompressible and Viscous Fluids](https://animation.rwth-aachen.de/publication/051/)

Nearest neighbor search using ideas from [Compressed Neighbour Lists for SPH, Stefan Band et al.](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13890). Actual compression is WIP (see #3)

Some more links to resources in the code.

To find even more resources about fluid simulation in general check out [my gist on CFD](https://gist.github.com/Wumpf/b3e953984de8b0efdf2c65e827a1ccc3) where I continously gather links and short descriptions on various concepts.