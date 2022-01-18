use criterion::criterion_main;
mod benchmarks;

criterion_main! {
    benchmarks::smoothing_kernel::smoothing_kernel,
    benchmarks::morton::morton,
    benchmarks::neighborhood_search::neighborhood_search,
    benchmarks::update_densities::update_densities,
}
