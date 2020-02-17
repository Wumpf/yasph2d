use criterion::criterion_main;
mod benchmarks;

criterion_main! {
    benchmarks::smoothing_kernel::smoothing_kernel,
    benchmarks::morton::morton
}