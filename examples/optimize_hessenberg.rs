use eigen_rs::core::matrix::MatrixX;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let size = 256;
    println!("Benchmarking Hessenberg Optimization (N={})", size);

    let mut a = MatrixX::<f64>::new_dynamic(size, size).unwrap();
    // Fill random non-symmetric
    for i in 0..size * size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 17) as f64 * 0.3 + (i % 5) as f64;
    }

    // Warmup
    for _ in 0..5 {
        black_box(eigen_rs::core::decompositions::HessenbergDecomposition::new(&a).unwrap());
    }

    let start = Instant::now();
    let iterations = 20;
    for _ in 0..iterations {
        black_box(eigen_rs::core::decompositions::HessenbergDecomposition::new(&a).unwrap());
    }
    let duration = start.elapsed();
    println!(
        "Hessenberg Time: {:.2} ms per iter",
        duration.as_secs_f64() * 1000.0 / iterations as f64
    );
}
