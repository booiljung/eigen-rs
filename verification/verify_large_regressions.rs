use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;
use std::time::Instant;

fn main() {
    let size = 65536;
    let iterations = 10000;

    let mut a = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(size, 1).unwrap();
    let mut b = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(size, 1).unwrap();

    // Initialize with something non-trivial
    for i in 0..size {
        *a.get_mut(i, 0).unwrap() = 1.0;
        *b.get_mut(i, 0).unwrap() = 2.0;
    }

    // Warmup
    for _ in 0..100 {
        let _ = a.dot(&b);
    }

    let start = Instant::now();
    let mut black_box = 0.0;
    for _ in 0..iterations {
        black_box += a.dot(&b);
    }
    let duration = start.elapsed();

    println!("Vector Size: {}", size);
    println!("Iterations: {}", iterations);
    println!("Total Time: {:?}", duration);
    println!("Result (to prevent opt): {}", black_box);
}
