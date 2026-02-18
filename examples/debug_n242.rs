use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;
use std::time::Instant;
use std::ops::Mul; // Fix 1: Import Mul trait for .mul()

use eigen_rs::core::xpr::MatrixXpr; // Fix 2: Import MatrixXpr for .eval()

fn bench_matmul(n: usize) -> u128 {
    let mut a = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(n, n).unwrap();
    let mut b = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(n, n).unwrap();
    let mut c = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(n, n).unwrap();
    
    // Fill with pattern
    for i in 0..n {
        for j in 0..n {
            *a.get_mut(i, j).unwrap() = ((i + j) % 10) as f32;
            *b.get_mut(i, j).unwrap() = ((i * j) % 10) as f32;
        }
    }
    
    let iterations = 20;
    
    // Warmup
    let _ = (&a).mul(&b); // Use reference for Mul if implemented for &Matrix

    let start = Instant::now();
    for _ in 0..iterations {
        // Use assign() as seen in tests/matrix_mul_test.rs
        c.assign(&(&a).mul(&b)).unwrap();
    }
    let duration = start.elapsed();
    
    let avg_ns = duration.as_nanos() / iterations;
    
    // Validate to prevent DCE
    // Use manual sum if as_slice is missing
    let mut sum: f32 = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += *c.get(i, j).unwrap();
        }
    }
    if sum == 123456.789 { println!("Sum: {}", sum); }

    avg_ns
}

fn main() {
    println!("Benchmarking MatMul for N=242 (Problematic) vs N=256 (Baseline)...");
    
    // N=242
    let t_242 = bench_matmul(242);
    println!("N=242: {} ns", t_242);
    
    // N=256
    let t_256 = bench_matmul(256);
    println!("N=256: {} ns", t_256);
    
    let ratio = (t_242 as f64) / (t_256 as f64);
    println!("Ratio (242/256): {:.2}x (Expected ~1.0x or slightly less)", ratio);
    
    // Strides
    println!("N=242 Stride: {} bytes", 242 * 4);
    println!("N=256 Stride: {} bytes", 256 * 4);
}
