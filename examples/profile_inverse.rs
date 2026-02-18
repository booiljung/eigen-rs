use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;
use std::time::Instant;

fn main() {
    let n = 380;
    println!("Profiling Inverse for N={}", n);

    let mut a = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(n, n).unwrap();
    // Fill with random data
    for i in 0..n {
        for j in 0..n {
            *a.get_mut(i, j).unwrap() = ((i + j) % 10) as f32 + 1.0;
        }
    }
    // Make it invertible (diagonally dominant)
    for i in 0..n {
        *a.get_mut(i, i).unwrap() += n as f32;
    }

    let iterations = 20;

    // Warmup
    let _ = a.inverse();

    let start_total = Instant::now();
    for _ in 0..iterations {
        let _ = a.inverse();
    }
    let duration_total = start_total.elapsed();
    println!("Total Inverse Time (Avg): {} us", duration_total.as_micros() / iterations);

    // Profile Separation
    let start_lu = Instant::now();
    for _ in 0..iterations {
        let _ = a.partial_piv_lu();
    }
    let duration_lu = start_lu.elapsed();
    let avg_lu = duration_lu.as_micros() / iterations;
    println!("LU Factorization Time (Avg): {} us", avg_lu);

    let lu = a.partial_piv_lu().unwrap();
    let mut identity = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(n, n).unwrap();
    identity.set_identity();

    let start_solve = Instant::now();
    for _ in 0..iterations {
        let _ = lu.solve(&identity);
    }
    let duration_solve = start_solve.elapsed();
    let avg_solve = duration_solve.as_micros() / iterations;
    println!("Solve Time (Avg): {} us", avg_solve);

    println!("Breakdown:");
    println!("  LU:    {} us ({:.1}%)", avg_lu, (avg_lu as f64 / (avg_lu + avg_solve) as f64) * 100.0);
    println!("  Solve: {} us ({:.1}%)", avg_solve, (avg_solve as f64 / (avg_lu + avg_solve) as f64) * 100.0);
}
