use eigen_rs::core::matrix::MatrixX;
use std::time::Instant;
use std::hint::black_box;

fn main() {
    let size = 512;
    println!("Benchmarking LDLT Optimization (N={})", size);

    let mut a = MatrixX::<f64>::new_dynamic(size, size).unwrap();
    // Fill random
    for i in 0..size * size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 17) as f64;
    }

    // Make symmetric M = A * A^T + I * size
    let mut sym = MatrixX::<f64>::new_dynamic(size, size).unwrap();
    sym.assign_product(&(&a * &a.transpose())).unwrap();
    for i in 0..size {
        *sym.get_mut(i, i).unwrap() += size as f64;
    }

    // Warmup
    for _ in 0..5 {
        black_box(sym.clone().ldlt().unwrap());
    }

    let start = Instant::now();
    let iterations = 20;
    for _ in 0..iterations {
        // We clone inside the loop to ensure we restart with fresh data 
        // (LDLT is in-place usually? Wait, LDLT::new takes reference but creates new matrix internally)
        // LDLT::new(matrix: &Matrix) -> Result<Self>
        // Internally it copies into a new matrix. So we can reuse `sym`.
        black_box(sym.ldlt().unwrap());
    }
    let duration = start.elapsed();
    println!("LDLT Time: {:.2} ms per iter", duration.as_secs_f64() * 1000.0 / iterations as f64);
}
