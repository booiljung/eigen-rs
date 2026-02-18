extern crate eigen_rs;
use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::decompositions::llt::LLT;
use std::time::Instant;

fn bench_size(n: usize) {
    println!("\n=== Size N={} ===", n);
    let mut a = MatrixX::<f32>::new_dynamic(n, n).unwrap();
    let mut b = MatrixX::<f32>::new_dynamic(n, n).unwrap();
    let mut c = MatrixX::<f32>::new_dynamic(n, n).unwrap();

    // Initialize
    for i in 0..n {
        for j in 0..n {
            *a.get_mut(i, j).unwrap() = (i + j) as f32 * 0.01;
            *b.get_mut(i, j).unwrap() = (i as f32 - j as f32) * 0.01;
        }
    }
    // Make 'a' positive definite for LLT
    for i in 0..n {
        *a.get_mut(i, i).unwrap() += n as f32;
    }

    // Warmup
    c.assign(&(&a + &b)).unwrap();

    // Bench MatAdd
    let start = Instant::now();
    let iters = 5000;
    for _ in 0..iters {
        c.assign(&(&a + &b)).unwrap();
    }
    let duration = start.elapsed();
    println!("MatAdd: {:.2} ns/iter", duration.as_nanos() as f64 / iters as f64);

    // Bench MatMul
    // Reduce iters for MM
    let mm_iters = 100;
    let start = Instant::now();
    for _ in 0..mm_iters {
        c.assign(&(&a * &b)).unwrap();
    }
    let duration = start.elapsed();
    println!("MatMul: {:.2} us/iter", duration.as_micros() as f64 / mm_iters as f64);

    // Bench LLT
    let start = Instant::now();
    for _ in 0..1000 {
        let _llt = LLT::new(&a).ok();
    }
    let duration = start.elapsed();
    println!("LLT:    {:.2} us/iter", duration.as_micros() as f64 / 1000.0);
}

fn main() {
    let sizes = vec![288, 289, 290, 314, 315, 316, 327, 328, 329];
    for size in sizes {
        bench_size(size);
    }
}
