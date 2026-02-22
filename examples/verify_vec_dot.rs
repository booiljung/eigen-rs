use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;
use eigen_rs::core::storage::Storage;
use std::time::Instant;

fn init_vector(v: &mut Matrix<f32, DynamicStorage<f32>>) {
    for i in 0..v.size() {
        *v.get_mut(i, 0).unwrap() = (i % 17) as f32 / 10.0;
    }
}

fn main() {
    let sizes = vec![16, 64, 256, 1024, 4096, 16384, 65536];

    println!("One-off VecDot Benchmark (ns/iter)");
    println!("Size | Rust (ns)");
    println!("---|---");

    for size in sizes {
        let mut v1 = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(size, 1).unwrap();
        init_vector(&mut v1);
        let mut v2 = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(size, 1).unwrap();
        init_vector(&mut v2);

        if size == 65536 {
            println!(
                "DEBUG: v1 alignment: {}",
                v1.storage().data().as_ptr() as usize % 32
            );
            println!(
                "DEBUG: v2 alignment: {}",
                v2.storage().data().as_ptr() as usize % 32
            );
        }

        let mut res = 0.0;
        let iter = if size < 1000 { 100_000 } else { 10_000 };

        // Warmup
        for _ in 0..100 {
            res += v1.dot(&v2);
        }

        let start = Instant::now();
        for _ in 0..iter {
            let val = v1.dot(&v2);
            res += val;
            // Add side effect to prevent total elision if smart
            if size > 0 {
                unsafe {
                    *v1.get_mut(0, 0).unwrap() += 1e-9;
                }
            }
        }
        let duration = start.elapsed();

        println!("{} | {}", size, duration.as_nanos() / iter as u128);
        std::hint::black_box(res);
    }
}
