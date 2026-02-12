use std::time::Instant;
use eigen_rs::core::matrix::MatrixX;

fn bench_matmul(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    let mut b = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    let mut c = MatrixX::<f32>::new_dynamic(size, size).unwrap();

    // Randomize
    for i in 0..size*size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 100) as f32;
        *b.get_mut(i % size, i / size).unwrap() = ((i + 1) % 100) as f32;
    }

    let iterations = if size < 128 { 100 } else { 20 };

    // Warm up
    for _ in 0..5 {
        c.assign_product(&(&a * &b)).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        c.assign_product(&(&a * &b)).unwrap();
    }
    let duration = start.elapsed().as_nanos();
    
    println!("MatMul,{},{}", size, duration / iterations as u128);
}

fn bench_llt(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    // Make SPD
    let mut temp = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size*size {
        *temp.get_mut(i % size, i / size).unwrap() = (i % 100) as f32;
    }
    // M = A * A.T + I * size
    let mut m = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    m.assign_product(&(&temp * &temp.transpose())).unwrap();
    for i in 0..size {
        *m.get_mut(i, i).unwrap() += size as f32;
    }

    let iterations = if size < 128 { 100 } else { 20 };

    // Warm up
    for _ in 0..5 {
        let _ = m.llt();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = m.llt();
    }
    let duration = start.elapsed().as_nanos();
    
    println!("LLT,{},{}", size, duration / iterations as u128);
}

fn bench_svd(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size*size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 100) as f32;
    }

    let iterations = if size < 64 { 20 } else { 5 };

    // Warm up
    for _ in 0..2 {
        let _ = a.jacobi_svd();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.jacobi_svd();
    }
    let duration = start.elapsed().as_nanos();
    
    println!("SVD,{},{}", size, duration / iterations as u128);
}

fn bench_eigenvalues(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size*size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 100) as f32;
    }
    // Make symmetric M = A + A.T
    let mut m = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size {
        for j in 0..size {
            let val = a.get(i, j).unwrap() + a.get(j, i).unwrap();
            *m.get_mut(i, j).unwrap() = val;
        }
    }

    let iterations = if size < 64 { 20 } else { 5 };

    // Warm up
    for _ in 0..2 {
        let _ = m.self_adjoint_eigen_solver(true); // compute vectors too? C++ bench implies yes
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = m.self_adjoint_eigen_solver(true);
    }
    let duration = start.elapsed().as_nanos();
    
    println!("EigenValues,{},{}", size, duration / iterations as u128);
}

fn main() {
    let sizes = [64, 128, 256];
    let small_sizes = [32, 64];

    for size in sizes {
        bench_matmul(size);
    }
    for size in sizes {
        bench_llt(size);
    }
    for size in small_sizes {
        bench_svd(size);
    }
    for size in small_sizes {
        bench_eigenvalues(size);
    }
}
