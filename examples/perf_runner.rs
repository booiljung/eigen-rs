use std::time::Instant;
use eigen_rs::core::matrix::MatrixX;
use std::hint::black_box;
use eigen_rs::core::xpr::MatrixXpr;



fn init_matrix(m: &mut MatrixX<f32>) {
    let rows = m.rows();
    for i in 0..m.size() {
        let r = i % rows;
        let c = i / rows;
        *m.get_mut(r, c).unwrap() = ((r + c * rows) % 17) as f32 / 10.0;
    }
}

fn init_vector(v: &mut MatrixX<f32>) {
    for i in 0..v.size() {
        *v.get_mut(i, 0).unwrap() = (i % 17) as f32 / 10.0;
    }
}

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

fn bench_vector_ops(size: usize) {
    let mut v1 = MatrixX::<f32>::new_dynamic(size, 1).unwrap(); init_vector(&mut v1);
    let mut v2 = MatrixX::<f32>::new_dynamic(size, 1).unwrap(); init_vector(&mut v2);
    // v2 += 1.0
    for i in 0..size { *v2.get_mut(i, 0).unwrap() += 1.0; }

    let iterations = if size < 1000 { 10000 } else { 1000 };
    
    // Dot
    let start = Instant::now();
    let mut dot_res = 0.0;
    for _ in 0..iterations {
        dot_res += black_box(&v1).dot(black_box(&v2));
        // Manual dependency removed in favor of black_box, but to match C++ checksums we might need it?
        // C++ had v1(0) += ... 
        // If I don't modify v1, checksum differs.
        // I WILL add modification back to match C++ logic/checksum.
        *v1.get_mut(0,0).unwrap() += 0.00001; 
    }
    let duration = start.elapsed().as_nanos();
    println!("VecDot,{},{},{}", size, duration / iterations as u128, dot_res / iterations as f32);
    
    // Norm
    let start = Instant::now();
    let mut norm_res = 0.0;
    for _ in 0..iterations {
        norm_res += black_box(&v1).norm();
        *v1.get_mut(0,0).unwrap() += 0.00001; 
    }
    let duration = start.elapsed().as_nanos();
    println!("VecNorm,{},{},{}", size, duration / iterations as u128, norm_res / iterations as f32);
}

fn bench_matrix_arithmetic(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap(); init_matrix(&mut a);
    let mut b = MatrixX::<f32>::new_dynamic(size, size).unwrap(); init_matrix(&mut b);
    let mut c = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    
    let iterations = if size < 128 { 1000 } else { 100 };

    // Add
    let start = Instant::now();
    for _ in 0..iterations {
        c.assign(&(black_box(&a) + black_box(&b))).unwrap(); 
        // C++ Bench did NOT modify 'a' in final version (lines 128+ in eigen_bench.cpp).
        // C++: c.noalias() = a + b;
        // So no modification needed here.
    }
    let duration = start.elapsed().as_nanos();
    println!("MatAdd,{},{},{}", size, duration / iterations as u128, c.sum());
    
    // Scale
    init_matrix(&mut a); // Reset
    let start = Instant::now();
    for _ in 0..iterations {
        a.scale(1.01);
    }
    let duration = start.elapsed().as_nanos();
    println!("MatScale,{},{},{}", size, duration / iterations as u128, a.sum());
}



fn bench_geometry() {
    use eigen_rs::geometry::Quaternion;
    use eigen_rs::core::matrix::{Matrix, FixedStorage};
    use eigen_rs::core::xpr::MatrixXpr; 
    
    let iterations = 10_000_000;
    
    // Cross Product (3D)
    let mut v1 = Matrix::<f32, FixedStorage<f32, 3, 1, 3>>::from_array([1.0, 2.0, 3.0]);
    let v2 = Matrix::<f32, FixedStorage<f32, 3, 1, 3>>::from_array([4.0, 5.0, 6.0]);
    
    let start = Instant::now();
    for _ in 0..iterations {
        let v3 = black_box(&v1).cross(black_box(&v2));
        *v1.get_mut(0,0).unwrap() += v3.get(0,0).unwrap() * 0.001; 
    }
    let duration = start.elapsed().as_nanos();
    println!("Cross3D,3,{},{}", duration / iterations as u128, v1.sum());
    
    // Quaternion Mul
    let mut q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let q2 = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    
    let start = Instant::now();
    for _ in 0..iterations {
        q1 = black_box(&q1) * black_box(&q2);
    }
    let duration = start.elapsed().as_nanos();
    let q_sum = q1.w() + q1.x() + q1.y() + q1.z();
    println!("QuatMul,4,{},{}", duration / iterations as u128, q_sum);
}

fn bench_dense_decomp_extra(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    // fill
     for i in 0..size*size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 17) as f32;
    }
    
    let iterations = if size < 64 { 20 } else { 5 };
    
    // LU
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.partial_piv_lu();
    }
    let duration = start.elapsed().as_nanos();
    println!("LU,{},{}", size, duration / iterations as u128);
    
    // QR
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.householder_qr();
    }
    let duration = start.elapsed().as_nanos();
    println!("QR,{},{}", size, duration / iterations as u128);
}

fn bench_sparse(size: usize) {
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::xpr::MatrixXpr; // Import for sum()
    
    let mut sp = SparseMatrix::new(size, size, StorageOrder::RowMajor);
    let density = 0.05;
    let nnz = (size as f32 * size as f32 * density) as usize;
    
    let mut triplets = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let r = (k * 17) % size;
        let c = (k * 29) % size;
        let v = (k % 100) as f32 / 10.0;
        triplets.push(Triplet::new(r, c, v));
    }
    sp.set_from_triplets(triplets);
    
    let mut v = MatrixX::<f32>::new_dynamic(size, 1).unwrap(); init_vector(&mut v);
    
    let iterations = if size < 256 { 1000 } else { 100 };
    
    // SpMV
    let mut res_sum = 0.0;
    let start = Instant::now();
    for _ in 0..iterations {
        let res = (&sp * &v).unwrap();
        res_sum = res.sum();
        *v.get_mut(0,0).unwrap() += 0.00001; // Prevent hoisting
    }
    let duration = start.elapsed().as_nanos();
    println!("SpMV,{},{},{}", size, duration / iterations as u128, res_sum);
   
    // SpMM
    let mut m = MatrixX::<f32>::new_dynamic(size, 32).unwrap();
    for i in 0..size {
        for j in 0..32 {
             *m.get_mut(i, j).unwrap() = ((i + j) % 11) as f32;
        }
    }
    
    let start = Instant::now();
    for _ in 0..iterations {
        let res = (&sp * &m).unwrap();
        res_sum = res.sum();
        *m.get_mut(0,0).unwrap() += 0.00001; // Prevent hoisting
    }
    let duration = start.elapsed().as_nanos();
    println!("SpMM_Dense,{},{},{}", size, duration / iterations as u128, res_sum);
}

fn main() {
    println!("Running Performance Test (Sweep 16..384 + Large Vecs)");
    let args: Vec<String> = std::env::args().collect();
    
    // 1. Standard Sweep: 16 to 384, stride 13
    let mut sizes: Vec<usize> = (16..=384).step_by(13).collect();
    sizes.extend_from_slice(&[64, 128, 256]);
    sizes.sort();
    sizes.dedup();

    // 2. Small Sweep: 16 to 64, stride 7
    let mut small_sizes: Vec<usize> = (16..=64).step_by(7).collect();
    small_sizes.extend_from_slice(&[32, 64]);
    small_sizes.sort();
    small_sizes.dedup();

    for &size in &sizes {
        bench_matmul(size);
        bench_llt(size);
        bench_dense_decomp_extra(size);
        bench_matrix_arithmetic(size);
        bench_vector_ops(size); // Test unrolling boundaries
        bench_sparse(size);
    }
    
    // 3. Large Vector Benchmarks (for Prefetching/Throughput)
    let large_vec_sizes = [4096, 16384, 65536];
    for &size in &large_vec_sizes {
        bench_vector_ops(size);
    }

    // 4. Heavy Ops (Small only)
    for &size in &small_sizes {
        bench_svd(size);
        bench_eigenvalues(size);
    }
    
    bench_geometry();
}
