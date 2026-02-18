use eigen_rs::core::tensor::Tensor;
use eigen_rs::core::tensor::device::CpuDevice;
use std::time::Instant;

fn naive_contract(lhs: &Tensor<f32, 2>, rhs: &Tensor<f32, 2>, lhs_dim: usize, rhs_dim: usize) -> Tensor<f32, 2> {
    let l_dims = lhs.dims();
    let r_dims = rhs.dims();
    
    // Validate
    assert_eq!(l_dims[lhs_dim], r_dims[rhs_dim]);
    
    let k_size = l_dims[lhs_dim];
    
    // Result dims: LHS free dims + RHS free dims
    // For Rank 2 contracting 1 dim, result is Rank 2.
    // LHS free dim = 1 - lhs_dim.
    // RHS free dim = 1 - rhs_dim.
    let l_free = 1 - lhs_dim;
    let r_free = 1 - rhs_dim;
    
    let m = l_dims[l_free];
    let n = r_dims[r_free];
    
    let mut res = Tensor::<f32, 2>::new([m, n]).unwrap();
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k_size {
                let mut l_idx = [0, 0];
                l_idx[l_free] = i;
                l_idx[lhs_dim] = k;
                
                let mut r_idx = [0, 0];
                r_idx[r_free] = j;
                r_idx[rhs_dim] = k;
                
                sum += lhs.get(l_idx).unwrap() * rhs.get(r_idx).unwrap();
            }
            *res.get_mut([i, j]).unwrap() = sum;
        }
    }
    res
}

fn main() {
    println!("Benchmarking Tensor Contraction (GEMM vs Naive)");
    
    // Parameters
    // A: (M, K), B: (K, N) -> C: (M, N)
    // Contract dim 1 of A with dim 0 of B.
    // Use square matrices for simplicity
    let size = 512; // 512x512 matrices
    let iterations = 10;
    
    println!("Matrix Size: {}x{}", size, size);
    
    // Setup
    let mut t1 = Tensor::<f32, 2>::new([size, size]).unwrap();
    let mut t2 = Tensor::<f32, 2>::new([size, size]).unwrap();
    
    // Fill with data
    let data_size = size * size;
    for i in 0..data_size {
        t1.data_mut()[i] = ((i % 10) as f32) * 0.1;
        t2.data_mut()[i] = (((i + 1) % 10) as f32) * 0.1;
    }
    
    // 1. Warmup and Validation
    println!("Validating...");
    let res_gemm: Tensor<f32, 2> = t1.contract(&t2, 1, 0);
    // Naive
    let res_naive = naive_contract(&t1, &t2, 1, 0);
    
    // Verify
    let mut max_diff = 0.0;
    for i in 0..res_gemm.size() {
        let d = (res_gemm.data()[i] - res_naive.data()[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    if max_diff > 1e-2 {
        println!("WARNING: Validation Failed! Max diff: {}", max_diff);
    } else {
        println!("Validation Passed. Max diff: {}", max_diff);
    }
    
    // 2. Measure Naive
    println!("Measuring Naive...");
    let start_naive = Instant::now();
    for _ in 0..iterations {
        let _ = naive_contract(&t1, &t2, 1, 0);
    }
    let duration_naive = start_naive.elapsed();
    println!("Naive Time ({:?} iters): {:?}", iterations, duration_naive);
    
    // 3. Measure GEMM
    println!("Measuring GEMM...");
    let start_gemm = Instant::now();
    for _ in 0..iterations {
        let _: Tensor<f32, 2> = t1.contract(&t2, 1, 0);
    }
    let duration_gemm = start_gemm.elapsed();
    println!("GEMM Time  ({:?} iters): {:?}", iterations, duration_gemm);
    
    let speedup = duration_naive.as_secs_f64() / duration_gemm.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);
}
