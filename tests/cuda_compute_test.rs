#[cfg(feature = "cuda")]
use eigen_rs::core::matrix::{Matrix, MatrixX};
mod common;
#[cfg(feature = "cuda")]
use eigen_rs::core::storage::cuda::CudaStorage;

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_comprehensive_verify() {
    // 1. Run C++ Harness to get ground truth
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/comprehensive_verify.cpp").expect("Failed to run C++ harness");
    
    // Default size in harness is 128
    let size = 128;
    
    // Prepare Host Data matching C++ harness
    let mut h_a = vec![0.0f32; size];
    let mut h_b = vec![0.0f32; size];
    
    for i in 0..size {
        h_a[i] = i as f32;
        h_b[i] = (i * 2) as f32;
    }
    
    // Setup CUDA matrices
    let mut d_a = Matrix::<f32, CudaStorage<f32>>::new_dynamic(size, 1).unwrap();
    let mut d_b = Matrix::<f32, CudaStorage<f32>>::new_dynamic(size, 1).unwrap();
    
    d_a.storage_mut().copy_from_host(&h_a).unwrap();
    d_b.storage_mut().copy_from_host(&h_b).unwrap();
    
    // Results
    let mut d_res = Matrix::<f32, CudaStorage<f32>>::new_dynamic(size, 1).unwrap();
    let mut h_res = vec![0.0f32; size];

    // Compute ADD
    d_res.assign(&(&d_a + &d_b)).unwrap();
    d_res.storage().copy_to_host(&mut h_res).unwrap();
    
    // Store results for ADD to verify later
    let res_add = h_res.clone();
    
    // Compute SUB
    d_res.assign(&(&d_a - &d_b)).unwrap();
    d_res.storage().copy_to_host(&mut h_res).unwrap();
    let res_sub = h_res.clone();

    // Compute MUL (Scalar)
    d_res.assign(&(&d_a * 2.0f32)).unwrap();
    d_res.storage().copy_to_host(&mut h_res).unwrap();
    let res_mul = h_res.clone();

    // Verify against C++ Output
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 { continue; }
        
        let op = parts[0];
        
        match op {
            "ADD" => {
                let idx: usize = parts[1].parse().unwrap();
                let val: f32 = parts[2].parse().unwrap();
                assert!((res_add[idx] - val).abs() < 1e-5, "ADD mismatch at {}", idx);
            },
            "SUB" => {
                let idx: usize = parts[1].parse().unwrap();
                let val: f32 = parts[2].parse().unwrap();
                assert!((res_sub[idx] - val).abs() < 1e-5, "SUB mismatch at {}", idx);
            },
            "MUL" => {
                let idx: usize = parts[1].parse().unwrap();
                let val: f32 = parts[2].parse().unwrap();
                assert!((res_mul[idx] - val).abs() < 1e-5, "MUL mismatch at {}", idx);
            },
             _ => {}
        }
    }
}


#[cfg(feature = "cuda")]
#[test]
fn test_cublas_gemm() {
    use eigen_rs::core::storage::cublas::{CublasHandle, gemm_cublas};

    // 1. Setup
    let size = 128; // 128x128 matrices

    let mut h_a = vec![0.0f32; size * size];
    let mut h_b = vec![0.0f32; size * size];
    
    // Identity * Identity = Identity
    for i in 0..size {
         h_a[i * size + i] = 1.0;
         h_b[i * size + i] = 1.0;
    }

    // CUDA Storage
    let mut d_a = Matrix::<f32, CudaStorage<f32>>::new_dynamic(size, size).unwrap();
    let mut d_b = Matrix::<f32, CudaStorage<f32>>::new_dynamic(size, size).unwrap();
    let mut d_c = Matrix::<f32, CudaStorage<f32>>::new_dynamic(size, size).unwrap();

    d_a.storage_mut().copy_from_host(&h_a).unwrap();
    d_b.storage_mut().copy_from_host(&h_b).unwrap();
    
    // 2. Compute C = 1.0 * A * B + 0.0 * C using cuBLAS
    let handle = CublasHandle::new().unwrap();
    
    // Note: Rust eigen-rs is ColMajor. cuBLAS is ColMajor.
    // Dimensions: A(size x size), B(size x size), C(size x size)
    // No transpose needed.
    gemm_cublas(
        &handle, 
        false, false, 
        size, size, size, 
        1.0f32, 
        d_a.storage(), 
        d_b.storage(), 
        0.0f32, 
        d_c.storage_mut()
    ).unwrap();

    // 3. Verify
    let mut h_c = vec![0.0f32; size * size];
    d_c.storage().copy_to_host(&mut h_c).unwrap();

    for i in 0..size {
        for j in 0..size {
            let val = h_c[j * size + i]; // storage is col-major: col * rows + row
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((val - expected).abs() < 1e-5, "Mismatch at {},{}: {} != {}", i, j, val, expected);
        }
    }
}
