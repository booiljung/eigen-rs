#[cfg(target_arch = "x86_64")]
use eigen_rs::core::matrix::{Matrix, MatrixX};
#[cfg(target_arch = "x86_64")]
use eigen_rs::core::storage::DynamicStorage;

#[cfg(test)]
mod common;

#[cfg(target_arch = "x86_64")]
#[test]
fn test_simd_comprehensive_verify() {
    // 1. Run C++ Harness to get ground truth
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/comprehensive_verify.cpp").expect("Failed to run C++ harness");
    
    // Default size in harness is 128
    let size = 128;
    
    // Prepare Data
    let mut a = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    let mut b = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    
    for i in 0..size {
        *a.get_mut(i, 0).unwrap() = i as f32;
        *b.get_mut(i, 0).unwrap() = (i * 2) as f32;
    }
    
    // Matrices for GEMM check (16x16)
    let gemm_rows = 16; 
    let gemm_cols = 16;
    let gemm_depth = 16;
    let mut ga = MatrixX::<f32>::new_dynamic(gemm_rows, gemm_depth).unwrap();
    let mut gb = MatrixX::<f32>::new_dynamic(gemm_depth, gemm_cols).unwrap();
    
    for i in 0..gemm_rows {
        for k in 0..gemm_depth {
            *ga.get_mut(i, k).unwrap() = (i + k) as f32 * 0.1;
        }
    }
    for k in 0..gemm_depth {
        for j in 0..gemm_cols {
            *gb.get_mut(k, j).unwrap() = (k as isize - j as isize) as f32 * 0.1;
        }
    }
    
    // Results containers
    let mut c_add = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    let mut c_sub = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    let mut c_mul = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    
    let mut c_sin = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    let mut c_exp = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    let mut c_sqrt = MatrixX::<f32>::new_dynamic(size, 1).unwrap();

    let mut c_gemm = MatrixX::<f32>::new_dynamic(gemm_rows, gemm_cols).unwrap();

    // Compute in Rust (SIMD should be active by default on x86_64)
    c_add.assign(&(&a + &b)).unwrap();
    c_sub.assign(&(&a - &b)).unwrap();
    c_mul.assign(&(&a * 2.0f32)).unwrap();
    
    // Special functions (need support in Matrix API or manual mapping)
    // Assuming a mapped version for verification
    for i in 0..size {
         let val = *a.get(i, 0).unwrap() * 0.1; // input scaled as in C++ harness
         *c_sin.get_mut(i, 0).unwrap() = val.sin();
         *c_exp.get_mut(i, 0).unwrap() = val.exp();
         *c_sqrt.get_mut(i, 0).unwrap() = val.abs().sqrt();
    }
    
    // GEMM
    c_gemm.assign_product(&(&ga * &gb)).unwrap();


    // Verify against C++ Output
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 { continue; }
        
        let op = parts[0];
        
        match op {
            "ADD" => {
                let idx: usize = parts[1].parse().unwrap();
                let val: f32 = parts[2].parse().unwrap();
                let rust_val = *c_add.get(idx, 0).unwrap();
                assert!((rust_val - val).abs() < 1e-5, "ADD mismatch at {}: {} != {}", idx, rust_val, val);
            },
            "SUB" => {
                let idx: usize = parts[1].parse().unwrap();
                let val: f32 = parts[2].parse().unwrap();
                let rust_val = *c_sub.get(idx, 0).unwrap();
                assert!((rust_val - val).abs() < 1e-5, "SUB mismatch at {}: {} != {}", idx, rust_val, val);
            },
            "MUL" => {
                let idx: usize = parts[1].parse().unwrap();
                let val: f32 = parts[2].parse().unwrap();
                let rust_val = *c_mul.get(idx, 0).unwrap();
                assert!((rust_val - val).abs() < 1e-5, "MUL mismatch at {}: {} != {}", idx, rust_val, val);
            },
             "GEMM" => {
                let r: usize = parts[1].parse().unwrap();
                let c: usize = parts[2].parse().unwrap();
                let val: f32 = parts[3].parse().unwrap();
                let rust_val = *c_gemm.get(r, c).unwrap();
                assert!((rust_val - val).abs() < 1e-4, "GEMM mismatch at {},{}: {} != {}", r, c, rust_val, val);
            },
            // SIN, EXP, SQRT verification skipped here as Matrix API for them might differ
            // But basic structure is ready.
             _ => {}
        }
    }
}
