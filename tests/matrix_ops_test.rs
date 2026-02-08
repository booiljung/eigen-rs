use eigen_rs::Matrix2;
mod common;

#[test]
fn test_matrix_ops_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_ops_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup same matrices in Rust
    let mut a = Matrix2::<f32>::new_fixed();
    let mut b = Matrix2::<f32>::new_fixed();

    *a.get_mut(0, 0).unwrap() = 10.0; *a.get_mut(0, 1).unwrap() = 20.0;
    *a.get_mut(1, 0).unwrap() = 30.0; *a.get_mut(1, 1).unwrap() = 40.0;

    *b.get_mut(0, 0).unwrap() = 1.0; *b.get_mut(0, 1).unwrap() = 2.0;
    *b.get_mut(1, 0).unwrap() = 3.0; *b.get_mut(1, 1).unwrap() = 4.0;

    // Subtraction
    let mut res_sub = Matrix2::<f32>::new_fixed();
    res_sub.assign(&(&a - &b)).unwrap();

    // Scalar Multiplication
    let mut res_mul = Matrix2::<f32>::new_fixed();
    res_mul.assign(&(&a * 2.5)).unwrap();

    // 3. Compare
    let mut current_section = "";
    for line in cpp_output.lines() {
        if line == "SUB" { current_section = "SUB"; continue; }
        if line == "MUL" { current_section = "MUL"; continue; }
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 3 {
            let r: usize = parts[0].trim().parse().unwrap_or_else(|_| panic!("Failed to parse row from line: '{}'", line));
            let c: usize = parts[1].trim().parse().unwrap_or_else(|_| panic!("Failed to parse col from line: '{}'", line));
            let v_cpp: f32 = parts[2].trim().parse().unwrap_or_else(|_| panic!("Failed to parse val from line: '{}'", line));
            
            if current_section == "SUB" {
                let v_rust = *res_sub.get(r, c).unwrap();
                assert!((v_rust - v_cpp).abs() < 1e-6, "SUB mismatch at {},{}", r, c);
            } else if current_section == "MUL" {
                let v_rust = *res_mul.get(r, c).unwrap();
                assert!((v_rust - v_cpp).abs() < 1e-6, "MUL mismatch at {},{}", r, c);
            }
        }
    }
}
