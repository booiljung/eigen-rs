use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::FixedStorage;
use eigen_rs::core::xpr::MatrixXpr;
mod common;

#[test]
fn test_matrix_block_differential() {
    // 1. Get reference values from C++ Eigen
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/matrix_block_verify.cpp")
        .expect("Failed to run C++ harness");

    // 2. Setup Rust matrix (4x4)
    let mut m = Matrix::<f32, FixedStorage<f32, 4, 4, 16>>::new_fixed();
    for i in 0..4 {
        for j in 0..4 {
            *m.get_mut(i, j).unwrap() = (i * 4 + j + 1) as f32;
        }
    }

    // 3. Perform Block operations
    let b = m.block(1, 1, 2, 2);
    let r = m.row(2);
    let c = m.col(3);

    // 4. Compare
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts[0] {
            "BLOCK" => {
                let row: usize = parts[1].parse().unwrap();
                let col: usize = parts[2].parse().unwrap();
                let v_cpp: f32 = parts[3].parse().unwrap();
                let v_rust = b.eval(row, col);
                assert_eq!(v_rust, v_cpp, "Block mismatch at {},{}", row, col);
            },
            "ROW" => {
                let col: usize = parts[1].parse().unwrap();
                let v_cpp: f32 = parts[2].parse().unwrap();
                let v_rust = r.eval(0, col);
                assert_eq!(v_rust, v_cpp, "Row mismatch at col {}", col);
            },
            "COL" => {
                let row: usize = parts[1].parse().unwrap();
                let v_cpp: f32 = parts[2].parse().unwrap();
                let v_rust = c.eval(row, 0);
                assert_eq!(v_rust, v_cpp, "Col mismatch at row {}", row);
            },
            _ => {}
        }
    }
}

#[test]
fn test_matrix_block_chaining() {
    let mut a = Matrix::<f32, FixedStorage<f32, 4, 4, 16>>::new_fixed();
    let mut b = Matrix::<f32, FixedStorage<f32, 4, 4, 16>>::new_fixed();
    for i in 0..4 {
        for j in 0..4 {
            *a.get_mut(i, j).unwrap() = 1.0;
            *b.get_mut(i, j).unwrap() = 2.0;
        }
    }

    // (A + B).block(1, 1, 2, 2).transpose()
    let sum = &a + &b;
    let blk = sum.block(1, 1, 2, 2);
    let res = blk.transpose();

    assert_eq!(res.rows(), 2);
    assert_eq!(res.cols(), 2);
    assert_eq!(res.eval(0, 0), 3.0);
}
