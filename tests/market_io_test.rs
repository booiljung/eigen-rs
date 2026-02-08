use eigen_rs::core::sparse::SparseMatrix;
use eigen_rs::unsupported::market_io::{load_matrix_market, save_matrix_market};
use std::fs;

#[test]
fn test_matrix_market_roundtrip() {
    let path = "test_matrix.mtx";
    
    // Create a sparse matrix
    let mut mat = SparseMatrix::<f64>::new(3, 3);
    mat.insert(0, 0, 1.0);
    mat.insert(1, 1, 2.0);
    mat.insert(2, 2, 3.0);
    mat.insert(0, 2, 0.5); // Off-diagonal
    
    // Save
    save_matrix_market(path, &mat).expect("Failed to save mtx");
    
    // Load
    let loaded = load_matrix_market::<_, f64>(path).expect("Failed to load mtx");

    // Verify
    assert_eq!(loaded.rows(), 3);
    assert_eq!(loaded.cols(), 3);
    
    // Helper to check coeff
    let check = |m: &SparseMatrix<f64>, r, c, expect: f64| {
        // Simple search
        let mut found = 0.0;
        // Search CSR/CSC for (r,c)
        // Since we don't have random access O(1), iterate row/col?
        // Let's reuse standard getter if available, or just iterate.
        // Or better, convert both to dense and compare?
        // Or rely on  if implemented.
        // Assuming no  on SparseMatrix yet in public API for test?
        // Actually we can just iterate loaded and verify contents.
        for k in 0..m.outer_size() {
            for (curr_r, curr_c, val) in m.inner_iterator(k) {
                if curr_r == r && curr_c == c {
                     // Check approx
                    assert!((val - expect).abs() < 1e-9, "Mismatch at ({},{}): got {}, want {}", r,c, val, expect);
                    return;
                }
            }
        }
        if expect != 0.0 {
            panic!("Element ({},{}) not found, expected {}", r, c, expect);
        }
    };

    check(&loaded, 0, 0, 1.0);
    check(&loaded, 1, 1, 2.0);
    check(&loaded, 2, 2, 3.0);
    check(&loaded, 0, 2, 0.5);
    
    // Cleanup
    let _ = fs::remove_file(path);
}

#[test]
fn test_matrix_market_symmetric_load() {
    // Create a manual symmetric file
    let path = "test_sym.mtx";
    let content = r#"%%MatrixMarket matrix coordinate real symmetric
3 3 2
1 1 1.0
2 1 0.5
"#;
    // (1,1) = 1.0
    // (2,1) = 0.5 -> implies (1,2) = 0.5 too
    fs::write(path, content).unwrap();

    let loaded = load_matrix_market::<_, f64>(path).unwrap();

    let check = |m: &SparseMatrix<f64>, r, c, expect: f64| {
         let mut seen = false;
         for k in 0..m.outer_size() {
            for (curr_r, curr_c, val) in m.inner_iterator(k) {
                if curr_r == r && curr_c == c {
                    assert!((val - expect).abs() < 1e-9);
                    seen = true;
                }
            }
        }
        if expect != 0.0 && !seen { panic!("Missing nonzero ({},{})", r, c); }
    };
    
    // 1-based "1 1" -> 0,0
    check(&loaded, 0, 0, 1.0);
    // 1-based "2 1" -> 1,0
    check(&loaded, 1, 0, 0.5);
    // Symmetric part: 0,1
    check(&loaded, 0, 1, 0.5);

    let _ = fs::remove_file(path);
}
