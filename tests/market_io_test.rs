use eigen_rs::core::sparse::{SparseMatrix, Triplet, StorageOrder};
use eigen_rs::unsupported::market_io::{load_matrix_market, save_matrix_market};
use eigen_rs::core::sparse::iterators::InnerIterator;
use std::fs;

#[test]
fn test_matrix_market_roundtrip() {
    let path = "test_matrix.mtx";

    // Create a sparse matrix
    let mut mat = SparseMatrix::<f64>::new(3, 3, StorageOrder::ColMajor);
    let triplets = vec![
        Triplet::new(0, 0, 1.0),
        Triplet::new(1, 1, 2.0),
        Triplet::new(2, 2, 3.0),
        Triplet::new(0, 2, 0.5), // Off-diagonal
    ];
    mat.set_from_triplets(triplets);

    // Save
    save_matrix_market(path, &mat).expect("Failed to save mtx");

    // Load
    let loaded = load_matrix_market::<_, f64>(path).expect("Failed to load mtx");

    // Verify
    assert_eq!(loaded.rows(), 3);
    assert_eq!(loaded.cols(), 3);

    let check = |m: &SparseMatrix<f64>, r: usize, c: usize, expect: f64| {
         let mut found = false;
         for k in 0..m.outer_size() {
             let mut it = InnerIterator::new(m, k);
             while it.is_valid() {
                 let (curr_r, curr_c) = (it.row(), it.col());
                 let val = it.value();
                 if curr_r == r && curr_c == c {
                     assert!((val - expect).abs() < 1e-9, "Mismatch at ({},{}): got {}, want {}", r, c, val, expect);
                     found = true;
                 }
                 it.next();
             }
         }
         if expect != 0.0 && !found {
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
    // (1,1) -> (0,0) = 1.0
    // (2,1) -> (1,0) = 0.5 -> implies (0,1) = 0.5 too
    fs::write(path, content).unwrap();

    let loaded = load_matrix_market::<_, f64>(path).unwrap();

    let check = |m: &SparseMatrix<f64>, r: usize, c: usize, expect: f64| {
        let mut seen = false;
        for k in 0..m.outer_size() {
            let mut it = InnerIterator::new(m, k);
            while it.is_valid() {
                 let (curr_r, curr_c) = (it.row(), it.col());
                 let val = it.value();
                 if curr_r == r && curr_c == c {
                     assert!((val - expect).abs() < 1e-9);
                     seen = true;
                 }
                 it.next();
            }
        }
        if expect != 0.0 && !seen {
             panic!("Element ({},{}) not found, expected {}", r, c, expect);
        }
    };

    // 1-based "1 1" -> 0,0
    check(&loaded, 0, 0, 1.0);
    // 1-based "2 1" -> 1,0
    check(&loaded, 1, 0, 0.5);
    // Symmetric part: 0,1
    check(&loaded, 0, 1, 0.5);

    let _ = fs::remove_file(path);
}
