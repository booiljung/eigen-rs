use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder, Triplet};
use eigen_rs::core::xpr::MatrixXpr;

#[test]
fn test_parallel_assign() {
    let size = 200;
    let mut m1 = MatrixX::<f64>::new_dynamic(size, size).unwrap();
    let m2 = MatrixX::<f64>::new_dynamic(size, size).unwrap();
    let m3 = MatrixX::<f64>::new_dynamic(size, size).unwrap();

    // Fill with values
    let mut m2 = m2;
    let mut m3 = m3;
    for i in 0..size {
        for j in 0..size {
            *m2.get_mut(i, j).unwrap() = (i + j) as f64;
            *m3.get_mut(i, j).unwrap() = (i * j) as f64;
        }
    }

    // Parallel assign: m1 = m2 + m3
    m1.assign(&(&m2 + &m3)).unwrap();

    for i in 0..size {
        for j in 0..size {
            let expected = (i + j) as f64 + (i * j) as f64;
            assert!((*m1.get(i, j).unwrap() - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn test_parallel_reductions() {
    let size = 1000; // Total 1M elements, should trigger parallel path
    let mut m = MatrixX::<f64>::new_dynamic(size, size).unwrap();
    for i in 0..size {
        for j in 0..size {
            *m.get_mut(i, j).unwrap() = 1.0;
        }
    }

    assert_eq!(m.sum(), (size * size) as f64);
    assert_eq!(m.min(), 1.0);
    assert_eq!(m.max(), 1.0);

    // Change one value
    *m.get_mut(500, 500).unwrap() = 2.0;
    assert_eq!(m.max(), 2.0);
    *m.get_mut(100, 100).unwrap() = -1.0;
    assert_eq!(m.min(), -1.0);
}

#[test]
fn test_parallel_spmv() {
    let rows = 1000;
    let cols = 1000;
    let mut sparse = SparseMatrix::<f64>::new(rows, cols, StorageOrder::RowMajor);
    let mut triplets = Vec::new();
    for i in 0..rows {
        triplets.push(Triplet::new(i, i, 2.0));
    }
    sparse.set_from_triplets(triplets);

    let mut dense = MatrixX::<f64>::new_dynamic(cols, 100).unwrap();
    for i in 0..cols {
        for j in 0..100 {
            *dense.get_mut(i, j).unwrap() = 1.0;
        }
    }

    // Should trigger parallel SpMV
    let res = sparse.mul_dense(&dense).unwrap();

    assert_eq!(res.rows(), rows);
    assert_eq!(res.cols(), 100);
    for i in 0..rows {
        for j in 0..100 {
            assert_eq!(*res.get(i, j).unwrap(), 2.0);
        }
    }
}
