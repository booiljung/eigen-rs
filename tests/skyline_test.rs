use eigen_rs::core::matrix::MatrixX;
use eigen_rs::unsupported::skyline::SkylineMatrix;

#[test]
fn test_skyline_from_dense() {
    // 4x4 Symmetric Matrix
    // [ 4 -1  0  0 ]
    // [-1  4 -1  0 ]
    // [ 0 -1  4 -1 ]
    // [ 0  0 -1  4 ]
    // This is tridiagonal, so profile should be just bandwidth 1.

    let n = 4;
    let mut dense = MatrixX::<f64>::new_dynamic(n, n).unwrap();
    dense.set_zero();
    for i in 0..n {
        *dense.get_mut(i, i).unwrap() = 4.0;
        if i > 0 {
            *dense.get_mut(i, i - 1).unwrap() = -1.0;
        }
        if i < n - 1 {
            *dense.get_mut(i, i + 1).unwrap() = -1.0;
        }
    }

    let skyline = SkylineMatrix::from_dense(&dense);

    // Verify Diagonal
    for i in 0..n {
        assert_eq!(skyline.coeff(i, i), 4.0);
    }

    // Verify Lower Profile elements
    assert_eq!(skyline.coeff(1, 0), -1.0);
    assert_eq!(skyline.coeff(2, 1), -1.0);
    assert_eq!(skyline.coeff(3, 2), -1.0);

    // Verify Zero outside profile (implicitly)
    assert_eq!(skyline.coeff(2, 0), 0.0);

    // Check internal structure for tridiagonal
    // Row 0: first_nz=0 -> bandwidth 0 (no lower)
    // Row 1: first_nz=0 -> col 0 is NZ. i=1. len = 1-0 = 1.
    // Row 2: first_nz=1 -> col 1 is NZ. i=2. len = 2-1 = 1.
    // Row 3: first_nz=2 -> col 2 is NZ. i=3. len = 3-2 = 1.
    // Total lower size: 0 + 1 + 1 + 1 = 3.
    assert_eq!(skyline.storage.lower.len(), 3);
}

#[test]
fn test_skyline_variable_bandwidth() {
    // 5x5 Arrowhead Matrix
    // [ 1  1  1  1  1 ]
    // [ 1  2  0  0  0 ]
    // [ 1  0  3  0  0 ]
    // [ 1  0  0  4  0 ]
    // [ 1  0  0  0  5 ]
    // From lower perspective:
    // Row 0: diag
    // Row 1: (1,0)=1. first_col=0.
    // Row 2: (2,0)=1. first_col=0. (2,1)=0 stored? Yes because profile ensures (2,0) is stored.
    // So row 2 stores from col 0 to 1.
    // Row 3: (3,0)=1. first_col=0. Stores 0,1,2.
    // Row 4: (4,0)=1. first_col=0. Stores 0,1,2,3.

    let n = 5;
    let mut dense = MatrixX::<f64>::new_dynamic(n, n).unwrap();
    dense.set_zero();
    for i in 0..n {
        *dense.get_mut(i, i).unwrap() = (i + 1) as f64;
    }
    // Fill first row/col
    for i in 1..n {
        *dense.get_mut(i, 0).unwrap() = 1.0;
        *dense.get_mut(0, i).unwrap() = 1.0;
    }

    let mut skyline = SkylineMatrix::from_dense(&dense);

    assert_eq!(skyline.coeff(0, 0), 1.0);
    assert_eq!(skyline.coeff(4, 0), 1.0);

    // Zeros inside envelope should be stored as 0.0 but retrievable
    assert_eq!(skyline.coeff(4, 3), 0.0);

    // Modify a value inside envelope
    *skyline.coeff_mut(4, 3) = 9.0;
    assert_eq!(skyline.coeff(4, 3), 9.0);
}

#[test]
fn test_skyline_from_sparse() {
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};

    // 4x4 Sparse Matrix (Tridiagonal)
    // Same as dense test
    let n = 4;
    let mut sparse = SparseMatrix::<f64>::new(n, n, StorageOrder::ColMajor);

    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push(Triplet::new(i, i, 4.0));
        if i > 0 {
            triplets.push(Triplet::new(i, i - 1, -1.0));
        }
        if i < n - 1 {
            triplets.push(Triplet::new(i, i + 1, -1.0));
        }
    }
    sparse.set_from_triplets(triplets);

    // Now test direct from_sparse
    let skyline = SkylineMatrix::from_sparse(&sparse);
    
    // Verify same properties
    assert_eq!(skyline.coeff(1, 0), -1.0);
    assert_eq!(skyline.coeff(2, 0), 0.0);
    assert_eq!(skyline.storage.lower.len(), 3);
}
