use eigen_rs::core::sparse::solvers::SimplicialLDLT;
use eigen_rs::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder, Triplet};
use eigen_rs::core::matrix::{Matrix, DynamicStorage};

#[test]
fn test_ldlt_basic() {
    // A = [ 2 -1  0 ]
    //     [-1  2 -1 ]
    //     [ 0 -1  2 ]
    // LDLT: 
    // L = [ 1    0    0 ]
    //     [-0.5  1    0 ]
    //     [ 0   -2/3  1 ]
    // D = [ 2, 1.5, 4/3 ]
    
    // 2 * (-0.5)^2 + 1.5 = 0.5 + 1.5 = 2. Correct.
    
    let mut a = SparseMatrix::<f64>::new(3, 3, StorageOrder::ColMajor);
    a.set_from_triplets(vec![
        Triplet::new(0, 0, 2.0), Triplet::new(0, 1, -1.0),
        Triplet::new(1, 0, -1.0), Triplet::new(1, 1, 2.0), Triplet::new(1, 2, -1.0),
        Triplet::new(2, 1, -1.0), Triplet::new(2, 2, 2.0),
    ]);

    let mut ldlt = SimplicialLDLT::new();
    ldlt.compute(&a).unwrap();

    let b = Matrix::<f64, DynamicStorage<f64>>::from_vec(3, 1, vec![1.0, 0.0, 1.0]).unwrap();
    // Solution should be [1, 1, 1].
    // 2*1 -1*1 = 1
    // -1*1 + 2*1 -1*1 = 0
    // -1*1 + 2*1 = 1
    
    let x = ldlt.solve(&b).unwrap();
    println!("x = {:?}", x);
    
    assert!((x.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
    assert!((x.get(1, 0).unwrap() - 1.0).abs() < 1e-10);
    assert!((x.get(2, 0).unwrap() - 1.0).abs() < 1e-10);
}
