use eigen_rs::core::matrix::MatrixX;
use eigen_rs::unsupported::kronecker_product;
use eigen_rs::core::xpr::MatrixXpr;

#[test]
fn test_kronecker_product_dense() {
    // A = [1, 2]
    //     [3, 4]
    let mut a = MatrixX::<f64>::new_dynamic(2, 2).unwrap();
    *a.get_mut(0, 0).unwrap() = 1.0; *a.get_mut(0, 1).unwrap() = 2.0;
    *a.get_mut(1, 0).unwrap() = 3.0; *a.get_mut(1, 1).unwrap() = 4.0;

    // B = [0, 5]
    //     [6, 7]
    let mut b = MatrixX::<f64>::new_dynamic(2, 2).unwrap();
    *b.get_mut(0, 0).unwrap() = 0.0; *b.get_mut(0, 1).unwrap() = 5.0;
    *b.get_mut(1, 0).unwrap() = 6.0; *b.get_mut(1, 1).unwrap() = 7.0;

    // Expected A (x) B:
    // [0, 5, 0, 10]
    // [6, 7, 12, 14]
    // [0, 15, 0, 20]
    // [18, 21, 24, 28]

    let k = kronecker_product(&a, &b);

    assert_eq!(k.rows(), 4);
    assert_eq!(k.cols(), 4);

    let expected_data = vec![
        0.0, 5.0, 0.0, 10.0,
        6.0, 7.0, 12.0, 14.0,
        0.0, 15.0, 0.0, 20.0,
        18.0, 21.0, 24.0, 28.0
    ];

    for i in 0..4 {
        for j in 0..4 {
            let val = k.eval(i, j);
            let expected_val = expected_data[i * 4 + j];
            assert!((val - expected_val).abs() < 1e-10, "Mismatch at ({}, {}): {} != {}", i, j, val, expected_val);
        }
    }
}

#[test]
fn test_kronecker_product_vectors() {
    // u = [1]
    //     [2]
    let mut u = MatrixX::<f64>::new_dynamic(2, 1).unwrap(); // Column vector
    *u.get_mut(0, 0).unwrap() = 1.0;
    *u.get_mut(1, 0).unwrap() = 2.0;

    // v = [3, 4]
    let mut v = MatrixX::<f64>::new_dynamic(1, 2).unwrap(); // Row vector
    *v.get_mut(0, 0).unwrap() = 3.0;
    *v.get_mut(0, 1).unwrap() = 4.0;

    // u (x) v =
    // [3, 4]
    // [6, 8]

    let k = kronecker_product(&u, &v);

    assert_eq!(k.rows(), 2);
    assert_eq!(k.cols(), 2);

    let expected_data = vec![
        3.0, 4.0,
        6.0, 8.0
    ];

    for i in 0..2 {
        for j in 0..2 {
            let val = k.eval(i, j);
            let expected_val = expected_data[i * 2 + j];
            assert!((val - expected_val).abs() < 1e-10, "Mismatch at ({}, {}): {} != {}", i, j, val, expected_val);
        }
    }
}
