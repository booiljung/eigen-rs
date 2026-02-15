use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::xpr::MatrixXpr;

#[test]
fn test_vec_dot_simd_correctness() {
    let size = 1024 + 17; // Use size > 64 and restricted to packet
    let mut v1 = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    let mut v2 = MatrixX::<f32>::new_dynamic(size, 1).unwrap();

    // Fill with values
    for i in 0..size {
         *v1.get_mut(i, 0).unwrap() = (i % 10) as f32;
         *v2.get_mut(i, 0).unwrap() = ((i + 1) % 10) as f32;
    }

    // Expected dot product
    let mut expected = 0.0;
    for i in 0..size {
        expected += *v1.get(i, 0).unwrap() * *v2.get(i, 0).unwrap();
    }

    // Actual
    let actual = v1.dot(&v2);

    println!("Expected: {}, Actual: {}", expected, actual);
    assert!((actual - expected).abs() < 1e-4);
}

#[test]
fn test_vec_dot_small() {
    let size = 13;
    let mut v1 = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    let mut v2 = MatrixX::<f32>::new_dynamic(size, 1).unwrap();

    for i in 0..size {
         *v1.get_mut(i, 0).unwrap() = 1.0;
         *v2.get_mut(i, 0).unwrap() = 2.0;
    }

    let expected = (size as f32) * 2.0;
    let actual = v1.dot(&v2);
    
    assert!((actual - expected).abs() < 1e-5);
}
