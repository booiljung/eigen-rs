use eigen_rs::core::tensor::Tensor;

#[test]
fn test_tensor_contraction_simple() {
    // A: 2x3 tensor
    let mut a = Tensor::<f64, 2>::new([2, 3]).unwrap();
    // Fill A with 1.0, 2.0...
    // [1 3 5]
    // [2 4 6] (Note: column-major filling in assign logic usually, but let's check manually)
    // Actually Tensor::assign uses modular arithmetic which usually implies the last dimension is contiguous if indices are [i, j, k]
    // Let's just set manually.
    for i in 0..2 {
        for j in 0..3 {
            *a.get_mut([i, j]).unwrap() = (i + j) as f64;
        }
    }
    
    // B: 3x2 tensor
    let mut b = Tensor::<f64, 2>::new([3, 2]).unwrap();
    // Fill B
    for i in 0..3 {
        for j in 0..2 {
            *b.get_mut([i, j]).unwrap() = (i * j) as f64;
        }
    }
    
    // Contract A along dim 1 (size 3) and B along dim 0 (size 3)
    // Result should be 2x2.
    // C[i, j] = Sum_k A[i, k] * B[k, j] (Matrix Multiplication A * B)
    let c: Tensor<f64, 2> = a.contract(&b, 1, 0);
    
    assert_eq!(c.dims(), [2, 2]);
    
    // Expected:
    // A = [[0, 1, 2], [1, 2, 3]]
    // B = [[0, 0], [0, 1], [0, 2]]
    // C = A * B
    // c[0,0] = 0*0 + 1*0 + 2*0 = 0
    // c[0,1] = 0*0 + 1*1 + 2*2 = 5
    // c[1,0] = 1*0 + 2*0 + 3*0 = 0
    // c[1,1] = 1*0 + 2*1 + 3*2 = 8
    
    assert!((c.get([0, 0]).unwrap() - 0.0).abs() < 1e-10);
    assert!((c.get([0, 1]).unwrap() - 5.0).abs() < 1e-10);
    assert!((c.get([1, 0]).unwrap() - 0.0).abs() < 1e-10);
    assert!((c.get([1, 1]).unwrap() - 8.0).abs() < 1e-10);
}

#[test]
fn test_tensor_broadcasting() {
    // A: 2x1 tensor (column vector-like)
    let mut a = Tensor::<f64, 2>::new([2, 1]).unwrap();
    *a.get_mut([0, 0]).unwrap() = 1.0;
    *a.get_mut([1, 0]).unwrap() = 2.0;

    // Broadcast to 2x2
    // Expected: [[1, 1], [2, 2]]
    let b = a.broadcast([2, 2]).expect("Broadcast failed");

    assert_eq!(b.dims(), [2, 2]);
    assert_eq!(*b.get([0, 0]).unwrap(), 1.0);
    assert_eq!(*b.get([0, 1]).unwrap(), 1.0);
    assert_eq!(*b.get([1, 0]).unwrap(), 2.0);
    assert_eq!(*b.get([1, 1]).unwrap(), 2.0);
}

