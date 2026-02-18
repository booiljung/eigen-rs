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

#[test]
fn test_tensor_device_creation() {
    use eigen_rs::core::tensor::device::CpuDevice;
    
    // Explicit device creation
    let t = Tensor::<f32, 2, CpuDevice>::new_with_device([2, 2], CpuDevice::default()).unwrap();
    assert_eq!(t.dims(), [2, 2]);
    assert_eq!(t.size(), 4);
    
    // Implicit (default generic)
    let t2 = Tensor::<f32, 2>::new([2, 2]).unwrap();
    assert_eq!(t2.dims(), [2, 2]);
}

#[test]
fn test_tensor_multi_dim_contraction() {
    // 2x2x2
    let mut t1 = Tensor::<f32, 3>::new([2, 2, 2]).unwrap();
    for x in t1.data_mut() { *x = 1.0; }
    
    let mut t2 = Tensor::<f32, 3>::new([2, 2, 2]).unwrap();
    for x in t2.data_mut() { *x = 2.0; }
    
    // Contract (1, 0) and (2, 1)
    let dims = [(1, 0), (2, 1)];
    let t3: Tensor<f32, 2> = t1.contract_dims(&t2, &dims);
    
    assert_eq!(t3.dims(), [2, 2]);
    for x in t3.data() {
        assert_eq!(*x, 8.0);
    }
}

#[test]
fn test_tensor_permute_reshape() {
    // 2x3 Tensor
    let mut t = Tensor::<f32, 2>::new([2, 3]).unwrap();
    // Fill with linear index
    let mut idx = 0.0;
    // Col-major fill in memory naturally via get_mut
    for j in 0..3 {
       for i in 0..2 {
           *t.get_mut([i, j]).unwrap() = idx;
           idx += 1.0;
       }
    }
    // Data: [0, 1, 2, 3, 4, 5]
    // Matrix:
    // 0 2 4
    // 1 3 5
    
    // 1. Reshape to 3x2
    // Strided reshape (linear copy)
    // New Memory: [0, 1, 2, 3, 4, 5]
    // New Matrix (3x2 ColMajor):
    // 0 3
    // 1 4
    // 2 5
    let t_reshaped = t.reshape([3, 2]).unwrap();
    assert_eq!(t_reshaped.dims(), [3, 2]);
    assert_eq!(t_reshaped.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(*t_reshaped.get([1, 0]).unwrap(), 1.0);
    assert_eq!(*t_reshaped.get([0, 1]).unwrap(), 3.0);
    
    // 2. Permute (Transpose)
    // Swap dim 0 and 1.
    // Original: 2x3. New: 3x2.
    // Matrix:
    // 0 1
    // 2 3
    // 4 5
    let t_permuted = t.permute([1, 0]).unwrap();
    assert_eq!(t_permuted.dims(), [3, 2]);
    assert_eq!(*t_permuted.get([0, 0]).unwrap(), 0.0);
    assert_eq!(*t_permuted.get([1, 0]).unwrap(), 2.0); // Was (0,1) -> 2.0
    assert_eq!(*t_permuted.get([2, 0]).unwrap(), 4.0);
    assert_eq!(*t_permuted.get([0, 1]).unwrap(), 1.0); // Was (1,0) -> 1.0
}
