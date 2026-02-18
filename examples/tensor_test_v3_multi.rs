use eigen_rs::core::tensor::Tensor;
use eigen_rs::core::tensor::device::CpuDevice;

fn main() {
    println!("Testing Generalized Tensor Contraction...");

    // 1. Create two 3D tensors: 2x2x2
    // We will contract along dim 1 and dim 2
    // Result should be 2x2 ? No.
    // T1: 2x2x2. T2: 2x2x2.
    // Contract T1(i, j, k) with T2(x, y, z)
    // Contract T1 dim 1 (j) with T2 dim 0 (x)
    // Contract T1 dim 2 (k) with T2 dim 1 (y)
    // Free: T1(i), T2(z).
    // Result: 2x2 Tensor C(i, z).
    
    let mut t1 = Tensor::<f32, 3>::new([2, 2, 2]).unwrap();
    // Fill with 1.0
    for x in t1.data_mut() { *x = 1.0; }
    
    let mut t2 = Tensor::<f32, 3>::new([2, 2, 2]).unwrap();
    // Fill with 2.0
    for x in t2.data_mut() { *x = 2.0; }
    
    // Contraction pairs: (1, 0), (2, 1).
    let dims = [(1, 0), (2, 1)];
    
    // Output Rank: 3 + 3 - 2*2 = 2.
    let t3: Tensor<f32, 2> = t1.contract_dims(&t2, &dims);
    
    println!("Result Dims: {:?}", t3.dims()); // Should be [2, 2]
    
    // Expected Value:
    // Sum over j(Size 2), k(Size 2). Total 4 terms per element.
    // t1 value = 1. t2 value = 2. Product = 2.
    // Sum = 4 terms * 2 = 8.
    
    let data = t3.data();
    println!("Data: {:?}", data);
    
    for &val in data {
        assert_eq!(val, 8.0);
    }

    println!("✅ Multi-Dim Contraction Test Passed!");
}
