use eigen_rs::core::tensor::Tensor;
use eigen_rs::core::tensor::device::CpuDevice;

fn main() {
    println!("Testing Tensor Device Abstraction...");

    // 1. Create a 2x2 tensor (Default Device = CPU)
    let mut t1 = Tensor::<f32, 2>::new([2, 2]).unwrap();
    
    // Fill data (Col-Major)
    let data = t1.data_mut().unwrap();
    // Col 0: 1, 2
    data[0] = 1.0; // (0,0)
    data[1] = 2.0; // (1,0)
    // Col 1: 3, 4
    data[2] = 3.0; // (0,1)
    data[3] = 4.0; // (1,1)
    // Matrix form:
    // 1 3
    // 2 4

    println!("Tensor 1 created. Size: {}", t1.size());
    assert_eq!(t1.get([0, 0]), Some(&1.0));
    assert_eq!(t1.get([1, 1]), Some(&4.0));

    // 2. Create another tensor explicitly with CpuDevice
    let mut t2 = Tensor::<f32, 2, CpuDevice>::new_with_device([2, 2], CpuDevice::default()).unwrap();
    let data2 = t2.data_mut().unwrap();
    data2[0] = 1.0; // (0,0)
    data2[1] = 0.0; // (1,0)
    data2[2] = 0.0; // (0,1)
    data2[3] = 1.0; // (1,1)
    // Identity matrix
    
    // 3. Contract along dim 1 of t1 and dim 0 of t2 (MatMul-like)
    // t1 cols (dim 1) <-> t2 rows (dim 0)
    // Result should be 2x2
    let t3 = t1.contract(&t2, 1, 0);
    
    // Expected: 
    // [1 3] * [1 0] = [1 3]
    // [2 4]   [0 1]   [2 4]
    
    println!("Contraction result dims: {:?}", t3.dims());
    println!("Result data: {:?}", t3.data());

    assert_eq!(t3.get([0, 0]), Some(&1.0));
    assert_eq!(t3.get([1, 0]), Some(&2.0));
    assert_eq!(t3.get([0, 1]), Some(&3.0));
    assert_eq!(t3.get([1, 1]), Some(&4.0));

    println!("✅ Tensor Device Abstraction Test Passed!");
}
