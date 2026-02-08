
use eigen_rs::{Matrix, Map};
use eigen_rs::core::storage::DynamicStorage;

#[test]
fn test_set_methods() {
    let mut m = Matrix::<f64, DynamicStorage<f64>>::new_dynamic(3, 3).unwrap();
    
    // setConstant
    m.set_constant(5.0);
    for i in 0..3 { for j in 0..3 { assert_eq!(*m.get(i, j).unwrap(), 5.0); } }

    // setZero
    m.set_zero();
    for i in 0..3 { for j in 0..3 { assert_eq!(*m.get(i, j).unwrap(), 0.0); } }

    // setIdentity
    m.set_identity();
    for i in 0..3 { 
        for j in 0..3 { 
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(*m.get(i, j).unwrap(), expected); 
        } 
    }
}

#[test]
fn test_map_read_write() {
    // 1. Create a raw buffer (e.g., from C)
    let rows = 3;
    let cols = 3;
    let mut buffer = vec![0.0f64; rows * cols];
    // Fill buffer: 0, 1, 2, ...
    for i in 0..buffer.len() { buffer[i] = i as f64; }

    unsafe {
        // 2. Map it
        let mut map = Map::<f64>::new(buffer.as_mut_ptr(), rows, cols);
        
        // Check reading
        // Column-major: 
        // 0 3 6
        // 1 4 7
        // 2 5 8
        assert_eq!(*map.get(0, 1).unwrap(), 3.0);
        assert_eq!(*map.get(1, 1).unwrap(), 4.0);
        
        // Check writing
        *map.get_mut(0, 0).unwrap() = 100.0;
        assert_eq!(buffer[0], 100.0); // Buffer should be modified
    }
}
