use eigen_rs::core::arch::{Packet, ScalarPacket};
use eigen_rs::core::matrix::Matrix;
use eigen_rs::core::storage::DynamicStorage;
use eigen_rs::core::xpr::MatrixXpr;

#[cfg(target_arch = "x86_64")]
use eigen_rs::core::arch::x86::SsePacketF32;

#[test]
fn test_simd_scalar_packet_eval() {
    let mut a = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(4, 4).unwrap();
    let mut b = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(4, 4).unwrap();

    for i in 0..4 {
        for j in 0..4 {
            *a.get_mut(i, j).unwrap() = (i + j) as f32;
            *b.get_mut(i, j).unwrap() = 1.0;
        }
    }

    let sum_expr = &a + &b;

    // Verify ScalarPacket eval (size 1)
    for i in 0..4 {
        for j in 0..4 {
            let p: ScalarPacket<f32> = sum_expr.packet_eval(i, j);
            assert_eq!(p.0, (i + j + 1) as f32);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_simd_sse_packet_eval() {
    let mut a = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(8, 4).unwrap();
    for i in 0..8 {
        for j in 0..4 {
            *a.get_mut(i, j).unwrap() = (i + j * 10) as f32;
        }
    }

    // In Column-Major, packet_eval(0, 0) should load elements from the first column: (0,0), (1,0), (2,0), (3,0)
    let p: SsePacketF32 = a.packet_eval(0, 0);
    let mut res = [0.0f32; 4];
    unsafe {
        p.store(res.as_mut_ptr());
    }

    assert_eq!(res[0], 0.0);
    assert_eq!(res[1], 1.0);
    assert_eq!(res[2], 2.0);
    assert_eq!(res[3], 3.0);

    // Test arithmetic propagation
    let sum_expr = &a + &a;
    let p2: SsePacketF32 = sum_expr.packet_eval(0, 0);
    unsafe {
        p2.store(res.as_mut_ptr());
    }
    assert_eq!(res[0], 0.0);
    assert_eq!(res[1], 2.0);
    assert_eq!(res[2], 4.0);
    assert_eq!(res[3], 6.0);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_simd_gemm_blocked_correctness() {
    use eigen_rs::core::ops::gemm::gemm_cm;

    // Test with a larger matrix that exceeds BLOCK_SIZE (64) or at least tests the logic
    // For unit test, 128x128 is reasonable.
    let m = 128;
    let mut a = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(m, m).unwrap();
    let mut b = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(m, m).unwrap();
    let mut c = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(m, m).unwrap();

    for i in 0..m {
        for j in 0..m {
            *a.get_mut(i, j).unwrap() = (i + j) as f32;
            *b.get_mut(i, j).unwrap() = if i == j { 1.0 } else { 0.0 };
        }
    }

    gemm_cm(&a, &b, &mut c).unwrap();

    for i in 0..m {
        for j in 0..m {
            let val = *c.get(i, j).unwrap();
            let expected = *a.get(i, j).unwrap();
            if (val - expected).abs() > 1e-5 {
                panic!("GEMM mismatch at ({}, {}): {} != {}", i, j, val, expected);
            }
        }
    }
}
