use eigen_rs::core::geometry::{Quaternion, Transform, Translation};
use eigen_rs::core::matrix::Vector3;
use std::time::Instant;

fn main() {
    println!("Benchmarking Geometry Module (Scalar vs SIMD)...");

    const ITERATIONS: usize = 100_000;

    // 1. Quaternion Multiplication
    let q1 = Quaternion::<f32>::new(0.1, 0.2, 0.3, 0.9);
    let q1 = {
        let mut q = q1;
        q.normalize();
        q.conjugate()
    };

    let q2 = Quaternion::<f32>::new(0.4, 0.5, 0.6, 0.1);
    let q2 = {
        let mut q = q2;
        q.normalize();
        q.conjugate()
    };

    // normalization check
    let mut q1 = Quaternion::<f32>::new(0.1, 0.2, 0.3, 0.9);
    q1.normalize();
    let mut q2 = Quaternion::<f32>::new(0.4, 0.5, 0.6, 0.1);
    q2.normalize();

    // Scalar Baseline
    let start = Instant::now();
    let mut res_scalar = q1;
    for _ in 0..ITERATIONS {
        res_scalar = res_scalar * q2;
        // prevent explosion
        res_scalar.normalize();
    }
    let dur_scalar = start.elapsed();
    println!("Quaternion Mul (Scalar): {:?}", dur_scalar);

    let start = Instant::now();
    let mut res_simd = q1;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        unsafe {
            for _ in 0..ITERATIONS {
                res_simd = res_simd.hamilton_product_simd(&q2);
                res_simd.normalize(); // Normalize to keep it fair/comparable and prevent explosion
            }
        }
    }
    let dur_simd = start.elapsed();
    println!("Quaternion Mul (SIMD):   {:?}", dur_simd);

    // Verify Correctness
    let diff = (res_scalar.x() - res_simd.x()).abs()
        + (res_scalar.y() - res_simd.y()).abs()
        + (res_scalar.z() - res_simd.z()).abs()
        + (res_scalar.w() - res_simd.w()).abs();
    println!("Quaternion diff: {:.2e}", diff);

    // 2. Quaternion Rotation
    let v = Vector3::<f32>::from_array([1.0, 2.0, 3.0]);

    let start = Instant::now();
    let mut v_scalar = v;
    for _ in 0..ITERATIONS {
        v_scalar = q1 * v_scalar;
    }
    let dur_rot_scalar = start.elapsed();
    println!("Quaternion Rot (Scalar): {:?}", dur_rot_scalar);

    let start = Instant::now();
    let mut v_simd = v;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        unsafe {
            for _ in 0..ITERATIONS {
                v_simd = q1.rotate_vector_simd(&v_simd);
            }
        }
    }
    let dur_rot_simd = start.elapsed();
    println!("Quaternion Rot (SIMD):   {:?}", dur_rot_simd);

    let v_diff_x = (v_scalar.get(0, 0).unwrap() - v_simd.get(0, 0).unwrap()).abs();
    let v_diff_y = (v_scalar.get(1, 0).unwrap() - v_simd.get(1, 0).unwrap()).abs();
    let v_diff_z = (v_scalar.get(2, 0).unwrap() - v_simd.get(2, 0).unwrap()).abs();
    let prob_diff_rot = v_diff_x + v_diff_y + v_diff_z;
    println!("Rotation diff: {:.2e}", prob_diff_rot);

    // 3. Transform Multiplication
    let t1 = Transform::<f32, 4, 3, 16>::from_parts(
        Translation::new(Vector3::from_array([1.0, 2.0, 3.0])),
        q1,
    );
    let t2 = Transform::<f32, 4, 3, 16>::from_parts(
        Translation::new(Vector3::from_array([4.0, 5.0, 6.0])),
        q2,
    );

    let start = Instant::now();
    let mut tf_scalar = t1.clone();
    for _ in 0..ITERATIONS {
        tf_scalar = tf_scalar.mul(&t2);
    }
    let dur_tf_scalar = start.elapsed();
    println!("Transform Mul (Scalar):  {:?}", dur_tf_scalar);

    let start = Instant::now();
    let mut tf_simd = t1.clone();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        unsafe {
            for _ in 0..ITERATIONS {
                tf_simd = tf_simd.mul_simd(&t2);
            }
        }
    }
    let dur_tf_simd = start.elapsed();
    println!("Transform Mul (SIMD):    {:?}", dur_tf_simd);

    // Check diff
    let mut diff_tf = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            diff_tf += (*tf_scalar.matrix().get(i, j).unwrap()
                - *tf_simd.matrix().get(i, j).unwrap())
            .abs();
        }
    }
    println!("Transform diff: {:.2e}", diff_tf);

    // 4. Transform Point
    let p = Vector3::<f32>::from_array([10.0, 20.0, 30.0]);

    let start = Instant::now();
    let mut tp_scalar = p;
    for _ in 0..ITERATIONS {
        tp_scalar = t1.transform_point(&tp_scalar);
        // prevent explosion
        if tp_scalar.get(0, 0).unwrap().abs() > 1e10 {
            tp_scalar.set_constant(1.0);
        }
    }
    let dur_tp_scalar = start.elapsed();
    println!("Transform Point(Scalar): {:?}", dur_tp_scalar);

    let start = Instant::now();
    let mut tp_simd = p;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        unsafe {
            for _ in 0..ITERATIONS {
                tp_simd = t1.transform_point_simd(&tp_simd);
                if tp_simd.get(0, 0).unwrap().abs() > 1e10 {
                    tp_simd.set_constant(1.0);
                }
            }
        }
    }
    let dur_tp_simd = start.elapsed();
    println!("Transform Point(SIMD):   {:?}", dur_tp_simd);

    let tp_diff_x = (tp_scalar.get(0, 0).unwrap() - tp_simd.get(0, 0).unwrap()).abs();
    let tp_diff_y = (tp_scalar.get(1, 0).unwrap() - tp_simd.get(1, 0).unwrap()).abs();
    let tp_diff_z = (tp_scalar.get(2, 0).unwrap() - tp_simd.get(2, 0).unwrap()).abs();
    let diff_tp = tp_diff_x + tp_diff_y + tp_diff_z;
    println!("Transform Point diff: {:.2e}", diff_tp);
}
