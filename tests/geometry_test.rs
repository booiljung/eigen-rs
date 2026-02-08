use eigen_rs::core::geometry::{Quaternion, AngleAxis, Transform3};
use eigen_rs::core::matrix::Vector3;
mod common;

#[test]
fn test_quaternion_identity() {
    let q = Quaternion::<f32>::identity();
    assert_eq!(q.x(), 0.0);
    assert_eq!(q.y(), 0.0);
    assert_eq!(q.z(), 0.0);
    assert_eq!(q.w(), 1.0);
}

#[test]
fn test_quaternion_rotation() {
    // 90 degrees around Z axis
    let axis = Vector3::<f32>::new_fixed();
    // Manual axis setup
    let mut axis = axis;
    *axis.get_mut(0, 0).unwrap() = 0.0;
    *axis.get_mut(1, 0).unwrap() = 0.0;
    *axis.get_mut(2, 0).unwrap() = 1.0;
    
    let angle = std::f32::consts::PI / 2.0;
    let aa = AngleAxis::new(angle, axis);
    let q = aa.to_quaternion();
    
    // Vector (1, 0, 0) rotated by 90 deg around Z should be (0, 1, 0)
    let mut v = Vector3::<f32>::new_fixed();
    *v.get_mut(0, 0).unwrap() = 1.0;
    *v.get_mut(1, 0).unwrap() = 0.0;
    *v.get_mut(2, 0).unwrap() = 0.0;
    
    let v_rot = q.rotate_vector(&v);
    
    assert!((v_rot.get(0, 0).unwrap().abs()) < 1e-6);
    assert!((v_rot.get(1, 0).unwrap() - 1.0).abs() < 1e-6);
    assert!((v_rot.get(2, 0).unwrap().abs()) < 1e-6);
}

#[test]
fn test_transform_translation() {
    let mut t = Transform3::<f32>::identity();
    // Translation (1, 2, 3)
    *t.matrix_mut().get_mut(0, 3).unwrap() = 1.0;
    *t.matrix_mut().get_mut(1, 3).unwrap() = 2.0;
    *t.matrix_mut().get_mut(2, 3).unwrap() = 3.0;
    
    let mut v = Vector3::<f32>::new_fixed();
    *v.get_mut(0, 0).unwrap() = 10.0;
    *v.get_mut(1, 0).unwrap() = 20.0;
    *v.get_mut(2, 0).unwrap() = 30.0;
    
    let v_trans = t.transform_point(&v);
    
    assert_eq!(*v_trans.get(0, 0).unwrap(), 11.0);
    assert_eq!(*v_trans.get(1, 0).unwrap(), 22.0);
    assert_eq!(*v_trans.get(2, 0).unwrap(), 33.0);
}

#[test]
fn test_quaternion_to_matrix() {
    // 90 degrees around X axis
    let mut axis = Vector3::<f32>::new_fixed();
    *axis.get_mut(0, 0).unwrap() = 1.0;
    
    let angle = std::f32::consts::PI / 2.0;
    let q = AngleAxis::new(angle, axis).to_quaternion();
    let r = q.to_rotation_matrix();
    
    // Rotation matrix for 90 deg around X:
    // [1  0  0]
    // [0  0 -1]
    // [0  1  0]
    
    assert!((r.get(0, 0).unwrap() - 1.0).abs() < 1e-6);
    assert!((r.get(1, 1).unwrap().abs()) < 1e-6);
    assert!((r.get(1, 2).unwrap() + 1.0).abs() < 1e-6);
    assert!((r.get(2, 1).unwrap() - 1.0).abs() < 1e-6);
    assert!((r.get(2, 2).unwrap().abs()) < 1e-6);
}

#[test]
fn test_geometry_differential() {
    let cpp_output = common::run_cpp_harness_stdout("tests/cpp_harness/geometry_verify.cpp").unwrap();
    
    for line in cpp_output.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        match parts[0] {
            "Q_MUL" => {
                let x: f32 = parts[1].parse().unwrap();
                let y: f32 = parts[2].parse().unwrap();
                let z: f32 = parts[3].parse().unwrap();
                let w: f32 = parts[4].parse().unwrap();
                
                let mut q1 = Quaternion::new(0.1, 0.2, 0.3, 1.0);
                q1.normalize();
                let mut q2 = Quaternion::new(0.5, 0.4, 0.3, 1.0);
                q2.normalize();
                let q_mul = q1 * q2;
                
                assert!((q_mul.x() - x).abs() < 1e-5);
                assert!((q_mul.y() - y).abs() < 1e-5);
                assert!((q_mul.z() - z).abs() < 1e-5);
                assert!((q_mul.w() - w).abs() < 1e-5);
            },
            "Q_ROT" => {
                let x: f32 = parts[1].parse().unwrap();
                let y: f32 = parts[2].parse().unwrap();
                let z: f32 = parts[3].parse().unwrap();
                
                let mut q1 = Quaternion::new(0.1, 0.2, 0.3, 1.0);
                q1.normalize();
                let mut v = Vector3::<f32>::new_fixed();
                *v.get_mut(0, 0).unwrap() = 1.0;
                *v.get_mut(1, 0).unwrap() = 2.0;
                *v.get_mut(2, 0).unwrap() = 3.0;
                
                let v_rot = q1.rotate_vector(&v);
                assert!((v_rot.get(0, 0).unwrap() - x).abs() < 1e-5);
                assert!((v_rot.get(1, 0).unwrap() - y).abs() < 1e-5);
                assert!((v_rot.get(2, 0).unwrap() - z).abs() < 1e-5);
            },
            "Q_SLERP" => {
                let x: f32 = parts[1].parse().unwrap();
                let y: f32 = parts[2].parse().unwrap();
                let z: f32 = parts[3].parse().unwrap();
                let w: f32 = parts[4].parse().unwrap();
                
                let mut q1 = Quaternion::new(0.1, 0.2, 0.3, 1.0);
                q1.normalize();
                let mut q2 = Quaternion::new(0.5, 0.4, 0.3, 1.0);
                q2.normalize();
                let q_slerp = q1.slerp(0.5, &q2);
                
                assert!((q_slerp.x() - x).abs() < 1e-5);
                assert!((q_slerp.y() - y).abs() < 1e-5);
                assert!((q_slerp.z() - z).abs() < 1e-5);
                assert!((q_slerp.w() - w).abs() < 1e-5);
            },
            "T_TRANS" => {
                let x: f32 = parts[1].parse().unwrap();
                let y: f32 = parts[2].parse().unwrap();
                let z: f32 = parts[3].parse().unwrap();
                
                let mut t = Transform3::<f32>::identity();
                *t.matrix_mut().get_mut(0, 3).unwrap() = 1.0;
                *t.matrix_mut().get_mut(1, 3).unwrap() = 2.0;
                *t.matrix_mut().get_mut(2, 3).unwrap() = 3.0;
                
                let mut q1 = Quaternion::new(0.1, 0.2, 0.3, 1.0);
                q1.normalize();
                let r = q1.to_rotation_matrix();
                // Apply rotation to the top-left 3x3 of t
                for i in 0..3 {
                    for j in 0..3 {
                        *t.matrix_mut().get_mut(i, j).unwrap() = *r.get(i, j).unwrap();
                    }
                }
                
                let mut p = Vector3::<f32>::new_fixed();
                *p.get_mut(0, 0).unwrap() = 10.0;
                *p.get_mut(1, 0).unwrap() = 20.0;
                *p.get_mut(2, 0).unwrap() = 30.0;
                
                let p_trans = t.transform_point(&p);
                assert!((p_trans.get(0, 0).unwrap() - x).abs() < 1e-5);
                assert!((p_trans.get(1, 0).unwrap() - y).abs() < 1e-5);
                assert!((p_trans.get(2, 0).unwrap() - z).abs() < 1e-5, "T_TRANS fail at z: Rust={} C++={}", p_trans.get(2, 0).unwrap(), z);
            },
            _ => {}
        }
    }
}
