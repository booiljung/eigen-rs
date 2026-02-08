
use eigen_rs::core::matrix::Vector3;
use eigen_rs::core::geometry::{Quaternion, AngleAxis, Translation3, Isometry3};

fn main() -> Result<(), String> {
    println!("=== Geometry Operations with eigen-rs ===");

    // 1. Quaternions
    println!("\n--- Quaternions ---");
    // Rotation of 90 degrees around Z axis
    let angle = std::f64::consts::PI / 2.0;
    let axis = Vector3::from_array([0.0, 0.0, 1.0]);
    // AngleAxis
    let aa = AngleAxis::new(angle, axis);
    // Convert to Quaternion
    let q = Quaternion::from_angle_axis(angle, axis);
    println!("Quaternion (90 deg Z): {:?}", q);

    let v = Vector3::from_array([1.0, 0.0, 0.0]);
    let v_rotated = q * v;
    println!("Rotated vector (1,0,0) -> {:?}", v_rotated); // Should be (0, 1, 0)

    // 2. Transformations
    println!("\n--- Transformations (Affine3) ---");
    let t_vec = Vector3::from_array([1.0, 2.0, 3.0]);
    let t = Translation3::new(t_vec);
    let r = q; // Rotation
    
    // Combine: T * R
    // Use Isometry3 (Transform3 alias)
    
    let iso = Isometry3::from_parts(t, r);
    println!("Isometry:\n{:?}", iso.matrix());
    
    let v2 = Vector3::from_array([1.0, 0.0, 0.0]);
    let v2_trans = iso.transform_point(&v2);
    println!("Transformed vector (1,0,0) -> {:?}", v2_trans);
    // Expected: Rotated to (0,1,0) then translated by (1,2,3) -> (1, 3, 3)

    // 3. Euler Angles
    println!("\n--- Euler Angles ---");
    // Placeholder API check
    // let euler = q.to_euler_angles(0, 1, 2); // ZYX
    // println!("Euler angles: {:?}", euler);
    
    Ok(())
}
