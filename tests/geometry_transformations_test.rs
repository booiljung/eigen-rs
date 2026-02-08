#[cfg(test)]
mod tests {
    use eigen_rs::core::geometry::{Translation, Scaling};
    use eigen_rs::core::matrix::Vector3;

    fn vec3(x: f64, y: f64, z: f64) -> Vector3<f64> {
        let mut v = Vector3::<f64>::zeros();
        *v.get_mut(0, 0).unwrap() = x;
        *v.get_mut(1, 0).unwrap() = y;
        *v.get_mut(2, 0).unwrap() = z;
        v
    }

    #[test]
    fn test_translation() {
        let t_vec = vec3(1.0, 2.0, 3.0);
        let t = Translation::<f64, 3>::new(t_vec);
        
        let p = vec3(10.0, 10.0, 10.0);
        let p_transformed = &t * &p;
        
        assert!((p_transformed.get(0, 0).unwrap() - 11.0).abs() < 1e-10);
        assert!((p_transformed.get(1, 0).unwrap() - 12.0).abs() < 1e-10);
        assert!((p_transformed.get(2, 0).unwrap() - 13.0).abs() < 1e-10);
        
        let t_inv = t.inverse();
        let p_identity = &t_inv * &p_transformed;
        assert!((p_identity.get(0, 0).unwrap() - 10.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_translation_composition() {
        let t1 = Translation::<f64, 3>::new(vec3(1.0, 0.0, 0.0));
        let t2 = Translation::<f64, 3>::new(vec3(0.0, 2.0, 0.0));
        
        let t3 = &t1 * &t2;
        let p = vec3(0.0, 0.0, 0.0);
        let p_transformed = &t3 * &p;
        
        assert!((p_transformed.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((p_transformed.get(1, 0).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_scaling() {
        let s = Scaling::<f64, 3>::uniform(2.0);
        let p = vec3(1.0, 2.0, 3.0);
        let p_scaled = &s * &p;
        
        assert!((p_scaled.get(0, 0).unwrap() - 2.0).abs() < 1e-10);
        assert!((p_scaled.get(1, 0).unwrap() - 4.0).abs() < 1e-10);
        assert!((p_scaled.get(2, 0).unwrap() - 6.0).abs() < 1e-10);
        
        let s_non_uniform = Scaling::<f64, 3>::new(vec3(0.5, 1.0, 2.0));
        let p_non_uniform = &s_non_uniform * &p;
        
        assert!((p_non_uniform.get(0, 0).unwrap() - 0.5).abs() < 1e-10);
        assert!((p_non_uniform.get(1, 0).unwrap() - 2.0).abs() < 1e-10);
        assert!((p_non_uniform.get(2, 0).unwrap() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_quaternion() {
        use eigen_rs::core::geometry::Quaternion;
        // Rotate 90 degrees around Z axis
        let angle = std::f64::consts::PI / 2.0;
        let axis = vec3(0.0, 0.0, 1.0);
        // q = [sin(theta/2) * axis, cos(theta/2)]
        let s = (angle / 2.0).sin();
        let c = (angle / 2.0).cos();
        let q = Quaternion::new(0.0, 0.0, 1.0 * s, c);
        
        let v = vec3(1.0, 0.0, 0.0);
        let v_rot = q.rotate_vector(&v);
        
        // Should be (0, 1, 0)
        assert!(v_rot.get(0, 0).unwrap().abs() < 1e-10);
        assert!((v_rot.get(1, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!(v_rot.get(2, 0).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_angle_axis() {
        use eigen_rs::core::geometry::AngleAxis;
        let angle = std::f64::consts::PI; // 180 degrees
        let axis = vec3(1.0, 0.0, 0.0); // X axis
        let aa = AngleAxis::new(angle, axis);
        
        let q = aa.to_quaternion();
        let v = vec3(0.0, 1.0, 0.0);
        let v_rot = q.rotate_vector(&v);
        
        // 180 deg around X should map (0, 1, 0) to (0, -1, 0)
        assert!(v_rot.get(0, 0).unwrap().abs() < 1e-10);
        assert!((v_rot.get(1, 0).unwrap() + 1.0).abs() < 1e-10);
        assert!(v_rot.get(2, 0).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_transform() {
        use eigen_rs::core::geometry::{Transform3, Quaternion, Translation, Scaling};
        
        // Create components
        let t_vec = vec3(1.0, 2.0, 3.0);
        let t = Translation::<f64, 3>::new(t_vec);
        let tf_t = Transform3::<f64>::from_translation(&t);
        
        // Rotate 90 degrees around Z axis
        let angle = std::f64::consts::PI / 2.0; 
        let s_angle = (angle/2.0).sin();
        let c_angle = (angle/2.0).cos();
        let q = Quaternion::new(0.0, 0.0, 1.0 * s_angle, c_angle);
        let tf_r = Transform3::<f64>::from_rotation(&q);
        
        let s = Scaling::<f64, 3>::uniform(2.0);
        let tf_s = Transform3::<f64>::from_scaling(&s);
        
        // Composition: T * R * S
        // p_trans = T * (R * (S * p))
        let composite = tf_t.mul(&tf_r).mul(&tf_s);
        
        // Point p = (1, 0, 0)
        // 1. Scale by 2 -> (2, 0, 0)
        // 2. Rotate 90 deg Z -> (0, 2, 0)
        // 3. Translate (1, 2, 3) -> (1, 4, 3)
        
        let p = vec3(1.0, 0.0, 0.0);
        let p_transformed = composite.transform_point(&p);
        
        assert!((p_transformed.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((p_transformed.get(1, 0).unwrap() - 4.0).abs() < 1e-10);
        assert!((p_transformed.get(2, 0).unwrap() - 3.0).abs() < 1e-10);
    }
}
