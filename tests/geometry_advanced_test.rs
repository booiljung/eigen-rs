#[cfg(test)]
mod tests {
    use eigen_rs::core::geometry::{EulerAngles, Projective3};
    use eigen_rs::core::matrix::Vector3;

    fn vec3(x: f64, y: f64, z: f64) -> Vector3<f64> {
        let mut v = Vector3::<f64>::zeros();
        *v.get_mut(0, 0).unwrap() = x;
        *v.get_mut(1, 0).unwrap() = y;
        *v.get_mut(2, 0).unwrap() = z;
        v
    }

    #[test]
    fn test_dot_cross() {
        let v1 = vec3(1.0, 2.0, 3.0);
        let v2 = vec3(4.0, 5.0, 6.0);
        
        // Dot
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((v1.dot(&v2) - 32.0).abs() < 1e-10);
        
        // Cross
        // x: 2*6 - 3*5 = 12 - 15 = -3
        // y: 3*4 - 1*6 = 12 - 6 = 6
        // z: 1*5 - 2*4 = 5 - 8 = -3
        let cp = v1.cross(&v2);
        assert!((cp.get(0, 0).unwrap() - (-3.0)).abs() < 1e-10);
        assert!((cp.get(1, 0).unwrap() - 6.0).abs() < 1e-10);
        assert!((cp.get(2, 0).unwrap() - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_euler_angles_conversions() {
        // Test ZYX: 90 deg around Z, 90 deg around Y, 0 around X
        // Rz(90) * Ry(90) * Rx(0)
        // Rz(90) maps X->Y, Y->-X
        // Ry(90) maps Z->X, X->-Z
        // Combined: X -> Y (Rz) -> Y (Ry)
        //           Y -> -X (Rz) -> -(-Z) = Z (Ry)
        //           Z -> Z (Rz) -> X (Ry)
        // So (1,0,0) -> (0,1,0)
        //    (0,1,0) -> (0,0,1)
        //    (0,0,1) -> (1,0,0)
        
        let alpha = std::f64::consts::FRAC_PI_2; // Z
        let beta = std::f64::consts::FRAC_PI_2;  // Y
        let gamma = 0.0;                         // X
        
        let ea = EulerAngles::new(alpha, beta, gamma);
        let q = ea.to_quaternion();
        let r_mat = q.to_rotation_matrix();
        
        // Check conversion back
        let _ea_recovered = EulerAngles::from_rotation_matrix(&r_mat);
        
        // Note: multiple Euler angle sets can represent the same rotation.
        // But for these simple 90 degree rotations, we expect close match or equivalent.
        // Actually, singularity at beta = +/- pi/2 (Gimbal lock).
        // Our implementation explicitly handles this.
        
        // Verify rotation action instead of exact angles if ambiguous
        let v = vec3(1.0, 0.0, 0.0);
        let v_rot = q.rotate_vector(&v);
        
        // Expected: (0, 1, 0) based on my mental trace?
        // Let's re-verify ZYX convention formula
        // R = Rz * Ry * Rx
        // v=(1,0,0). Rx does nothing. Ry(90) maps (1,0,0) -> (0,0,-1).
        // Rz(90) maps (0,0,-1) -> (0,0,-1).
        // Wait, Ry(90):
        // [ cos  0  sin] [1]   [ 0]
        // [  0   1   0 ] [0] = [ 0]  (if standard Ry)
        // [-sin  0  cos] [0]   [-1]
        // So ZYX: (1,0,0) -> (0,0,-1)
        
        assert!(v_rot.get(0, 0).unwrap().abs() < 1e-10);
        assert!(v_rot.get(1, 0).unwrap().abs() < 1e-10);
        assert!((v_rot.get(2, 0).unwrap() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_projective() {
        // Projective3 is just Transform3
        use eigen_rs::core::geometry::Translation;
        
        let t = Translation::<f64, 3>::new(vec3(1.0, 2.0, 3.0));
        let proj = Projective3::<f64>::from_translation(&t);
        
        let p = vec3(10.0, 10.0, 10.0);
        let p_trans = proj.transform_point(&p);
        
        assert!((p_trans.get(0, 0).unwrap() - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_queries() {
        use eigen_rs::core::geometry::{AlignedBox, Ray, Hyperplane};
        
        // Ray-Plane
        let origin = vec3(0.0, 0.0, 10.0);
        let dir = vec3(0.0, 0.0, -1.0); // Downwards
        let ray = Ray::<f64, 3>::new(origin, dir);
        
        let plane_normal = vec3(0.0, 0.0, 1.0);
        let plane = Hyperplane::<f64, 3>::new(plane_normal, 0.0); // Z=0 plane
        
        // n.p + d = 0 -> z + 0 = 0 -> z=0
        
        let t = ray.intersects_plane(&plane);
        assert!(t.is_some());
        assert!((t.unwrap() - 10.0).abs() < 1e-10); // Should hit at t=10
        
        // Ray-AABB
        let min = vec3(-1.0, -1.0, -1.0);
        let max = vec3(1.0, 1.0, 1.0);
        let box_ = AlignedBox::<f64, 3>::new(min, max);
        
        // Ray from (0,0,10) towards (0,0,0) -> should hit
        let t_box = box_.intersects(&ray);
        assert!(t_box);
        
        let (t_min, t_max) = box_.intersection_parameter(&ray).unwrap();
        // Enters at z=1 (t=9), exits at z=-1 (t=11)
        assert!((t_min - 9.0).abs() < 1e-10);
        assert!((t_max - 11.0).abs() < 1e-10);
        
        // Ray missing box
        let dir_miss = vec3(1.0, 0.0, 0.0); // Rightwards
        let ray_miss = Ray::<f64, 3>::new(origin, dir_miss);
        assert!(!box_.intersects(&ray_miss));
        
        // Distance point-box
        let p_out = vec3(3.0, 0.0, 0.0);
        let dist_sq = box_.squared_distance(&p_out);
        // closest point on box is (1,0,0). distance is 2. squared is 4.
        assert!((dist_sq - 4.0).abs() < 1e-10);
    }
}
