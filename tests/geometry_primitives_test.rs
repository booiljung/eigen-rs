#[cfg(test)]
mod tests {
    use eigen_rs::core::geometry::{AlignedBox, Hyperplane, ParametrizedLine, Ray};
    use eigen_rs::core::matrix::Vector3;

    fn vec3(x: f64, y: f64, z: f64) -> Vector3<f64> {
        let mut v = Vector3::<f64>::zeros();
        *v.get_mut(0, 0).unwrap() = x;
        *v.get_mut(1, 0).unwrap() = y;
        *v.get_mut(2, 0).unwrap() = z;
        v
    }

    #[test]
    fn test_aligned_box() {
        let min = vec3(0.0, 0.0, 0.0);
        let max = vec3(1.0, 1.0, 1.0);
        let mut box3 = AlignedBox::<f64, 3>::new(min, max);

        assert!(box3.contains(&min));
        assert!(box3.contains(&max));
        assert!(box3.contains(&vec3(0.5, 0.5, 0.5)));

        let outside = vec3(1.5, 0.5, 0.5);
        assert!(!box3.contains(&outside));

        let p = vec3(2.0, 2.0, 2.0);
        box3.extend(&p);
        assert!(box3.contains(&p));
        assert!((box3.max().get(0, 0).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperplane() {
        // Plane z = 0, normal = [0, 0, 1], offset = 0
        let normal = vec3(0.0, 0.0, 1.0);
        let plane = Hyperplane::<f64, 3>::new(normal, 0.0);

        let p1 = vec3(1.0, 2.0, 3.0);
        let dist = plane.signed_distance(&p1);
        assert!((dist - 3.0).abs() < 1e-10);

        // Helper to check vector approx equality
        let proj = plane.projection(&p1);
        assert!((proj.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((proj.get(1, 0).unwrap() - 2.0).abs() < 1e-10);
        assert!(proj.get(2, 0).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_parametrized_line() {
        // Line along x-axis
        let origin = Vector3::<f64>::zeros();
        let direction = vec3(1.0, 0.0, 0.0);
        let line = ParametrizedLine::<f64, 3>::new(origin, direction);

        let p = line.point_at(2.0);
        assert!((p.get(0, 0).unwrap() - 2.0).abs() < 1e-10);

        let p_off = vec3(2.0, 3.0, 0.0);
        let dist = line.distance(&p_off);
        assert!((dist - 3.0).abs() < 1e-10);

        let proj = line.projection(&p_off);
        assert!((proj.get(0, 0).unwrap() - 2.0).abs() < 1e-10);
        assert!(proj.get(1, 0).unwrap().abs() < 1e-10);
    }

    #[test]
    fn test_ray() {
        // Ray along x-axis from origin
        let origin = Vector3::<f64>::zeros();
        let direction = vec3(1.0, 0.0, 0.0);
        let ray = Ray::<f64, 3>::new(origin, direction);

        // Point "behind" the ray
        let p_behind = vec3(-2.0, 3.0, 0.0);
        // Projection onto line would be (-2, 0, 0), dist 3.
        // Projection onto RAY should be clamped to origin (0, 0, 0)? No, t_clamped = 0.
        // So dist is dist(origin, p_behind) = sqrt(4 + 9) = sqrt(13) ~ 3.605

        let dist = ray.distance(&p_behind);
        let expected = (4.0f64 + 9.0).sqrt();
        assert!((dist - expected).abs() < 1e-10);

        let p_forward = vec3(2.0, 3.0, 0.0);
        let dist_f = ray.distance(&p_forward);
        assert!((dist_f - 3.0).abs() < 1e-10);
    }
}
