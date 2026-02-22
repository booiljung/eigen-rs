use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eigen_rs::core::geometry::{AlignedBox, EulerAngles, Ray};
use eigen_rs::core::matrix::Vector3;

fn vec3(x: f64, y: f64, z: f64) -> Vector3<f64> {
    let mut v = Vector3::<f64>::zeros();
    *v.get_mut(0, 0).unwrap() = x;
    *v.get_mut(1, 0).unwrap() = y;
    *v.get_mut(2, 0).unwrap() = z;
    v
}

fn bench_dot(c: &mut Criterion) {
    let v1 = vec3(1.0, 2.0, 3.0);
    let v2 = vec3(4.0, 5.0, 6.0);

    c.bench_function("vector3_dot", |b| {
        b.iter(|| black_box(&v1).dot(black_box(&v2)))
    });
}

fn bench_cross(c: &mut Criterion) {
    let v1 = vec3(1.0, 2.0, 3.0);
    let v2 = vec3(4.0, 5.0, 6.0);

    c.bench_function("vector3_cross", |b| {
        b.iter(|| black_box(&v1).cross(black_box(&v2)))
    });
}

fn bench_euler_conversion(c: &mut Criterion) {
    let ea = EulerAngles::new(0.5, 0.4, 0.3);

    c.bench_function("euler_to_quaternion", |b| {
        b.iter(|| black_box(&ea).to_quaternion())
    });

    let q = ea.to_quaternion();
    let mat = q.to_rotation_matrix();

    c.bench_function("rotation_matrix_to_euler", |b| {
        b.iter(|| EulerAngles::from_rotation_matrix(black_box(&mat)))
    });
}

fn bench_ray_aabb(c: &mut Criterion) {
    let min = vec3(-1.0, -1.0, -1.0);
    let max = vec3(1.0, 1.0, 1.0);
    let box_ = AlignedBox::<f64, 3>::new(min, max);

    let origin = vec3(0.0, 0.0, 10.0);
    let dir = vec3(0.0, 0.0, -1.0);
    let ray = Ray::<f64, 3>::new(origin, dir);

    c.bench_function("ray_aabb_intersection", |b| {
        b.iter(|| black_box(&box_).intersects(black_box(&ray)))
    });
}

criterion_group!(
    benches,
    bench_dot,
    bench_cross,
    bench_euler_conversion,
    bench_ray_aabb
);
criterion_main!(benches);
