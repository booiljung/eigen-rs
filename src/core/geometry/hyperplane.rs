use crate::core::scalar::Scalar;
use crate::core::matrix::Matrix;
use crate::core::storage::FixedStorage;
use crate::core::geometry::ray::Ray;

/// A hyperplane is defined by the equation `normal . x + offset = 0`.
#[derive(Clone, Debug, PartialEq)]
pub struct Hyperplane<T: Scalar, const RANK: usize> {
    normal: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
    offset: T,
}

impl<T: Scalar, const RANK: usize> Hyperplane<T, RANK> {
    pub fn new(normal: Matrix<T, FixedStorage<T, RANK, 1, RANK>>, offset: T) -> Self {
        // Normal should theoretically be normalized for distance calculations to be metric.
        // But Eigen allows non-normalized. We assume user provides what they want or we normalize?
        // Eigen's Hyperplane(Normal, offset) constructor assumes normalized if you use signedDistance directly.
        // We'll store as is.
        Self { normal, offset }
    }
    
    pub fn from_normal_and_point(normal: Matrix<T, FixedStorage<T, RANK, 1, RANK>>, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> Self {
        // n . p + d = 0  => d = - n . p
        let mut dot = T::default();
        for i in 0..RANK {
            dot += *normal.get(i, 0).unwrap() * *point.get(i, 0).unwrap();
        }
        let offset = -dot;
        Self { normal, offset }
    }
    
    pub fn normalize(&mut self) {
        let mut norm_sq = T::default();
        for i in 0..RANK {
            let val = *self.normal.get(i, 0).unwrap();
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > T::default() {
            let inv_norm = T::from_f64(1.0) / norm;
            for i in 0..RANK {
                *self.normal.get_mut(i, 0).unwrap() *= inv_norm;
            }
            self.offset *= inv_norm;
        }
    }

    pub fn signed_distance(&self, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> T {
        let mut dot = T::default();
        for i in 0..RANK {
            dot += *self.normal.get(i, 0).unwrap() * *point.get(i, 0).unwrap();
        }
        dot + self.offset
    }
    
    pub fn projection(&self, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        let dist = self.signed_distance(point);
        let mut proj = *point;
        // p_proj = p - dist * n
        for i in 0..RANK {
            let n_i = *self.normal.get(i, 0).unwrap();
            *proj.get_mut(i, 0).unwrap() -= dist * n_i;
        }
        proj
    }
    
    pub fn intersection_with_ray(&self, ray: &Ray<T, RANK>) -> Option<T> {
        ray.intersects_plane(self)
    }
    
    pub fn normal(&self) -> &Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        &self.normal
    }
    
    pub fn offset(&self) -> T {
        self.offset
    }
}
