use crate::core::geometry::ray::Ray;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::FixedStorage;

/// Axis-Aligned Bounding Box.
#[derive(Clone, Debug, PartialEq)]
pub struct AlignedBox<T: Scalar, const RANK: usize> {
    min: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
    max: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
}

impl<T: Scalar, const RANK: usize> AlignedBox<T, RANK> {
    /// Creates a new AABB with given min and max points.
    pub fn new(
        min: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
        max: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
    ) -> Self {
        Self { min, max }
    }

    /// Creates an empty AABB (min > max).
    pub fn new_empty() -> Self {
        let max_val = T::from_f64(f64::MAX);
        let min_val = T::from_f64(f64::MIN);

        let mut min = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();
        let mut max = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();

        for i in 0..RANK {
            *min.get_mut(i, 0).unwrap() = max_val;
            *max.get_mut(i, 0).unwrap() = min_val;
        }

        Self { min, max }
    }

    pub fn min(&self) -> &Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        &self.min
    }

    pub fn max(&self) -> &Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        &self.max
    }

    pub fn center(&self) -> Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        let mut center = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();
        for i in 0..RANK {
            let min_val = *self.min.get(i, 0).unwrap();
            let max_val = *self.max.get(i, 0).unwrap();
            *center.get_mut(i, 0).unwrap() = (min_val + max_val) / T::from_f64(2.0);
        }
        center
    }

    pub fn sizes(&self) -> Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        let mut sizes = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();
        for i in 0..RANK {
            let min_val = *self.min.get(i, 0).unwrap();
            let max_val = *self.max.get(i, 0).unwrap();
            *sizes.get_mut(i, 0).unwrap() = (max_val - min_val).abs();
        }
        sizes
    }

    pub fn contains(&self, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> bool {
        for i in 0..RANK {
            let p = *point.get(i, 0).unwrap();
            if p < *self.min.get(i, 0).unwrap() || p > *self.max.get(i, 0).unwrap() {
                return false;
            }
        }
        true
    }

    pub fn extend(&mut self, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) {
        for i in 0..RANK {
            let p = *point.get(i, 0).unwrap();
            let min_i = *self.min.get(i, 0).unwrap();
            let max_i = *self.max.get(i, 0).unwrap();

            if p < min_i {
                *self.min.get_mut(i, 0).unwrap() = p;
            }
            if p > max_i {
                *self.max.get_mut(i, 0).unwrap() = p;
            }
        }
    }

    pub fn intersection(&self, other: &Self) -> Self {
        let mut new_min = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();
        let mut new_max = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();

        for i in 0..RANK {
            let val1 = *self.min.get(i, 0).unwrap();
            let val2 = *other.min.get(i, 0).unwrap();
            *new_min.get_mut(i, 0).unwrap() = if val1 > val2 { val1 } else { val2 };

            let val1 = *self.max.get(i, 0).unwrap();
            let val2 = *other.max.get(i, 0).unwrap();
            *new_max.get_mut(i, 0).unwrap() = if val1 < val2 { val1 } else { val2 };
        }

        // If intersection is empty (min > max in any dim), return standard empty box?
        // Or just return the mathematically computed box (which will be invalid/empty).
        // Let's check validity? users can check is_empty()

        Self {
            min: new_min,
            max: new_max,
        }
    }

    pub fn is_empty(&self) -> bool {
        for i in 0..RANK {
            if *self.min.get(i, 0).unwrap() > *self.max.get(i, 0).unwrap() {
                return true;
            }
        }
        false
    }

    /// Squared distance from this box to a point.
    pub fn squared_distance(&self, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> T {
        let mut dist_sq = T::default();
        for i in 0..RANK {
            let p = *point.get(i, 0).unwrap();
            let min = *self.min.get(i, 0).unwrap();
            let max = *self.max.get(i, 0).unwrap();

            if p < min {
                let d = min - p;
                dist_sq += d * d;
            } else if p > max {
                let d = p - max;
                dist_sq += d * d;
            }
        }
        dist_sq
    }

    /// Ray intersection parameter range (t_min, t_max).
    /// Uses Slab method.
    pub fn intersection_parameter(&self, ray: &Ray<T, RANK>) -> Option<(T, T)> {
        let mut t_min = T::from_f64(f64::MIN); // Or 0.0? Ray starts at 0.0 usually.
        let mut t_max = T::from_f64(f64::MAX);

        // Ray: o + t*d
        let origin = ray.origin();
        let dir = ray.direction();

        for i in 0..RANK {
            let o = *origin.get(i, 0).unwrap();
            let d = *dir.get(i, 0).unwrap();
            let min = *self.min.get(i, 0).unwrap();
            let max = *self.max.get(i, 0).unwrap();

            if d.abs() < T::epsilon() {
                // Ray is parallel to slab.
                if o < min || o > max {
                    return None;
                }
            } else {
                let inv_d = T::from_f64(1.0) / d;
                let mut t1 = (min - o) * inv_d;
                let mut t2 = (max - o) * inv_d;

                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }

                if t1 > t_min {
                    t_min = t1;
                }
                if t2 < t_max {
                    t_max = t2;
                }

                if t_min > t_max {
                    return None;
                }
            }
        }

        // t_max should be >= 0 for intersection with Ray (half-line)?
        // Let's refine based on typical ray casting. t >= 0.
        // If t_max < 0, box is behind ray.
        if t_max < T::default() {
            return None;
        }

        // Clamp t_min to 0 if it was negative (origin inside box)
        if t_min < T::default() {
            t_min = T::default();
        }

        Some((t_min, t_max))
    }

    pub fn intersects(&self, ray: &Ray<T, RANK>) -> bool {
        self.intersection_parameter(ray).is_some()
    }
}
