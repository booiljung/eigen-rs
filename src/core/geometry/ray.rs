use crate::core::geometry::hyperplane::Hyperplane;
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::FixedStorage;

/// A half-line, defined by an origin and a direction vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Ray<T: Scalar, const RANK: usize> {
    origin: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
    direction: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
}

impl<T: Scalar, const RANK: usize> Ray<T, RANK> {
    pub fn new(
        origin: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
        direction: Matrix<T, FixedStorage<T, RANK, 1, RANK>>,
    ) -> Self {
        Self { origin, direction }
    }

    pub fn point_at(&self, t: T) -> Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        // Ray is defined for t >= 0
        let t_valid = if t < T::default() { T::default() } else { t };
        let mut p = self.origin;
        for i in 0..RANK {
            *p.get_mut(i, 0).unwrap() += *self.direction.get(i, 0).unwrap() * t_valid;
        }
        p
    }

    pub fn distance(&self, point: &Matrix<T, FixedStorage<T, RANK, 1, RANK>>) -> T {
        // Project point onto line, clamp t >= 0
        let mut diff = Matrix::<T, FixedStorage<T, RANK, 1, RANK>>::zeros();
        for i in 0..RANK {
            *diff.get_mut(i, 0).unwrap() =
                *point.get(i, 0).unwrap() - *self.origin.get(i, 0).unwrap();
        }

        let mut num = T::default();
        let mut den = T::default();

        for i in 0..RANK {
            let d_i = *self.direction.get(i, 0).unwrap();
            num += d_i * *diff.get(i, 0).unwrap();
            den += d_i * d_i;
        }

        let t = num / den;
        let t_clamped = if t < T::default() { T::default() } else { t };

        let proj_point = self.point_at(t_clamped);

        let mut dist_sq = T::default();
        for i in 0..RANK {
            let delta = *point.get(i, 0).unwrap() - *proj_point.get(i, 0).unwrap();
            dist_sq += delta * delta;
        }
        dist_sq.sqrt()
    }

    pub fn intersects_plane(&self, plane: &Hyperplane<T, RANK>) -> Option<T> {
        // n . (o + t*d) + offset = 0
        // n . o + t * (n . d) + offset = 0
        // t = - (n . o + offset) / (n . d)
        // t = - signed_distance(origin) / (n . d)

        let dist = plane.signed_distance(&self.origin);
        let mut denom = T::default();
        let normal = plane.normal();

        for i in 0..RANK {
            denom += *normal.get(i, 0).unwrap() * *self.direction.get(i, 0).unwrap();
        }

        if denom.abs() < T::epsilon() {
            // Ray parallel to plane
            return None;
        }

        let t = -dist / denom;

        if t >= T::default() {
            Some(t)
        } else {
            None
        }
    }

    pub fn origin(&self) -> &Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        &self.origin
    }

    pub fn direction(&self) -> &Matrix<T, FixedStorage<T, RANK, 1, RANK>> {
        &self.direction
    }
}
