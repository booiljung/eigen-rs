use crate::core::matrix::Matrix;
use crate::core::storage::FixedStorage;
use crate::core::scalar::Scalar;

/// Represents an affine transformation in space.
/// Internally stored as a NxN matrix (where N = DIM + 1).
pub struct Transform<T: Scalar, const N: usize, const DIM: usize, const SIZE: usize> {
    matrix: Matrix<T, FixedStorage<T, N, N, SIZE>>,
}

impl<T: Scalar, const N: usize, const DIM: usize, const SIZE: usize> Transform<T, N, DIM, SIZE> {
    /// Creates an identity transformation.
    pub fn identity() -> Self {
        let mut matrix = Matrix::<T, FixedStorage<T, N, N, SIZE>>::new_fixed();
        for i in 0..N {
            for j in 0..N {
                *matrix.get_mut(i, j).unwrap() = if i == j { T::from_usize(1) } else { T::default() };
            }
        }
        Self { matrix }
    }

    pub fn matrix(&self) -> &Matrix<T, FixedStorage<T, N, N, SIZE>> {
        &self.matrix
    }

    pub fn matrix_mut(&mut self) -> &mut Matrix<T, FixedStorage<T, N, N, SIZE>> {
        &mut self.matrix
    }

    /// Composes two transformations (self * other).
    pub fn mul(&self, other: &Self) -> Self {
        let mut res_matrix = Matrix::<T, FixedStorage<T, N, N, SIZE>>::new_fixed();
        for i in 0..N {
            for j in 0..N {
                let mut sum = T::default();
                for k in 0..N {
                    sum += (*self.matrix.get(i, k).unwrap()) * (*other.matrix.get(k, j).unwrap());
                }
                *res_matrix.get_mut(i, j).unwrap() = sum;
            }
        }
        Self { matrix: res_matrix }
    }

    /// Applies the transformation to a point (as a vector with implicit 1 in the last component).
    pub fn transform_point<const VSIZE: usize>(&self, p: &Matrix<T, FixedStorage<T, DIM, 1, VSIZE>>) -> Matrix<T, FixedStorage<T, DIM, 1, VSIZE>> {
        let mut res = Matrix::<T, FixedStorage<T, DIM, 1, VSIZE>>::new_fixed();
        for i in 0..DIM {
            let mut sum = *self.matrix.get(i, DIM).unwrap(); // Translation part
            for j in 0..DIM {
                sum += (*self.matrix.get(i, j).unwrap()) * (*p.get(j, 0).unwrap());
            }
            *res.get_mut(i, 0).unwrap() = sum;
        }
        res
    }

    pub fn from_translation(t: &crate::core::geometry::Translation<T, DIM>) -> Self {
        let mut tf = Self::identity();
        for i in 0..DIM {
            *tf.matrix.get_mut(i, DIM).unwrap() = *t.vector.get(i, 0).unwrap();
        }
        tf
    }

    pub fn from_scaling(s: &crate::core::geometry::Scaling<T, DIM>) -> Self {
        let mut tf = Self::identity();
        for i in 0..DIM {
            *tf.matrix.get_mut(i, i).unwrap() = *s.coeffs.get(i, 0).unwrap();
        }
        tf
    }
}

// Specialization for 3D rotation
impl<T: Scalar> Transform<T, 4, 3, 16> {
    pub fn from_rotation(q: &crate::core::geometry::Quaternion<T>) -> Self {
        let r = q.to_rotation_matrix();
        let mut tf = Self::identity();
        for i in 0..3 {
            for j in 0..3 {
                *tf.matrix.get_mut(i, j).unwrap() = *r.get(i, j).unwrap();
            }
        }
        tf
    }

    pub fn from_parts(t: crate::core::geometry::Translation<T, 3>, r: crate::core::geometry::Quaternion<T>) -> Self {
        let tf_t = Self::from_translation(&t);
        let tf_r = Self::from_rotation(&r);
        tf_t.mul(&tf_r) // Translation * Rotation
    }
}


pub type Transform2<T> = Transform<T, 3, 2, 9>;
pub type Transform3<T> = Transform<T, 4, 3, 16>;

/// Projective transformation (Homogeneous coordinates).
/// For 3D, this is a 4x4 matrix, same as Transform3 but conceptually allows perspective (last row != [0,0,0,1]).
/// Examples: Projection matrices.
pub type Projective3<T> = Transform<T, 4, 3, 16>;
pub type Projective2<T> = Transform<T, 3, 2, 9>;
