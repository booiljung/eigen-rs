use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::{FixedStorage, Storage};

/// Represents an affine transformation in space.
/// Internally stored as a NxN matrix (where N = DIM + 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform<T: Scalar, const N: usize, const DIM: usize, const SIZE: usize> {
    matrix: Matrix<T, FixedStorage<T, N, N, SIZE>>,
}

impl<T: Scalar<Real = T> + PartialOrd, const N: usize, const DIM: usize, const SIZE: usize> Transform<T, N, DIM, SIZE> {
    /// Creates an identity transformation.
    pub fn identity() -> Self {
        let mut matrix = Matrix::<T, FixedStorage<T, N, N, SIZE>>::new_fixed();
        for i in 0..N {
            for j in 0..N {
                *matrix.get_mut(i, j).unwrap() = if i == j {
                    T::from_usize(1)
                } else {
                    T::default()
                };
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() &&
               N == 4 && DIM == 3 && SIZE == 16 &&
               is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                unsafe {
                    let lhs_f32: &Transform<f32, 4, 3, 16> = &*(self as *const Transform<T, N, DIM, SIZE> as *const Transform<f32, 4, 3, 16>);
                    let rhs_f32: &Transform<f32, 4, 3, 16> = &*(other as *const Transform<T, N, DIM, SIZE> as *const Transform<f32, 4, 3, 16>);
                    let res = lhs_f32.mul_simd(rhs_f32);
                    return *( &res as *const Transform<f32, 4, 3, 16> as *const Transform<T, N, DIM, SIZE> );
                }
            }
        }

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
    pub fn transform_point<const VSIZE: usize>(
        &self,
        p: &Matrix<T, FixedStorage<T, DIM, 1, VSIZE>>,
    ) -> Matrix<T, FixedStorage<T, DIM, 1, VSIZE>> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() &&
               N == 4 && DIM == 3 && SIZE == 16 &&
               is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                unsafe {
                    // We need to check VSIZE constraint or transmute carefully.
                    // transform_point_simd is generic over VSIZE.
                    // But we need to ensure T is f32.
                    // Pointer cast is safer than transmute for generic constraints.
                     let lhs_f32: &Transform<f32, 4, 3, 16> = &*(self as *const Transform<T, N, DIM, SIZE> as *const Transform<f32, 4, 3, 16>);
                     let p_f32: &Matrix<f32, FixedStorage<f32, 3, 1, VSIZE>> = &*(p as *const Matrix<T, FixedStorage<T, DIM, 1, VSIZE>> as *const Matrix<f32, FixedStorage<f32, 3, 1, VSIZE>>);
                     let res = lhs_f32.transform_point_simd(p_f32);
                     return *( &res as *const Matrix<f32, FixedStorage<f32, 3, 1, VSIZE>> as *const Matrix<T, FixedStorage<T, DIM, 1, VSIZE>> );
                }
            }
        }

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
impl<T: Scalar<Real = T> + PartialOrd> Transform<T, 4, 3, 16> {
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

    pub fn from_parts(
        t: crate::core::geometry::Translation<T, 3>,
        r: crate::core::geometry::Quaternion<T>,
    ) -> Self {
        let tf_t = Self::from_translation(&t);
        let tf_r = Self::from_rotation(&r);
        tf_t.mul(&tf_r) // Translation * Rotation
    }
}

// SIMD Implementation for Transform 3D (4x4) f32
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Transform<f32, 4, 3, 16> {
    /// Vectorized Matrix Multiplication (Self * Other) for 4x4 f32 using AVX.
    #[target_feature(enable = "avx,fma")]
    pub unsafe fn mul_simd(&self, other: &Self) -> Self {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let a_ptr = self.matrix.storage().data().as_ptr() as *const f32;
        let b_ptr = other.matrix.storage().data().as_ptr() as *const f32;
        let mut res = Self::identity();
        let c_ptr = res.matrix.storage_mut().data_mut().as_mut_ptr() as *mut f32;

        let a_col0 = _mm_loadu_ps(a_ptr.add(0));
        let a_col1 = _mm_loadu_ps(a_ptr.add(4));
        let a_col2 = _mm_loadu_ps(a_ptr.add(8));
        let a_col3 = _mm_loadu_ps(a_ptr.add(12));

        for j in 0..4 {
             let b_col_ptr = b_ptr.add(j * 4);
             let b_col = _mm_loadu_ps(b_col_ptr);
             
             let b0 = _mm_permute_ps(b_col, 0x00);
             let b1 = _mm_permute_ps(b_col, 0x55);
             let b2 = _mm_permute_ps(b_col, 0xAA);
             let b3 = _mm_permute_ps(b_col, 0xFF);

             let mut sum = _mm_mul_ps(a_col0, b0);
             sum = _mm_fmadd_ps(a_col1, b1, sum);
             sum = _mm_fmadd_ps(a_col2, b2, sum);
             sum = _mm_fmadd_ps(a_col3, b3, sum);

             _mm_storeu_ps(c_ptr.add(j * 4), sum);
        }
        res
    }

    /// Vectorized Transform Point (Ax) for 3D point (4x1 with w=1) using AVX.
    #[target_feature(enable = "avx,fma")]
    pub unsafe fn transform_point_simd<const VSIZE: usize>(
        &self,
        p: &Matrix<f32, FixedStorage<f32, 3, 1, VSIZE>>,
    ) -> Matrix<f32, FixedStorage<f32, 3, 1, VSIZE>> {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        
        
        let a_ptr = self.matrix.storage().data().as_ptr() as *const f32;
        // p_ptr unused as we construct vector manually


        // Load p as [x, y, z, 1.0]
        let p_vec = _mm_set_ps(1.0, *p.get(2,0).unwrap(), *p.get(1,0).unwrap(), *p.get(0,0).unwrap()); 

        let x = _mm_permute_ps(p_vec, 0x00);
        let y = _mm_permute_ps(p_vec, 0x55);
        let z = _mm_permute_ps(p_vec, 0xAA);
        let w = _mm_permute_ps(p_vec, 0xFF);

        let col0 = _mm_loadu_ps(a_ptr.add(0));
        let col1 = _mm_loadu_ps(a_ptr.add(4));
        let col2 = _mm_loadu_ps(a_ptr.add(8));
        let col3 = _mm_loadu_ps(a_ptr.add(12));

        let mut sum = _mm_mul_ps(col0, x);
        sum = _mm_fmadd_ps(col1, y, sum);
        sum = _mm_fmadd_ps(col2, z, sum);
        sum = _mm_fmadd_ps(col3, w, sum);

        let mut res = Matrix::<f32, FixedStorage<f32, 3, 1, VSIZE>>::new_fixed();
        
        // Extract 3 elements
        let res_arr: [f32; 4] = std::mem::transmute(sum);
        *res.get_mut(0, 0).unwrap() = res_arr[0];
        *res.get_mut(1, 0).unwrap() = res_arr[1];
        *res.get_mut(2, 0).unwrap() = res_arr[2];
        
        res
    }
}

pub type Transform2<T> = Transform<T, 3, 2, 9>;
pub type Transform3<T> = Transform<T, 4, 3, 16>;

/// Projective transformation (Homogeneous coordinates).
/// For 3D, this is a 4x4 matrix, same as Transform3 but conceptually allows perspective (last row != [0,0,0,1]).
/// Examples: Projection matrices.
pub type Projective3<T> = Transform<T, 4, 3, 16>;
pub type Projective2<T> = Transform<T, 3, 2, 9>;
