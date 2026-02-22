use crate::core::matrix::Vector4;
use crate::core::scalar::Scalar;
use crate::core::storage::Storage;

/// Represents a rotation in 3D space as a Quaternion (x, y, z, w).
#[derive(Debug)]
pub struct Quaternion<T: Scalar> {
    coeffs: Vector4<T>,
}

impl<T: Scalar> Clone for Quaternion<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Scalar> Copy for Quaternion<T> {}

impl<T: Scalar<Real = T> + PartialOrd> Quaternion<T> {
    /// Creates a new Quaternion from components.
    pub fn new(x: T, y: T, z: T, w: T) -> Self {
        let mut coeffs = Vector4::<T>::new_fixed();
        *coeffs.get_mut(0, 0).unwrap() = x;
        *coeffs.get_mut(1, 0).unwrap() = y;
        *coeffs.get_mut(2, 0).unwrap() = z;
        *coeffs.get_mut(3, 0).unwrap() = w;
        Self { coeffs }
    }

    /// Identity Quaternion (0, 0, 0, 1).
    pub fn identity() -> Self {
        Self::new(
            T::from_usize(0),
            T::from_usize(0),
            T::from_usize(0),
            T::from_usize(1),
        )
    }

    /// Creates a Quaternion from an angle and axis.
    pub fn from_angle_axis(angle: T, axis: crate::core::matrix::Vector3<T>) -> Self {
        let half_angle = angle / T::from_usize(2);
        let s = half_angle.sin();
        let c = half_angle.cos();
        Self::new(
            *axis.get(0, 0).unwrap() * s,
            *axis.get(1, 0).unwrap() * s,
            *axis.get(2, 0).unwrap() * s,
            c,
        )
    }

    pub fn x(&self) -> T {
        *self.coeffs.get(0, 0).unwrap()
    }
    pub fn y(&self) -> T {
        *self.coeffs.get(1, 0).unwrap()
    }
    pub fn z(&self) -> T {
        *self.coeffs.get(2, 0).unwrap()
    }
    pub fn w(&self) -> T {
        *self.coeffs.get(3, 0).unwrap()
    }

    pub fn coeffs(&self) -> &Vector4<T> {
        &self.coeffs
    }

    /// Returns the squared norm of the quaternion.
    pub fn norm_sq(&self) -> T {
        let mut sum = T::default();
        for i in 0..4 {
            let val = *self.coeffs.get(i, 0).unwrap();
            sum += val * val;
        }
        sum
    }

    /// Returns the norm of the quaternion.
    pub fn norm(&self) -> T {
        self.norm_sq().sqrt()
    }

    /// Normalizes the quaternion in-place.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n != T::default() {
            let inv_n = T::from_usize(1) / n;
            for i in 0..4 {
                *self.coeffs.get_mut(i, 0).unwrap() *= inv_n;
            }
        }
    }

    /// Returns the conjugate of the quaternion (-x, -y, -z, w).
    pub fn conjugate(&self) -> Self {
        let zero = T::default();
        Self::new(zero - self.x(), zero - self.y(), zero - self.z(), self.w())
    }

    /// Multiplication of two quaternions (Hamilton product).
    pub fn hamilton_product(&self, other: &Self) -> Self {
        let x1 = self.x();
        let y1 = self.y();
        let z1 = self.z();
        let w1 = self.w();
        let x2 = other.x();
        let y2 = other.y();
        let z2 = other.z();
        let w2 = other.w();

        Self::new(
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    }

    /// Rotates a 3D vector by this quaternion.
    pub fn rotate_vector(
        &self,
        v: &crate::core::matrix::Vector3<T>,
    ) -> crate::core::matrix::Vector3<T> {
        let q_v = Self::new(
            *v.get(0, 0).unwrap(),
            *v.get(1, 0).unwrap(),
            *v.get(2, 0).unwrap(),
            T::default(),
        );
        let res_q = self
            .hamilton_product(&q_v)
            .hamilton_product(&self.conjugate());

        let mut res = crate::core::matrix::Vector3::<T>::new_fixed();
        *res.get_mut(0, 0).unwrap() = res_q.x();
        *res.get_mut(1, 0).unwrap() = res_q.y();
        *res.get_mut(2, 0).unwrap() = res_q.z();
        res
    }

    /// Converts the quaternion to a 3x3 rotation matrix.
    pub fn to_rotation_matrix(&self) -> crate::core::matrix::Matrix3<T> {
        let mut res = crate::core::matrix::Matrix3::<T>::new_fixed();
        let x = self.x();
        let y = self.y();
        let z = self.z();
        let w = self.w();

        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        let one = T::from_usize(1);

        *res.get_mut(0, 0).unwrap() = one - (yy + zz);
        *res.get_mut(0, 1).unwrap() = xy - wz;
        *res.get_mut(0, 2).unwrap() = xz + wy;

        *res.get_mut(1, 0).unwrap() = xy + wz;
        *res.get_mut(1, 1).unwrap() = one - (xx + zz);
        *res.get_mut(1, 2).unwrap() = yz - wx;

        *res.get_mut(2, 0).unwrap() = xz - wy;
        *res.get_mut(2, 1).unwrap() = yz + wx;
        *res.get_mut(2, 2).unwrap() = one - (xx + yy);

        res
    }

    /// Spherical linear interpolation between two quaternions.
    pub fn slerp(&self, t: T, other: &Self) -> Self {
        let mut dot = self.x() * other.x()
            + self.y() * other.y()
            + self.z() * other.z()
            + self.w() * other.w();

        let mut v1 = *other;
        if dot < T::default() {
            dot = T::default() - dot;
            v1 = Self::new(
                T::default() - other.x(),
                T::default() - other.y(),
                T::default() - other.z(),
                T::default() - other.w(),
            );
        }

        if dot > T::from_usize(1) - T::epsilon() {
            // Linear interpolation for very close quaternions
            let one_minus_t = T::from_usize(1) - t;
            let mut res = Self::new(
                one_minus_t * self.x() + t * v1.x(),
                one_minus_t * self.y() + t * v1.y(),
                one_minus_t * self.z() + t * v1.z(),
                one_minus_t * self.w() + t * v1.w(),
            );
            res.normalize();
            return res;
        }

        let theta_0 = Scalar::acos(dot);
        let theta = theta_0 * t;
        let sin_theta = Scalar::sin(theta);
        let sin_theta_0 = Scalar::sin(theta_0);

        let s0 = Scalar::sin(theta_0 - theta) / sin_theta_0;
        let s_t = sin_theta / sin_theta_0;

        Self::new(
            s0 * self.x() + s_t * v1.x(),
            s0 * self.y() + s_t * v1.y(),
            s0 * self.z() + s_t * v1.z(),
            s0 * self.w() + s_t * v1.w(),
        )
    }
}

// SIMD Implementation for f32
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl Quaternion<f32> {
    /// Vectorized Hamilton Product for f32 using AVX.
    #[target_feature(enable = "avx,fma")]
    pub unsafe fn hamilton_product_simd(&self, other: &Self) -> Self {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        // Load q1 and q2 into registers
        // q1 = [x1, y1, z1, w1]
        // q2 = [x2, y2, z2, w2]
        // Storage is Column Major in Vector4, so data is contiguous in memory locally
        let q1_ptr = self.coeffs.storage().data().as_ptr() as *const f32;
        let q2_ptr = other.coeffs.storage().data().as_ptr() as *const f32;

        let q1 = _mm_loadu_ps(q1_ptr);
        let q2 = _mm_loadu_ps(q2_ptr);

        // Reference Algorithm:
        // result.x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        // result.y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        // result.z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        // result.w = w1*w2 - x1*x2 - y1*y2 - z1*z2

        // Optimized SSE/AVX shuffle method:
        // t1 = q1.ymzw * q2.zwyx = [y1*z2, z1*w2, w1*y2, x1*x2] ?? check shuffle masks
        // Need careful permutation.
        // Let's use a simpler permute approach suitable for SSE.

        // q1_x = [x1, x1, x1, x1]
        let q1_x = _mm_permute_ps(q1, 0x00);
        // q1_y = [y1, y1, y1, y1]
        let q1_y = _mm_permute_ps(q1, 0x55);
        // q1_z = [z1, z1, z1, z1]
        let q1_z = _mm_permute_ps(q1, 0xAA);
        // q1_w = [w1, w1, w1, w1]
        let q1_w = _mm_permute_ps(q1, 0xFF);

        // q2 = [x2, y2, z2, w2]

        // Term 1: w1 * q2 = [w1x2, w1y2, w1z2, w1w2]
        let t1 = _mm_mul_ps(q1_w, q2);

        // Term 2: x1 * q2_perm1
        // For x: + x1*w2. For y: - x1*z2. For z: + x1*y2. For w: - x1*x2.
        // Target: .w .z .y .x -> 3 2 1 0 => [w2, z2, y2, x2]
        // Swizzle q2 for term 2: [w2, z2, y2, x2] (0x1B: 00 01 10 11 = 0 1 2 3 reversed? No. 3,2,1,0)
        let q2_swz1 = _mm_permute_ps(q2, 0x1B); // [w2, z2, y2, x2]
        let mut t2 = _mm_mul_ps(q1_x, q2_swz1);
        // Correct signs for t2: [+ + - -] -> mul by [1, -1, 1, -1]? No wait.
        // Formula:
        // x: + x1*w2 (Correct, index 0 of swz1 is w2)
        // y: - x1*z2 (Correct, index 1 of swz1 is z2, need neg)
        // z: + x1*y2 (Correct, index 2 of swz1 is y2)
        // w: - x1*x2 (Correct, index 3 of swz1 is x2, need neg)
        // Sign Mask: [1.0, -1.0, 1.0, -1.0]
        let sign_mask1 = _mm_set_ps(-1.0, 1.0, -1.0, 1.0); // Little Endian: w, z, y, x
        t2 = _mm_mul_ps(t2, sign_mask1);

        // Term 3: y1 * q2_perm2
        // For x: + y1*z2. For y: + y1*w2. For z: - y1*x2. For w: - y1*y2.
        // Target q2 elements: [z2, w2, x2, y2]
        // Permute q2: [x2, y2, z2, w2] -> [z2, w2, x2, y2] (Mask: 3 0 1 2 ? No. 2,3,0,1 => 10 11 00 01 = 0xB1? Wait. _MM_SHUFFLE(z,y,x,w) selects indices.
        // We want indices 2, 3, 0, 1. _MM_SHUFFLE(1, 0, 3, 2) = 01 00 11 10 = 0x4E.
        let q2_swz2 = _mm_permute_ps(q2, 0x4E);
        let mut t3 = _mm_mul_ps(q1_y, q2_swz2);
        // Signs: [+ + - -] => x(+), y(+), z(-), w(-)
        // Mask: [-1.0, -1.0, 1.0, 1.0]
        let sign_mask2 = _mm_set_ps(-1.0, -1.0, 1.0, 1.0);
        t3 = _mm_mul_ps(t3, sign_mask2);

        // Term 4: z1 * q2_perm3
        // For x: - z1*y2. For y: + z1*x2. For z: + z1*w2. For w: - z1*z2.
        // Target q2 elements: [y2, x2, w2, z2]
        // Permute q2: [x2, y2, z2, w2] -> [y2, x2, w2, z2].
        // Indices: 1, 0, 3, 2. _MM_SHUFFLE(2, 3, 0, 1) = 10 11 00 01 = 0xB1.
        let q2_swz3 = _mm_permute_ps(q2, 0xB1);
        let mut t4 = _mm_mul_ps(q1_z, q2_swz3);
        // Signs: [- + + -] => x(-), y(+), z(+), w(-)
        // Mask: [-1.0, 1.0, 1.0, -1.0]
        let sign_mask3 = _mm_set_ps(-1.0, 1.0, 1.0, -1.0);
        t4 = _mm_mul_ps(t4, sign_mask3);

        // Sum everything
        let sum = _mm_add_ps(_mm_add_ps(t1, t2), _mm_add_ps(t3, t4));

        let mut res = Self::new(0.0, 0.0, 0.0, 0.0);
        let res_ptr = res.coeffs.storage_mut().data_mut().as_mut_ptr() as *mut f32;
        _mm_storeu_ps(res_ptr, sum);
        res
    }

    /// Vectorized Rotate Vector for f32 using AVX.
    #[target_feature(enable = "avx,fma")]
    pub unsafe fn rotate_vector_simd(
        &self,
        v: &crate::core::matrix::Vector3<f32>,
    ) -> crate::core::matrix::Vector3<f32> {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        // Formula: v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)

        let q_ptr = self.coeffs.storage().data().as_ptr() as *const f32;
        let v_ptr = v.storage().data().as_ptr() as *const f32;

        let q = _mm_loadu_ps(q_ptr); // [x, y, z, w]
        let _v_vec = _mm_maskload_ps(v_ptr, _mm_set_epi32(0, -1, -1, -1)); // [vx, vy, vz, 0] safely? Or simplified load.
                                                                           // Use a safe load or just loadu if we verify buffer size (Vector3 has padding? DynamicStorage might not).
                                                                           // For Vector3 FixedStorage, it has size 3. Reading 4th float is unsafe if it's the end of page.
                                                                           // Assuming user provides valid vector, usually safe to read 4 bytes if aligned, but let's be careful.
                                                                           // To be safe without maskload (which is slow), we can construct manually or assume padding.
                                                                           // Let's use `_mm_set_ps` for safety since Vector3 is small.
        let v_vec_safe = _mm_set_ps(
            0.0,
            *v.get(2, 0).unwrap(),
            *v.get(1, 0).unwrap(),
            *v.get(0, 0).unwrap(),
        );

        let w = _mm_permute_ps(q, 0xFF); // [w, w, w, w]
        let two = _mm_set1_ps(2.0);

        // Cross Product Helper: a x b = [ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx]
        // Utilizing shuffles.
        let cross = |a: __m128, b: __m128| {
            let _a_yzx = _mm_permute_ps(a, 0xC9); // 3 0 2 1 => 11 00 10 01 (w, z, y, x)? No.
                                                  // _MM_SHUFFLE(3, 0, 2, 1) = 11 00 10 01. Indices: 1, 2, 0, 3 (w remains at 3).
                                                  // Desired: [y, z, x, _]

            // Reference Macro: _MM_SHUFFLE(3, 0, 2, 1) -> Indices 1, 2, 0, 3
            // _MM_SHUFFLE(3, 0, 2, 1) = (3<<6)|(0<<4)|(2<<2)|1 = 0xC9
            let a_shuf1 = _mm_permute_ps(a, 0xC9);
            // _MM_SHUFFLE(3, 1, 0, 2) = (3<<6)|(1<<4)|(0<<2)|2 = 0xD2
            let b_shuf1 = _mm_permute_ps(b, 0xD2); // 2, 0, 1, 3

            let mul1 = _mm_mul_ps(a_shuf1, b_shuf1);

            let a_shuf2 = _mm_permute_ps(a, 0xD2);
            let b_shuf2 = _mm_permute_ps(b, 0xC9);

            let mul2 = _mm_mul_ps(a_shuf2, b_shuf2);

            _mm_sub_ps(mul1, mul2)
        };

        // q_w * v
        let wv = _mm_mul_ps(w, v_vec_safe);

        // cross(q_xyz, v)
        let cx = cross(q, v_vec_safe);

        // cross(q_xyz, v) + q_w * v
        let sum = _mm_add_ps(cx, wv);

        // cross(q_xyz, sum)
        let cx2 = cross(q, sum);

        // 2 * cx2
        let scaled = _mm_mul_ps(two, cx2);

        // v + scaled
        let res_vec = _mm_add_ps(v_vec_safe, scaled);

        let mut res = crate::core::matrix::Vector3::<f32>::new_fixed();
        // Store only 3 floats
        let res_arr: [f32; 4] = std::mem::transmute(res_vec);
        *res.get_mut(0, 0).unwrap() = res_arr[0];
        *res.get_mut(1, 0).unwrap() = res_arr[1];
        *res.get_mut(2, 0).unwrap() = res_arr[2];
        res
    }
}

impl<T: Scalar<Real = T> + PartialOrd> std::ops::Mul for Quaternion<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                && is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
            {
                unsafe {
                    let lhs_f32: &Quaternion<f32> =
                        &*(&self as *const Quaternion<T> as *const Quaternion<f32>);
                    let rhs_f32: &Quaternion<f32> =
                        &*(&rhs as *const Quaternion<T> as *const Quaternion<f32>);
                    let res = lhs_f32.hamilton_product_simd(rhs_f32);
                    return *(&res as *const Quaternion<f32> as *const Quaternion<T>);
                }
            }
        }
        self.hamilton_product(&rhs)
    }
}

impl<T: Scalar<Real = T> + PartialOrd> std::ops::Mul<&Quaternion<T>> for Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: &Quaternion<T>) -> Self::Output {
        self.hamilton_product(rhs)
    }
}

impl<T: Scalar<Real = T> + PartialOrd> std::ops::Mul<Quaternion<T>> for &Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        self.hamilton_product(&rhs)
    }
}

impl<T: Scalar<Real = T> + PartialOrd> std::ops::Mul<&Quaternion<T>> for &Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: &Quaternion<T>) -> Self::Output {
        self.hamilton_product(rhs)
    }
}

// Q * Vector3
impl<T: Scalar<Real = T> + PartialOrd> std::ops::Mul<crate::core::matrix::Vector3<T>>
    for Quaternion<T>
{
    type Output = crate::core::matrix::Vector3<T>;
    fn mul(self, rhs: crate::core::matrix::Vector3<T>) -> Self::Output {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                && is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
            {
                unsafe {
                    let lhs_f32: &Quaternion<f32> =
                        &*(&self as *const Quaternion<T> as *const Quaternion<f32>);
                    let rhs_f32: &crate::core::matrix::Vector3<f32> = &*(&rhs
                        as *const crate::core::matrix::Vector3<T>
                        as *const crate::core::matrix::Vector3<f32>);
                    let res = lhs_f32.rotate_vector_simd(rhs_f32);
                    return *(&res as *const crate::core::matrix::Vector3<f32>
                        as *const crate::core::matrix::Vector3<T>);
                }
            }
        }
        self.rotate_vector(&rhs)
    }
}

impl<T: Scalar<Real = T> + PartialOrd> std::ops::Mul<&crate::core::matrix::Vector3<T>>
    for Quaternion<T>
{
    type Output = crate::core::matrix::Vector3<T>;
    fn mul(self, rhs: &crate::core::matrix::Vector3<T>) -> Self::Output {
        self.rotate_vector(rhs)
    }
}

impl<T: Scalar<Real = T> + PartialOrd> std::ops::Mul<crate::core::matrix::Vector3<T>>
    for &Quaternion<T>
{
    type Output = crate::core::matrix::Vector3<T>;
    fn mul(self, rhs: crate::core::matrix::Vector3<T>) -> Self::Output {
        self.rotate_vector(&rhs)
    }
}

impl<T: Scalar<Real = T> + PartialOrd> std::ops::Mul<&crate::core::matrix::Vector3<T>>
    for &Quaternion<T>
{
    type Output = crate::core::matrix::Vector3<T>;
    fn mul(self, rhs: &crate::core::matrix::Vector3<T>) -> Self::Output {
        self.rotate_vector(rhs)
    }
}
