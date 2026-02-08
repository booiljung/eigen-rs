use crate::core::matrix::Vector4;
use crate::core::scalar::Scalar;

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

impl<T: Scalar> Quaternion<T> {
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

impl<T: Scalar> std::ops::Mul for Quaternion<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        self.hamilton_product(&rhs)
    }
}

impl<T: Scalar> std::ops::Mul<&Quaternion<T>> for Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: &Quaternion<T>) -> Self::Output {
        self.hamilton_product(rhs)
    }
}

impl<T: Scalar> std::ops::Mul<Quaternion<T>> for &Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: Quaternion<T>) -> Self::Output {
        self.hamilton_product(&rhs)
    }
}

impl<T: Scalar> std::ops::Mul<&Quaternion<T>> for &Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: &Quaternion<T>) -> Self::Output {
        self.hamilton_product(rhs)
    }
}

// Q * Vector3
impl<T: Scalar> std::ops::Mul<crate::core::matrix::Vector3<T>> for Quaternion<T> {
    type Output = crate::core::matrix::Vector3<T>;
    fn mul(self, rhs: crate::core::matrix::Vector3<T>) -> Self::Output {
        self.rotate_vector(&rhs)
    }
}

impl<T: Scalar> std::ops::Mul<&crate::core::matrix::Vector3<T>> for Quaternion<T> {
    type Output = crate::core::matrix::Vector3<T>;
    fn mul(self, rhs: &crate::core::matrix::Vector3<T>) -> Self::Output {
        self.rotate_vector(rhs)
    }
}

impl<T: Scalar> std::ops::Mul<crate::core::matrix::Vector3<T>> for &Quaternion<T> {
    type Output = crate::core::matrix::Vector3<T>;
    fn mul(self, rhs: crate::core::matrix::Vector3<T>) -> Self::Output {
        self.rotate_vector(&rhs)
    }
}

impl<T: Scalar> std::ops::Mul<&crate::core::matrix::Vector3<T>> for &Quaternion<T> {
    type Output = crate::core::matrix::Vector3<T>;
    fn mul(self, rhs: &crate::core::matrix::Vector3<T>) -> Self::Output {
        self.rotate_vector(rhs)
    }
}
