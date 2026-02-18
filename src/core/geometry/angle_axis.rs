use crate::core::geometry::quaternion::Quaternion;
use crate::core::matrix::Vector3;
use crate::core::scalar::Scalar;

/// Represents a rotation defined by an axis and an angle.
pub struct AngleAxis<T: Scalar> {
    axis: Vector3<T>,
    angle: T,
}

impl<T: Scalar<Real = T> + PartialOrd> AngleAxis<T> {
    /// Creates a new AngleAxis from an angle and a normalized axis.
    pub fn new(angle: T, axis: Vector3<T>) -> Self {
        Self { axis, angle }
    }

    pub fn axis(&self) -> &Vector3<T> {
        &self.axis
    }
    pub fn angle(&self) -> T {
        self.angle
    }

    /// Converts this rotation to a Quaternion.
    pub fn to_quaternion(&self) -> Quaternion<T> {
        let half_angle = self.angle / T::from_usize(2);
        let s = half_angle.sin();
        let c = half_angle.cos();

        Quaternion::new(
            *self.axis.get(0, 0).unwrap() * s,
            *self.axis.get(1, 0).unwrap() * s,
            *self.axis.get(2, 0).unwrap() * s,
            c,
        )
    }

    /// Converts this rotation to a 3x3 rotation matrix.
    pub fn to_rotation_matrix(&self) -> crate::core::matrix::Matrix3<T> {
        self.to_quaternion().to_rotation_matrix()
    }
}
