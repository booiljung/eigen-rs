use crate::core::scalar::Scalar;
use crate::core::geometry::Quaternion;

/// Represents a rotation using three Euler angles and an axis convention.
/// Currently supports ZYX (Yaw-Pitch-Roll) convention.
#[derive(Clone, Debug, PartialEq)]
pub struct EulerAngles<T: Scalar> {
    pub alpha: T, // Around Z (Yaw)
    pub beta: T,  // Around Y (Pitch)
    pub gamma: T, // Around X (Roll)
}

impl<T: Scalar> EulerAngles<T> {
    /// Creates a new EulerAngles (ZYX convention).
    /// alpha: angle around Z
    /// beta: angle around Y
    /// gamma: angle around X
    pub fn new(alpha: T, beta: T, gamma: T) -> Self {
        Self { alpha, beta, gamma }
    }

    /// Converts Euler angles to a Quaternion (ZYX convention).
    pub fn to_quaternion(&self) -> Quaternion<T> {
        // Q = Qz * Qy * Qx
        let half = T::from_f64(0.5);
        
        let c1 = (self.alpha * half).cos();
        let s1 = (self.alpha * half).sin();
        let c2 = (self.beta * half).cos();
        let s2 = (self.beta * half).sin();
        let c3 = (self.gamma * half).cos();
        let s3 = (self.gamma * half).sin();
        
        // ZYX order
        Quaternion::new(
            c1 * c2 * s3 - s1 * s2 * c3, // x
            c1 * s2 * c3 + s1 * c2 * s3, // y
            s1 * c2 * c3 - c1 * s2 * s3, // z
            c1 * c2 * c3 + s1 * s2 * s3  // w
        )
    }

    /// Converts a Rotation Matrix to EulerAngles (ZYX convention).
    pub fn from_rotation_matrix(mat: &crate::core::matrix::Matrix3<T>) -> Self {
        // Assuming ZYX (Yaw-Pitch-Roll)
        // R = Rz(a) * Ry(b) * Rx(c)
        // R31 = -sin(b)
        // R32 = cos(b)sin(c)
        // R33 = cos(b)cos(c)
        // R11 = cos(a)cos(b)
        // R21 = sin(a)cos(b)
        
        let zero = T::default();
        let one = T::from_f64(1.0);
        
        let r31 = *mat.get(2, 0).unwrap();
        let r32 = *mat.get(2, 1).unwrap();
        let r33 = *mat.get(2, 2).unwrap();
        let r21 = *mat.get(1, 0).unwrap();
        let r11 = *mat.get(0, 0).unwrap();
        
        // Pitch (beta), Yaw (alpha), Roll (gamma)
        let beta;
        let alpha;
        let gamma;
        
        if r31.abs() < one - T::epsilon() {
             // Standard case
             beta = -r31.asin();
             alpha = r21.atan2(r11);
             gamma = r32.atan2(r33);
        } else {
             // Gimbal lock
             alpha = zero;
             
             let r12 = *mat.get(0, 1).unwrap();
             let r22 = *mat.get(1, 1).unwrap();
             
             if r31 < zero {
                 // beta = pi/2
                 // asin(1) = pi/2.
                 beta = T::from_f64(std::f64::consts::FRAC_PI_2); 
                 gamma = alpha + r12.atan2(r22);
             } else {
                 // beta = -pi/2
                 beta = T::from_f64(-std::f64::consts::FRAC_PI_2);
                 gamma = -alpha + (-r12).atan2(-r22);
             }
        }
        
        Self::new(alpha, beta, gamma)
    }
}
