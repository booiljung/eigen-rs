//! Geometry module entry point.

pub mod aligned_box;
pub mod angle_axis;
pub mod euler_angles;
pub mod hyperplane;
pub mod parametrized_line;
pub mod quaternion;
pub mod ray;
pub mod scaling;
pub mod transform;
pub mod translation;

pub use aligned_box::AlignedBox;
pub use angle_axis::AngleAxis;
pub use euler_angles::EulerAngles;
pub use hyperplane::Hyperplane;
pub use parametrized_line::ParametrizedLine;
pub use quaternion::Quaternion;
pub use ray::Ray;
pub use scaling::Scaling;
pub use transform::{Projective2, Projective3, Transform, Transform2, Transform3};
pub use translation::Translation;

pub type Translation2<T> = translation::Translation<T, 2>;
pub type Translation3<T> = translation::Translation<T, 3>;
pub type Affine2<T> = transform::Transform2<T>;
pub type Affine3<T> = transform::Transform3<T>;
pub type Isometry2<T> = transform::Transform2<T>;
pub type Isometry3<T> = transform::Transform3<T>;
