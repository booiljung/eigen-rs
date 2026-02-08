//! Geometry module entry point.

pub mod quaternion;
pub mod angle_axis;
pub mod transform;
pub mod aligned_box;
pub mod hyperplane;
pub mod parametrized_line;
pub mod ray;
pub mod translation;
pub mod scaling;
pub mod euler_angles;

pub use quaternion::Quaternion;
pub use angle_axis::AngleAxis;
pub use transform::{Transform, Transform2, Transform3, Projective2, Projective3};
pub use aligned_box::AlignedBox;
pub use hyperplane::Hyperplane;
pub use parametrized_line::ParametrizedLine;
pub use ray::Ray;
pub use translation::Translation;
pub use scaling::Scaling;
pub use euler_angles::EulerAngles;

pub type Translation2<T> = translation::Translation<T, 2>;
pub type Translation3<T> = translation::Translation<T, 3>;
pub type Affine2<T> = transform::Transform2<T>;
pub type Affine3<T> = transform::Transform3<T>;
pub type Isometry2<T> = transform::Transform2<T>;
pub type Isometry3<T> = transform::Transform3<T>;
