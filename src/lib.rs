#![cfg_attr(not(test), no_std)]
#![doc = include_str!("../README.md")]

pub mod core;

pub use crate::core::matrix::{Matrix, Matrix2, Matrix3, Matrix4, MatrixX, Map};
pub use crate::core::storage::DYNAMIC;

pub use crate::core::decompositions;
pub use crate::core::geometry;
pub use crate::core::sparse;
pub use crate::core::scalar::Scalar;

pub mod unsupported;
