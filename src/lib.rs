#![cfg_attr(not(feature = "std"), no_std)]
#![doc = include_str!("../README.md")]

extern crate alloc;

pub mod core;

pub use crate::core::matrix::{Map, Matrix, Matrix2, Matrix3, Matrix4, MatrixX};
pub use crate::core::storage::DYNAMIC;

pub use crate::core::decompositions;
pub use crate::core::geometry;
pub use crate::core::scalar::Scalar;
pub use crate::core::sparse;

pub mod unsupported;
