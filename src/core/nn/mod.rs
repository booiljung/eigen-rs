//! Neural Network module for eigen-rs.

use crate::core::scalar::Scalar;
use crate::core::tensor::Tensor;

pub mod layers;

/// Trait representing a neural network layer.
pub trait Layer<T: Scalar> {
    /// Forward pass through the layer.
    fn forward(&self, input: &Tensor<T, 2>) -> Result<Tensor<T, 2>, String>;
}
