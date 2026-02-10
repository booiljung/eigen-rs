//! Linear (Fully Connected) layer.

use crate::core::matrix::MatrixX;
use crate::core::nn::Layer;
use crate::core::scalar::Scalar;
use crate::core::tensor::Tensor;
// use crate::core::storage::Storage;
// use crate::core::xpr::MatrixXpr;

/// A Linear layer (Y = X * W^T + b).
pub struct Linear<T: Scalar> {
    weight: Tensor<T, 2>,
    bias: Tensor<T, 1>,
}

impl<T: Scalar> Linear<T> {
    pub fn new(in_features: usize, out_features: usize) -> Result<Self, String> {
        Ok(Self {
            weight: Tensor::new([out_features, in_features])?,
            bias: Tensor::new([out_features])?,
        })
    }

    pub fn weight(&self) -> &Tensor<T, 2> {
        &self.weight
    }

    pub fn weight_mut(&mut self) -> &mut Tensor<T, 2> {
        &mut self.weight
    }

    pub fn bias(&self) -> &Tensor<T, 1> {
        &self.bias
    }

    pub fn bias_mut(&mut self) -> &mut Tensor<T, 1> {
        &mut self.bias
    }
}

impl<T: Scalar + 'static> Layer<T> for Linear<T> {
    fn forward(&self, input: &Tensor<T, 2>) -> Result<Tensor<T, 2>, String> {
        let x_mat = input.to_matrix()?;
        let w_mat = self.weight.to_matrix()?;

        let m = x_mat.rows();
        let n = w_mat.rows(); // out_features

        let mut y_mat = MatrixX::<T>::new_dynamic(m, n)?;

        // Y = X * W^T
        let w_t = w_mat.transpose();
        let prod = &x_mat * &w_t;

        y_mat.assign(&prod)?;

        // Add bias (broadcasting across batch)
        let bias_data = self.bias.data();
        for i in 0..m {
            for (j, val) in bias_data.iter().enumerate().take(n) {
                if let Some(res) = y_mat.get_mut(i, j) {
                    *res += *val;
                }
            }
        }

        Ok(Tensor::from_matrix(y_mat))
    }
}
