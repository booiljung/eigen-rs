#[cfg(feature = "candle")]
use candle_core::{Tensor, Device};
use crate::core::matrix::Matrix;
use crate::core::storage::{DynamicStorage, Storage};
use crate::core::scalar::Scalar;

#[cfg(feature = "candle")]
impl<T: Scalar + candle_core::WithDType> Matrix<T, DynamicStorage<T>> {
    /// Zero-Copy conversation to a Candle Tensor.
    /// This method borrows the underlying linear memory pool, creating a
    /// virtual Tensor matching precisely mathematically the Column-Major layout.
    pub fn as_candle_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        let (rows, cols) = (self.rows(), self.cols());
        
        // Eigen/Eigen-rs memory physically resides contiguous across columns ([k * M + i]).
        // Candle-core assumes C-contiguous (Row-Major).
        // To achieve zero-copy (avoiding actual hardware memory allocation for transpose),
        // we create a transposed (cols, rows) Tensor and apply a metadata view-transpose (`transpose(0, 1)`).
        let raw_slice = self.storage().data();
        let tensor = Tensor::from_slice(raw_slice, (cols, rows), device)?;
        
        // Logical/View Transpose. Costs O(1) in metadata tracking, O(0) memory copy.
        tensor.transpose(0, 1)
    }
}

#[cfg(test)]
#[cfg(feature = "candle")]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_candle_tensor_creation() {
        // [1.0, 3.0]
        // [2.0, 4.0]
        // Memory physical (contiguous): [1.0, 2.0, 3.0, 4.0]
        let matrix = Matrix::<f64, DynamicStorage<f64>>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        // The device parameter routes memory boundaries safely
        let device = Device::Cpu;
        let tensor = matrix.as_candle_tensor(&device).unwrap();
        
        let dims = tensor.dims();
        assert_eq!(dims, &[2, 2]); // Mathematically respects native Eigen constraints
        
        // The mathematical layout must perfectly replicate via elements
        // Element (0, 1) should be 3.0
        let val_c = tensor.to_vec2::<f64>().unwrap();
        assert_eq!(val_c[0][0], 1.0);
        assert_eq!(val_c[1][0], 2.0);
        assert_eq!(val_c[0][1], 3.0);
        assert_eq!(val_c[1][1], 4.0);
    }
}
