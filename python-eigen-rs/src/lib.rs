use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, ndarray::Array2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::xpr::MatrixXpr;

/// A Python wrapper for the Eigen-RS MatrixX<f64>
#[pyclass(name = "PyMatrixF64")]
pub struct PyMatrixF64 {
    pub(crate) inner: MatrixX<f64>,
}

#[pymethods]
impl PyMatrixF64 {
    /// Create a PyMatrixF64 from a 2D NumPy array.
    #[staticmethod]
    pub fn from_numpy(array: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let view = array.as_array();
        let rows = view.shape()[0];
        let cols = view.shape()[1];
        
        let mut mat = MatrixX::<f64>::new_dynamic(rows, cols)
            .map_err(|e| PyValueError::new_err(e))?;
        
        // Deep copy from NumPy to column-major Eigen format
        for r in 0..rows {
            for c in 0..cols {
                *mat.get_mut(r, c).unwrap() = view[[r, c]];
            }
        }
        
        Ok(PyMatrixF64 { inner: mat })
    }

    /// Convert the PyMatrixF64 back into a NumPy array.
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        let rows = self.inner.rows();
        let cols = self.inner.cols();
        
        // Extract to row-major vec for NumPy compatibility
        let mut data = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[r * cols + c] = *self.inner.get(r, c).unwrap();
            }
        }
        
        let arr = Array2::from_shape_vec((rows, cols), data).expect("Failed to create Array2");
        arr.into_pyarray(py)
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &PyMatrixF64) -> PyResult<Self> {
        let expr = &self.inner * &other.inner;
        let mut res = MatrixX::<f64>::new_dynamic(expr.rows(), expr.cols())
            .map_err(|e| PyValueError::new_err(e))?;
            
        res.assign(&expr).map_err(|e| PyValueError::new_err(e))?;
        Ok(PyMatrixF64 { inner: res })
    }
}

/// A pure Rust port of the Eigen library
#[pymodule]
fn python_eigen_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMatrixF64>()?;
    Ok(())
}
