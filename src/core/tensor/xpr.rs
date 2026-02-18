//! Expression templates for Tensors.

use crate::core::scalar::Scalar;
#[cfg(feature = "cuda")]
use crate::core::tensor::device::cuda::{CudaDevice, CudaStorage};

/// Trait for N-dimensional tensor expressions.
pub trait TensorXpr<T: Scalar, const RANK: usize> {
    /// Returns the dimensions of the tensor expression.
    fn dims(&self) -> [usize; RANK];

    /// Evaluates the expression at the given indices.
    fn eval(&self, indices: [usize; RANK]) -> T;

    /// Returns the total number of elements.
    fn size(&self) -> usize {
        self.dims().iter().product()
    }
    
    /// Returns a reference to the underlying CUDA storage if available.
    /// This is used for eager execution on GPU.
    #[cfg(feature = "cuda")]
    fn as_cuda_storage(&self) -> Option<&CudaStorage<T>> {
        None
    }

    /// Evaluates the expression directly into the output CUDA storage.
    #[cfg(feature = "cuda")]
    fn eval_on_cuda(&self, _device: &CudaDevice, _out: &mut CudaStorage<T>) -> Result<(), ()> {
        Err(())
    }
}

// TensorXpr implementations for Tensor moved to mod.rs to allow access to internals.

/// Lazy coefficient-wise addition.
/// Holds L and R by value to allow moving temporary expressions.
pub struct CwiseTensorAddOp<T: Scalar, const RANK: usize, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    lhs: L,
    rhs: R,
    dims: [usize; RANK],
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const RANK: usize, L, R> CwiseTensorAddOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    pub fn new(lhs: L, rhs: R) -> Result<Self, String> {
        if lhs.dims() != rhs.dims() {
            return Err("Dimension mismatch in Tensor addition".to_string());
        }
        Ok(Self {
            dims: lhs.dims(),
            lhs,
            rhs,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<T: Scalar, const RANK: usize, L, R> TensorXpr<T, RANK> for CwiseTensorAddOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.dims
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.lhs.eval(indices) + self.rhs.eval(indices)
    }
    
    #[cfg(feature = "cuda")]
    fn eval_on_cuda(&self, device: &CudaDevice, out: &mut CudaStorage<T>) -> Result<(), ()> {
        if let (Some(l_store), Some(r_store)) = (self.lhs.as_cuda_storage(), self.rhs.as_cuda_storage()) {
            CudaOpsHelper::add(device, out, l_store, r_store)
        } else {
            Err(())
        }
    }
}

// Helper trait to dispatch CUDA ops only for supported types
#[cfg(feature = "cuda")]
trait CudaOpsHelper<T: Scalar> {
    fn add(device: &CudaDevice, out: &mut CudaStorage<T>, a: &CudaStorage<T>, b: &CudaStorage<T>) -> Result<(), ()> {
        Err(())
    }
    
    // Add sub/mul/etc later
}

#[cfg(feature = "cuda")]
impl<T: Scalar + 'static> CudaOpsHelper<T> for T {
    fn add(device: &CudaDevice, out: &mut CudaStorage<T>, a: &CudaStorage<T>, b: &CudaStorage<T>) -> Result<(), ()> {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            let out_f32: &mut CudaStorage<f32> = unsafe { std::mem::transmute(out) };
            let a_f32: &CudaStorage<f32> = unsafe { std::mem::transmute(a) };
            let b_f32: &CudaStorage<f32> = unsafe { std::mem::transmute(b) };
            device.add(out_f32, a_f32, b_f32).map_err(|_| ())
        } else if TypeId::of::<T>() == TypeId::of::<f64>() {
            let out_f64: &mut CudaStorage<f64> = unsafe { std::mem::transmute(out) };
            let a_f64: &CudaStorage<f64> = unsafe { std::mem::transmute(a) };
            let b_f64: &CudaStorage<f64> = unsafe { std::mem::transmute(b) };
            device.add(out_f64, a_f64, b_f64).map_err(|_| ())
        } else {
            Err(())
        }
    }
}

/// Lazy coefficient-wise subtraction.
pub struct CwiseTensorSubOp<T: Scalar, const RANK: usize, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    lhs: L,
    rhs: R,
    dims: [usize; RANK],
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar, const RANK: usize, L, R> CwiseTensorSubOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    pub fn new(lhs: L, rhs: R) -> Result<Self, String> {
        if lhs.dims() != rhs.dims() {
            return Err("Dimension mismatch in Tensor subtraction".to_string());
        }
        Ok(Self {
            dims: lhs.dims(),
            lhs,
            rhs,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<T: Scalar, const RANK: usize, L, R> TensorXpr<T, RANK> for CwiseTensorSubOp<T, RANK, L, R>
where
    L: TensorXpr<T, RANK>,
    R: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.dims
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.lhs.eval(indices) - self.rhs.eval(indices)
    }
}

/// Lazy scalar multiplication (broadcasting).
pub struct CwiseTensorScalarMulOp<T: Scalar, const RANK: usize, X>
where
    X: TensorXpr<T, RANK>,
{
    xpr: X,
    scalar: T,
}

impl<T: Scalar, const RANK: usize, X> CwiseTensorScalarMulOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    pub fn new(xpr: X, scalar: T) -> Self {
        Self { xpr, scalar }
    }
}

impl<T: Scalar, const RANK: usize, X> TensorXpr<T, RANK> for CwiseTensorScalarMulOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.xpr.dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.xpr.eval(indices) * self.scalar
    }
}

/// Lazy scalar addition (broadcasting).
pub struct CwiseTensorScalarAddOp<T: Scalar, const RANK: usize, X>
where
    X: TensorXpr<T, RANK>,
{
    xpr: X,
    scalar: T,
}

impl<T: Scalar, const RANK: usize, X> CwiseTensorScalarAddOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    pub fn new(xpr: X, scalar: T) -> Self {
        Self { xpr, scalar }
    }
}

impl<T: Scalar, const RANK: usize, X> TensorXpr<T, RANK> for CwiseTensorScalarAddOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.xpr.dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.xpr.eval(indices) + self.scalar
    }
}

/// Lazy scalar subtraction (broadcasting).
pub struct CwiseTensorScalarSubOp<T: Scalar, const RANK: usize, X>
where
    X: TensorXpr<T, RANK>,
{
    xpr: X,
    scalar: T,
}

impl<T: Scalar, const RANK: usize, X> CwiseTensorScalarSubOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    pub fn new(xpr: X, scalar: T) -> Self {
        Self { xpr, scalar }
    }
}

impl<T: Scalar, const RANK: usize, X> TensorXpr<T, RANK> for CwiseTensorScalarSubOp<T, RANK, X>
where
    X: TensorXpr<T, RANK>,
{
    fn dims(&self) -> [usize; RANK] {
        self.xpr.dims()
    }

    fn eval(&self, indices: [usize; RANK]) -> T {
        self.xpr.eval(indices) - self.scalar
    }
}
