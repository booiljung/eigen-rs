//! Coefficient-wise operations for eigen-rs.

use crate::core::scalar::Scalar;
use crate::core::xpr::MatrixXpr;
// use crate::core::storage::Storage as _; // Try aliasing to avoid any potential conflict
use crate::core::storage::Storage;

/// Expression representing the coefficient-wise sum of two expressions.
pub struct CwiseAddOp<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    pub lhs: &'a L,
    pub rhs: &'a R,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, L, R> CwiseAddOp<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    pub fn new(lhs: &'a L, rhs: &'a R) -> Result<Self, String> {
        if lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols() {
            return Err(format!(
                "Dimension mismatch in addition: {}x{} vs {}x{}",
                lhs.rows(),
                lhs.cols(),
                rhs.rows(),
                rhs.cols()
            ));
        }
        Ok(Self {
            lhs,
            rhs,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<'a, T, L, R> crate::core::cuda::CudaDispatcher<T> for CwiseAddOp<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    fn try_assign_cuda<S: Storage<T>>(
        &self,
        _dest: &mut crate::core::matrix::Matrix<T, S>,
    ) -> Result<bool, String> {
        #[cfg(feature = "cuda")]
        {
            if let (Some(l_storage), Some(r_storage)) =
                (self.lhs.as_cuda_storage(), self.rhs.as_cuda_storage())
            {
                // Both operands are on CUDA.
                // We need to verify if dest is also on CUDA.
                if std::any::TypeId::of::<S>()
                    == std::any::TypeId::of::<crate::core::storage::cuda::CudaStorage<T>>()
                {
                    // Safety: We checked TypeId. Use row() and col() to build matrices for specialized call.
                    // This is still a bit round-about, but works.
                    // A better way is to call the kernel directly here.
                    let ctx = crate::core::cuda::get_cuda_context()?;
                    let n = _dest.size() as i32;
                    let a_ptr = l_storage.get_ptr(0, 0);
                    let b_ptr = r_storage.get_ptr(0, 0);
                    let c_ptr = _dest.storage_mut().get_ptr(0, 0) as *mut T;

                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                        unsafe {
                            ctx.launch_add_f32(
                                a_ptr as *const f32,
                                b_ptr as *const f32,
                                c_ptr as *mut f32,
                                n,
                            )?;
                        }
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }
}

impl<'a, T, L, R> MatrixXpr<T> for CwiseAddOp<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    fn rows(&self) -> usize {
        self.lhs.rows()
    }
    fn cols(&self) -> usize {
        self.lhs.cols()
    }

    #[inline]
    fn eval(&self, row: usize, col: usize) -> T {
        self.lhs.eval(row, col) + self.rhs.eval(row, col)
    }

    #[inline]
    fn packet_eval<P: crate::core::arch::Packet<T>>(&self, row: usize, col: usize) -> P
    where
        T: Scalar,
    {
        self.lhs.packet_eval::<P>(row, col) + self.rhs.packet_eval::<P>(row, col)
    }

    #[inline]
    fn has_linear_access(&self) -> bool {
        self.lhs.has_linear_access() && self.rhs.has_linear_access()
    }

    #[inline]
    fn eval_linear(&self, i: usize) -> T {
        self.lhs.eval_linear(i) + self.rhs.eval_linear(i)
    }

    #[inline]
    fn packet_eval_linear<P: crate::core::arch::Packet<T>>(&self, i: usize) -> P
    where
        T: Scalar,
    {
        self.lhs.packet_eval_linear::<P>(i) + self.rhs.packet_eval_linear::<P>(i)
    }

    #[inline(always)]
    fn try_eval_to<P: crate::core::arch::Packet<T>>(&self, dest: *mut T, size: usize) -> bool {
        if let (Some(l_ptr), Some(r_ptr)) = (self.lhs.as_ptr(), self.rhs.as_ptr()) {
            let packet_size = P::SIZE;
            let mut i = 0;

            // Heuristic for Streaming Store:
            // Use streaming stores only for very large matrices where cache pollution is a major concern.
            // Alignment check is strict for _mm256_stream_ps.
            let use_stream = size >= 32000 && (dest as usize) % 32 == 0;
            let prefetch_dist = 4 * packet_size;

            unsafe {
                if use_stream {
                    while i + 8 * packet_size <= size {
                        // Prefetch enabled for large streaming path
                        P::prefetch(l_ptr.add(i + prefetch_dist * 2));
                        P::prefetch(r_ptr.add(i + prefetch_dist * 2));

                        let a0 = P::load(l_ptr.add(i));
                        let b0 = P::load(r_ptr.add(i));
                        (a0 + b0).store_stream(dest.add(i));

                        let a1 = P::load(l_ptr.add(i + packet_size));
                        let b1 = P::load(r_ptr.add(i + packet_size));
                        (a1 + b1).store_stream(dest.add(i + packet_size));

                        let a2 = P::load(l_ptr.add(i + 2 * packet_size));
                        let b2 = P::load(r_ptr.add(i + 2 * packet_size));
                        (a2 + b2).store_stream(dest.add(i + 2 * packet_size));

                        let a3 = P::load(l_ptr.add(i + 3 * packet_size));
                        let b3 = P::load(r_ptr.add(i + 3 * packet_size));
                        (a3 + b3).store_stream(dest.add(i + 3 * packet_size));

                        let a4 = P::load(l_ptr.add(i + 4 * packet_size));
                        let b4 = P::load(r_ptr.add(i + 4 * packet_size));
                        (a4 + b4).store_stream(dest.add(i + 4 * packet_size));

                        let a5 = P::load(l_ptr.add(i + 5 * packet_size));
                        let b5 = P::load(r_ptr.add(i + 5 * packet_size));
                        (a5 + b5).store_stream(dest.add(i + 5 * packet_size));

                        let a6 = P::load(l_ptr.add(i + 6 * packet_size));
                        let b6 = P::load(r_ptr.add(i + 6 * packet_size));
                        (a6 + b6).store_stream(dest.add(i + 6 * packet_size));

                        let a7 = P::load(l_ptr.add(i + 7 * packet_size));
                        let b7 = P::load(r_ptr.add(i + 7 * packet_size));
                        (a7 + b7).store_stream(dest.add(i + 7 * packet_size));

                        i += 8 * packet_size;
                    }
                } else {
                    while i + 8 * packet_size <= size {
                        // Standard path (Now with Prefetch for L2/L3 latency hiding)
                        P::prefetch(l_ptr.add(i + prefetch_dist * 2));
                        P::prefetch(r_ptr.add(i + prefetch_dist * 2));

                        let a0 = P::load(l_ptr.add(i));
                        let b0 = P::load(r_ptr.add(i));
                        (a0 + b0).store(dest.add(i));

                        let a1 = P::load(l_ptr.add(i + packet_size));
                        let b1 = P::load(r_ptr.add(i + packet_size));
                        (a1 + b1).store(dest.add(i + packet_size));

                        let a2 = P::load(l_ptr.add(i + 2 * packet_size));
                        let b2 = P::load(r_ptr.add(i + 2 * packet_size));
                        (a2 + b2).store(dest.add(i + 2 * packet_size));

                        let a3 = P::load(l_ptr.add(i + 3 * packet_size));
                        let b3 = P::load(r_ptr.add(i + 3 * packet_size));
                        (a3 + b3).store(dest.add(i + 3 * packet_size));

                        let a4 = P::load(l_ptr.add(i + 4 * packet_size));
                        let b4 = P::load(r_ptr.add(i + 4 * packet_size));
                        (a4 + b4).store(dest.add(i + 4 * packet_size));

                        let a5 = P::load(l_ptr.add(i + 5 * packet_size));
                        let b5 = P::load(r_ptr.add(i + 5 * packet_size));
                        (a5 + b5).store(dest.add(i + 5 * packet_size));

                        let a6 = P::load(l_ptr.add(i + 6 * packet_size));
                        let b6 = P::load(r_ptr.add(i + 6 * packet_size));
                        (a6 + b6).store(dest.add(i + 6 * packet_size));

                        let a7 = P::load(l_ptr.add(i + 7 * packet_size));
                        let b7 = P::load(r_ptr.add(i + 7 * packet_size));
                        (a7 + b7).store(dest.add(i + 7 * packet_size));

                        i += 8 * packet_size;
                    }
                }

                // Remainder loops
                while i + packet_size <= size {
                    let a = P::load(l_ptr.add(i));
                    let b = P::load(r_ptr.add(i));
                    (a + b).store(dest.add(i));
                    i += packet_size;
                }
                while i < size {
                    *dest.add(i) = *l_ptr.add(i) + *r_ptr.add(i);
                    i += 1;
                }
            }
            return true;
        }
        false
    }
}

/// Expression representing the coefficient-wise subtraction of two expressions.
pub struct CwiseSubOp<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    pub lhs: &'a L,
    pub rhs: &'a R,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, L, R> CwiseSubOp<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    pub fn new(lhs: &'a L, rhs: &'a R) -> Result<Self, String> {
        if lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols() {
            return Err(format!(
                "Dimension mismatch in subtraction: {}x{} vs {}x{}",
                lhs.rows(),
                lhs.cols(),
                rhs.rows(),
                rhs.cols()
            ));
        }
        Ok(Self {
            lhs,
            rhs,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<'a, T, L, R> crate::core::cuda::CudaDispatcher<T> for CwiseSubOp<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    fn try_assign_cuda<S: Storage<T>>(
        &self,
        _dest: &mut crate::core::matrix::Matrix<T, S>,
    ) -> Result<bool, String> {
        #[cfg(feature = "cuda")]
        {
            if let (Some(l_storage), Some(r_storage)) =
                (self.lhs.as_cuda_storage(), self.rhs.as_cuda_storage())
            {
                if std::any::TypeId::of::<S>()
                    == std::any::TypeId::of::<crate::core::storage::cuda::CudaStorage<T>>()
                {
                    let ctx = crate::core::cuda::get_cuda_context()?;
                    let n = _dest.size() as i32;
                    let a_ptr = l_storage.get_ptr(0, 0);
                    let b_ptr = r_storage.get_ptr(0, 0);
                    let c_ptr = _dest.storage_mut().get_ptr(0, 0) as *mut T;

                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                        unsafe {
                            ctx.launch_sub_f32(
                                a_ptr as *const f32,
                                b_ptr as *const f32,
                                c_ptr as *mut f32,
                                n,
                            )?;
                        }
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }
}

impl<'a, T, L, R> MatrixXpr<T> for CwiseSubOp<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    fn rows(&self) -> usize {
        self.lhs.rows()
    }
    fn cols(&self) -> usize {
        self.lhs.cols()
    }

    #[inline]
    fn eval(&self, row: usize, col: usize) -> T {
        self.lhs.eval(row, col) - self.rhs.eval(row, col)
    }

    #[inline]
    fn packet_eval<P: crate::core::arch::Packet<T>>(&self, row: usize, col: usize) -> P
    where
        T: Scalar,
    {
        self.lhs.packet_eval::<P>(row, col) - self.rhs.packet_eval::<P>(row, col)
    }

    #[inline]
    fn has_linear_access(&self) -> bool {
        self.lhs.has_linear_access() && self.rhs.has_linear_access()
    }

    #[inline]
    fn eval_linear(&self, i: usize) -> T {
        self.lhs.eval_linear(i) - self.rhs.eval_linear(i)
    }

    #[inline]
    fn packet_eval_linear<P: crate::core::arch::Packet<T>>(&self, i: usize) -> P
    where
        T: Scalar,
    {
        self.lhs.packet_eval_linear::<P>(i) - self.rhs.packet_eval_linear::<P>(i)
    }

    #[inline(always)]
    fn try_eval_to<P: crate::core::arch::Packet<T>>(&self, dest: *mut T, size: usize) -> bool {
        if let (Some(l_ptr), Some(r_ptr)) = (self.lhs.as_ptr(), self.rhs.as_ptr()) {
            let packet_size = P::SIZE;
            let mut i = 0;

            // Heuristic for Streaming Store:
            let use_stream = size >= 32000 && (dest as usize) % 32 == 0;
            let prefetch_dist = 4 * packet_size;

            unsafe {
                if use_stream {
                    while i + 8 * packet_size <= size {
                        P::prefetch(l_ptr.add(i + prefetch_dist * 2));
                        P::prefetch(r_ptr.add(i + prefetch_dist * 2));

                        let a0 = P::load(l_ptr.add(i));
                        let b0 = P::load(r_ptr.add(i));
                        (a0 - b0).store_stream(dest.add(i));

                        let a1 = P::load(l_ptr.add(i + packet_size));
                        let b1 = P::load(r_ptr.add(i + packet_size));
                        (a1 - b1).store_stream(dest.add(i + packet_size));

                        let a2 = P::load(l_ptr.add(i + 2 * packet_size));
                        let b2 = P::load(r_ptr.add(i + 2 * packet_size));
                        (a2 - b2).store_stream(dest.add(i + 2 * packet_size));

                        let a3 = P::load(l_ptr.add(i + 3 * packet_size));
                        let b3 = P::load(r_ptr.add(i + 3 * packet_size));
                        (a3 - b3).store_stream(dest.add(i + 3 * packet_size));

                        let a4 = P::load(l_ptr.add(i + 4 * packet_size));
                        let b4 = P::load(r_ptr.add(i + 4 * packet_size));
                        (a4 - b4).store_stream(dest.add(i + 4 * packet_size));

                        let a5 = P::load(l_ptr.add(i + 5 * packet_size));
                        let b5 = P::load(r_ptr.add(i + 5 * packet_size));
                        (a5 - b5).store_stream(dest.add(i + 5 * packet_size));

                        let a6 = P::load(l_ptr.add(i + 6 * packet_size));
                        let b6 = P::load(r_ptr.add(i + 6 * packet_size));
                        (a6 - b6).store_stream(dest.add(i + 6 * packet_size));

                        let a7 = P::load(l_ptr.add(i + 7 * packet_size));
                        let b7 = P::load(r_ptr.add(i + 7 * packet_size));
                        (a7 - b7).store_stream(dest.add(i + 7 * packet_size));

                        i += 8 * packet_size;
                    }
                } else {
                    while i + 8 * packet_size <= size {
                        // Standard path (Now with Prefetch)
                        P::prefetch(l_ptr.add(i + prefetch_dist * 2));
                        P::prefetch(r_ptr.add(i + prefetch_dist * 2));

                        let a0 = P::load(l_ptr.add(i));
                        let b0 = P::load(r_ptr.add(i));
                        (a0 - b0).store(dest.add(i));

                        let a1 = P::load(l_ptr.add(i + packet_size));
                        let b1 = P::load(r_ptr.add(i + packet_size));
                        (a1 - b1).store(dest.add(i + packet_size));

                        let a2 = P::load(l_ptr.add(i + 2 * packet_size));
                        let b2 = P::load(r_ptr.add(i + 2 * packet_size));
                        (a2 - b2).store(dest.add(i + 2 * packet_size));

                        let a3 = P::load(l_ptr.add(i + 3 * packet_size));
                        let b3 = P::load(r_ptr.add(i + 3 * packet_size));
                        (a3 - b3).store(dest.add(i + 3 * packet_size));

                        let a4 = P::load(l_ptr.add(i + 4 * packet_size));
                        let b4 = P::load(r_ptr.add(i + 4 * packet_size));
                        (a4 - b4).store(dest.add(i + 4 * packet_size));

                        let a5 = P::load(l_ptr.add(i + 5 * packet_size));
                        let b5 = P::load(r_ptr.add(i + 5 * packet_size));
                        (a5 - b5).store(dest.add(i + 5 * packet_size));

                        let a6 = P::load(l_ptr.add(i + 6 * packet_size));
                        let b6 = P::load(r_ptr.add(i + 6 * packet_size));
                        (a6 - b6).store(dest.add(i + 6 * packet_size));

                        let a7 = P::load(l_ptr.add(i + 7 * packet_size));
                        let b7 = P::load(r_ptr.add(i + 7 * packet_size));
                        (a7 - b7).store(dest.add(i + 7 * packet_size));

                        i += 8 * packet_size;
                    }
                }

                while i + packet_size <= size {
                    let a = P::load(l_ptr.add(i));
                    let b = P::load(r_ptr.add(i));
                    (a - b).store(dest.add(i));
                    i += packet_size;
                }
                while i < size {
                    *dest.add(i) = *l_ptr.add(i) - *r_ptr.add(i);
                    i += 1;
                }
            }
            return true;
        }
        false
    }
}

/// Expression representing the coefficient-wise multiplication by a scalar.
pub struct CwiseScalarMulOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    pub xpr: &'a X,
    pub scalar: T,
}

impl<'a, T, X> CwiseScalarMulOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    pub fn new(xpr: &'a X, scalar: T) -> Self {
        Self { xpr, scalar }
    }
}

impl<'a, T, X> crate::core::cuda::CudaDispatcher<T> for CwiseScalarMulOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    fn try_assign_cuda<S: Storage<T>>(
        &self,
        _dest: &mut crate::core::matrix::Matrix<T, S>,
    ) -> Result<bool, String> {
        #[cfg(feature = "cuda")]
        {
            if let Some(x_storage) = self.xpr.as_cuda_storage() {
                if std::any::TypeId::of::<S>()
                    == std::any::TypeId::of::<crate::core::storage::cuda::CudaStorage<T>>()
                {
                    let ctx = crate::core::cuda::get_cuda_context()?;
                    let n = _dest.size() as i32;
                    let a_ptr = x_storage.get_ptr(0, 0);
                    let c_ptr = _dest.storage_mut().get_ptr(0, 0) as *mut T;

                    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                        let s_f32: f32 = unsafe { *(&self.scalar as *const T as *const f32) };
                        unsafe {
                            ctx.launch_scalar_mul_f32(
                                a_ptr as *const f32,
                                s_f32,
                                c_ptr as *mut f32,
                                n,
                            )?;
                        }
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }
}

impl<'a, T, X> MatrixXpr<T> for CwiseScalarMulOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    fn rows(&self) -> usize {
        self.xpr.rows()
    }
    fn cols(&self) -> usize {
        self.xpr.cols()
    }

    #[inline]
    fn eval(&self, row: usize, col: usize) -> T {
        self.xpr.eval(row, col) * self.scalar
    }

    #[inline]
    fn packet_eval<P: crate::core::arch::Packet<T>>(&self, row: usize, col: usize) -> P
    where
        T: Scalar,
    {
        self.xpr.packet_eval::<P>(row, col) * P::set1(self.scalar)
    }

    #[inline]
    fn has_linear_access(&self) -> bool {
        self.xpr.has_linear_access()
    }

    #[inline]
    fn eval_linear(&self, i: usize) -> T {
        self.xpr.eval_linear(i) * self.scalar
    }

    #[inline]
    fn packet_eval_linear<P: crate::core::arch::Packet<T>>(&self, i: usize) -> P
    where
        T: Scalar,
    {
        self.xpr.packet_eval_linear::<P>(i) * P::set1(self.scalar)
    }

    #[inline(always)]
    fn try_eval_to<P: crate::core::arch::Packet<T>>(&self, dest: *mut T, size: usize) -> bool {
        if let Some(x_ptr) = self.xpr.as_ptr() {
            let packet_size = P::SIZE;
            let mut i = 0;
            let s_packet = P::set1(self.scalar);

            // Prefetch distance
            let prefetch_dist = 4 * packet_size;

            unsafe {
                while i + 8 * packet_size <= size {
                    // Prefetch only x_ptr (no streaming store used here as size unknown or scalar mult might not benefit enough from NT)
                    // But for consistency:
                    P::prefetch(x_ptr.add(i + prefetch_dist * 2));

                    let a0 = P::load(x_ptr.add(i));
                    (a0 * s_packet).store(dest.add(i));

                    let a1 = P::load(x_ptr.add(i + packet_size));
                    (a1 * s_packet).store(dest.add(i + packet_size));

                    let a2 = P::load(x_ptr.add(i + 2 * packet_size));
                    (a2 * s_packet).store(dest.add(i + 2 * packet_size));

                    let a3 = P::load(x_ptr.add(i + 3 * packet_size));
                    (a3 * s_packet).store(dest.add(i + 3 * packet_size));

                    let a4 = P::load(x_ptr.add(i + 4 * packet_size));
                    (a4 * s_packet).store(dest.add(i + 4 * packet_size));

                    let a5 = P::load(x_ptr.add(i + 5 * packet_size));
                    (a5 * s_packet).store(dest.add(i + 5 * packet_size));

                    let a6 = P::load(x_ptr.add(i + 6 * packet_size));
                    (a6 * s_packet).store(dest.add(i + 6 * packet_size));

                    let a7 = P::load(x_ptr.add(i + 7 * packet_size));
                    (a7 * s_packet).store(dest.add(i + 7 * packet_size));

                    i += 8 * packet_size;
                }
                while i + packet_size <= size {
                    let a = P::load(x_ptr.add(i));
                    (a * s_packet).store(dest.add(i));
                    i += packet_size;
                }
                while i < size {
                    *dest.add(i) = *x_ptr.add(i) * self.scalar;
                    i += 1;
                }
            }
            return true;
        }
        false
    }
}

/// Expression representing the matrix product of two expressions.
pub struct Product<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    lhs: &'a L,
    rhs: &'a R,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, L, R> Product<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    pub fn new(lhs: &'a L, rhs: &'a R) -> Result<Self, String> {
        if lhs.cols() != rhs.rows() {
            return Err(format!(
                "Dimension mismatch in multiplication: L cols {} != R rows {}",
                lhs.cols(),
                rhs.rows()
            ));
        }
        Ok(Self {
            lhs,
            rhs,
            _phantom: std::marker::PhantomData,
        })
    }
    pub fn lhs(&self) -> &'a L {
        self.lhs
    }
    pub fn rhs(&self) -> &'a R {
        self.rhs
    }
}

impl<'a, T, L, R> crate::core::cuda::CudaDispatcher<T> for Product<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
}

impl<'a, T, L, R> MatrixXpr<T> for Product<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    fn rows(&self) -> usize {
        self.lhs.rows()
    }
    fn cols(&self) -> usize {
        self.rhs.cols()
    }

    fn eval(&self, row: usize, col: usize) -> T {
        let mut sum = T::default();
        let k_end = self.lhs.cols();
        for k in 0..k_end {
            sum += self.lhs.eval(row, k) * self.rhs.eval(k, col);
        }
        sum
    }
}

impl<'a, T, L, R> std::ops::Mul<T> for &'a Product<'a, T, L, R>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
{
    type Output = CwiseScalarMulOp<'a, T, Product<'a, T, L, R>>;

    fn mul(self, rhs: T) -> Self::Output {
        CwiseScalarMulOp::new(self, rhs)
    }
}

/// Expression representing the transpose of an expression.
#[derive(Clone, Copy)]
pub struct TransposeOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    xpr: &'a X,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, X> TransposeOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    pub fn new(xpr: &'a X) -> Self {
        Self {
            xpr,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, X> crate::core::cuda::CudaDispatcher<T> for TransposeOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
}

impl<'a, T, X> MatrixXpr<T> for TransposeOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    fn rows(&self) -> usize {
        self.xpr.cols()
    }
    fn cols(&self) -> usize {
        self.xpr.rows()
    }

    fn eval(&self, row: usize, col: usize) -> T {
        // Swapping row and col for transposition
        self.xpr.eval(col, row)
    }
}

/// Expression representing a sub-region (block) of an expression.
pub struct BlockOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    xpr: &'a X,
    start_row: usize,
    start_col: usize,
    rows: usize,
    cols: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, X> BlockOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    pub fn new(
        xpr: &'a X,
        start_row: usize,
        start_col: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, String> {
        if start_row + rows > xpr.rows() || start_col + cols > xpr.cols() {
            return Err(format!(
                "Block out of bounds: region {}x{} starting at ({}, {}) exceeds expression {}x{}",
                rows,
                cols,
                start_row,
                start_col,
                xpr.rows(),
                xpr.cols()
            ));
        }
        Ok(Self {
            xpr,
            start_row,
            start_col,
            rows,
            cols,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<'a, T, X> crate::core::cuda::CudaDispatcher<T> for BlockOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
}

impl<'a, T, X> MatrixXpr<T> for BlockOp<'a, T, X>
where
    T: Scalar,
    X: MatrixXpr<T>,
{
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }

    fn eval(&self, row: usize, col: usize) -> T {
        self.xpr.eval(self.start_row + row, self.start_col + col)
    }

    fn packet_eval<P: crate::core::arch::Packet<T>>(&self, row: usize, col: usize) -> P
    where
        T: Scalar,
    {
        self.xpr
            .packet_eval::<P>(self.start_row + row, self.start_col + col)
    }
}

pub mod gemm;
pub mod unary;
pub use unary::{CwiseUnaryOp, ScalarCos, ScalarExp, ScalarLog, ScalarSin, UnaryFunctor};

// Mul impls are in matrix.rs
