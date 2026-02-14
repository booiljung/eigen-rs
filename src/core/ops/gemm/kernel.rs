use crate::core::scalar::Scalar;

pub trait GemmKernel {
    type Elem: Scalar;

    /// Micro-kernel width (Rows)
    const MR: usize;
    /// Micro-kernel height (Cols)
    const NR: usize;

    /// Alignments needed
    const MR_DIV: usize = 1;
    const NR_DIV: usize = 1;

    /// Execute the micro-kernel: C += A * B
    ///
    /// - `kc`: Depths (common dimension)
    /// - `alpha`: Scaling factor
    /// - `a`: Packed LHS (MC x KC)
    /// - `b`: Packed RHS (KC x NC)
    /// - `c`: Output block (MC x NC) of size `rs_c * cs_c`
    /// - `rs_c`: Row stride of C
    /// - `cs_c`: Col stride of C
    /// # Safety
    /// Pointers must be valid.
    #[allow(clippy::too_many_arguments)]
    unsafe fn microkernel(
        kc: usize,
        alpha: Self::Elem,
        a: *const Self::Elem,
        b: *const Self::Elem,
        beta: Self::Elem,
        c: *mut Self::Elem,
        rs_c: isize,
        cs_c: isize,
    );

    /// Pack LHS matrix panel A into block-major format
    /// Default implementation uses generic scalar packing.
    /// # Safety
    /// Pointers must be valid.
    unsafe fn pack_lhs(
        kc: usize,
        mc: usize,
        a: *const Self::Elem,
        rs: isize,
        cs: isize,
        packed: *mut Self::Elem,
    ) {
        crate::core::ops::gemm::packing::pack_lhs(Self::MR, kc, mc, a, rs, cs, packed)
    }

    /// Pack RHS matrix panel B into block-major format
    /// Default implementation uses generic scalar packing.
    /// # Safety
    /// Pointers must be valid.
    unsafe fn pack_rhs(
        kc: usize,
        nc: usize,
        b: *const Self::Elem,
        rs: isize,
        cs: isize,
        packed: *mut Self::Elem,
    ) {
        crate::core::ops::gemm::packing::pack_rhs(Self::NR, kc, nc, b, rs, cs, packed)
    }
}
