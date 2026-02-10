//! Optimized Matrix Multiplication (GEMM) kernels.
use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::Storage;
use crate::core::xpr::MatrixXpr;

// Module definitions
pub mod arch;
pub mod kernel;
pub mod packing;

use self::kernel::GemmKernel;

// Default Blocking Parameters (L2/L3 cache dependent)
// For now, hardcoded reasonable defaults.
const MC: usize = 256; // tall & thin panel of A
const KC: usize = 128; // width of A panel / height of B panel
const NC: usize = 4096; // wide panel of B (streaming)

/// Blocked GEMM Driver
/// C += A * B
/// Blocked GEMM Driver
/// C += A * B
///
/// A: m x k
/// B: k x n
/// C: m x n
/// # Safety
/// Pointers must be valid.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_blocked<T, K>(
    m: usize,
    k: usize,
    n: usize,
    a: *const T,
    rs_a: isize,
    cs_a: isize,
    b: *const T,
    rs_b: isize,
    cs_b: isize,
    c: *mut T,
    rs_c: isize,
    cs_c: isize,
) -> Result<(), String>
where
    T: Scalar + Copy + Default + num_traits::One,
    K: GemmKernel<Elem = T>,
{
    // Buffers for packed panels
    // MC x KC, padded to multiple of MR
    let mc_rounded = MC.div_ceil(K::MR) * K::MR;
    let packed_a_len = mc_rounded * KC;
    let mut packed_a = vec![T::default(); packed_a_len];

    // KC x NC, padded to multiple of NR
    let nc_rounded = NC.div_ceil(K::NR) * K::NR;
    let packed_b_len = KC * nc_rounded;
    let mut packed_b = vec![T::default(); packed_b_len];

    // Macro-kernel loops (GOTO BLAS Style)
    // Loop 5: JC loop (Column blocks of C)
    for jc in (0..n).step_by(NC) {
        let nc_eff = std::cmp::min(n - jc, NC);

        // Loop 4: KC loop (Accumulation blocks via K)
        for pc in (0..k).step_by(KC) {
            let kc_eff = std::cmp::min(k - pc, KC);

            // Pack B (KC x NC) -> PackedB
            // Submatrix B(pc..pc+kc_eff, jc..jc+nc_eff)
            let b_sub_pointer = b.offset((pc as isize) * rs_b + (jc as isize) * cs_b);
            packing::pack_rhs::<T>(
                K::NR,
                kc_eff,
                nc_eff,
                b_sub_pointer,
                rs_b,
                cs_b,
                packed_b.as_mut_ptr(),
            );

            // Loop 3: IC loop (Row blocks of C)
            for ic in (0..m).step_by(MC) {
                let mc_eff = std::cmp::min(m - ic, MC);

                // Pack A (MC x KC) -> PackedA
                // Submatrix A(ic..ic+mc_eff, pc..pc+kc_eff)
                let a_sub_pointer = a.offset((ic as isize) * rs_a + (pc as isize) * cs_a);
                packing::pack_lhs::<T>(
                    K::MR,
                    kc_eff,
                    mc_eff,
                    a_sub_pointer,
                    rs_a,
                    cs_a,
                    packed_a.as_mut_ptr(),
                );

                // Micro-kernel Loop (jr, ir)
                // Temporary buffer for edge cases (allocated once per thread conceptually, but here locally)
                // To avoid alloc inside loop, we should alloc outside, but for simplicity/safety let's alloc here or use small stack array if possible?
                // Vec is safer.
                let mut tmp_c = vec![T::default(); K::MR * K::NR];

                for jr in (0..nc_eff).step_by(K::NR) {
                    let nr_curr = std::cmp::min(nc_eff - jr, K::NR);

                    for ir in (0..mc_eff).step_by(K::MR) {
                        let mr_curr = std::cmp::min(mc_eff - ir, K::MR);

                        // Call Micro-kernel
                        let a_ptr = packed_a.as_ptr().add(ir * kc_eff);
                        let b_ptr = packed_b.as_ptr().add(jr * kc_eff);

                        // Check if we are at edge
                        if mr_curr == K::MR && nr_curr == K::NR {
                            // Fast path: Direct write
                            let c_ptr = c.offset(
                                (ic as isize + ir as isize) * rs_c
                                    + (jc as isize + jr as isize) * cs_c,
                            );
                            K::microkernel(
                                kc_eff,
                                T::one(), // alpha
                                a_ptr,
                                b_ptr,
                                T::one(), // beta=1 (accumulate)
                                c_ptr,
                                rs_c,
                                cs_c,
                            );
                        } else {
                            // Slow path: Compute to tmp, then accumulation
                            // 1. Compute tmp = A * B (beta=0)
                            K::microkernel(
                                kc_eff,
                                T::one(),
                                a_ptr,
                                b_ptr,
                                T::default(), // beta=0
                                tmp_c.as_mut_ptr(),
                                1,              // rs (col-major)
                                K::MR as isize, // cs
                            );

                            // 2. Accumulate tmp to C
                            // C(ic+ir..ic+ir+mr_curr, jc+jr..jc+jr+nr_curr) += tmp
                            for j in 0..nr_curr {
                                for i in 0..mr_curr {
                                    let c_idx = (ic as isize + (ir + i) as isize) * rs_c
                                        + (jc as isize + (jr + j) as isize) * cs_c;
                                    let tmp_val = tmp_c[j * K::MR + i];
                                    *c.offset(c_idx) = *c.offset(c_idx) + tmp_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Baseline Column-Major GEMM that works on any MatrixXpr.
pub fn gemm_cm_unoptimized_xpr<T, L, R, SC>(
    a: &L,
    b: &R,
    c: &mut Matrix<T, SC>,
) -> Result<(), String>
where
    T: Scalar,
    L: MatrixXpr<T>,
    R: MatrixXpr<T>,
    SC: Storage<T>,
{
    let m = a.rows();
    let k_end = a.cols();
    let n = b.cols();

    // Clear C
    for j in 0..n {
        for i in 0..m {
            if let Some(val) = c.get_mut(i, j) {
                *val = T::default();
            }
        }
    }

    for j in 0..n {
        for k in 0..k_end {
            let b_val = b.eval(k, j);
            if b_val != T::default() {
                for i in 0..m {
                    *c.get_mut(i, j).unwrap() += a.eval(i, k) * b_val;
                }
            }
        }
    }
    Ok(())
}

pub fn gemm_cm<T, SA, SB, SC>(
    a: &Matrix<T, SA>,
    b: &Matrix<T, SB>,
    c: &mut Matrix<T, SC>,
) -> Result<(), String>
where
    T: Scalar + Copy + Default,
    SA: Storage<T>,
    SB: Storage<T>,
    SC: Storage<T>,
{
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let tid = std::any::TypeId::of::<T>();
        if tid == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
            use self::arch::x86::asm_kernel::AsmFmaKernelF32;
            c.set_zero();

            // Get pointers and assume Column-Major Dense for now (Standard Matrix)
            // TODO: Handle Strides correctly for Map/Slice.
            // Current Matrix impl is dense col-major.
            // a(i, k) is at [k * m + i]
            // rs = 1, cs = rows

            let a_ptr = a.storage().data().as_ptr(); // This assumes contiguous slice!
            let rs_a = 1;
            let cs_a = a.rows() as isize;

            let b_ptr = b.storage().data().as_ptr();
            let rs_b = 1;
            let cs_b = b.rows() as isize; // b.rows() is k

            let c_ptr = c.storage_mut().data_mut().as_mut_ptr();
            let rs_c = 1;
            let cs_c = c.rows() as isize;

            // Safety: Checked TypeId. Pointers are valid.
            unsafe {
                gemm_blocked::<f32, AsmFmaKernelF32>(
                    m,
                    k,
                    n,
                    a_ptr as *const f32,
                    rs_a,
                    cs_a,
                    b_ptr as *const f32,
                    rs_b,
                    cs_b,
                    c_ptr as *mut f32,
                    rs_c,
                    cs_c,
                )?;
            }
            return Ok(());
        }
        if tid == std::any::TypeId::of::<f64>() && is_x86_feature_detected!("fma") {
            use self::arch::x86::asm_kernel::AsmFmaKernelF64;
            c.set_zero();

            let a_ptr = a.storage().data().as_ptr();
            let rs_a = 1;
            let cs_a = a.rows() as isize;

            let b_ptr = b.storage().data().as_ptr();
            let rs_b = 1;
            let cs_b = b.rows() as isize;

            let c_ptr = c.storage_mut().data_mut().as_mut_ptr();
            let rs_c = 1;
            let cs_c = c.rows() as isize;

            unsafe {
                gemm_blocked::<f64, AsmFmaKernelF64>(
                    m,
                    k,
                    n,
                    a_ptr as *const f64,
                    rs_a,
                    cs_a,
                    b_ptr as *const f64,
                    rs_b,
                    cs_b,
                    c_ptr as *mut f64,
                    rs_c,
                    cs_c,
                )?;
            }
            return Ok(());
        }
    }

    // Fallback
    gemm_cm_unoptimized_xpr(a, b, c)
}
