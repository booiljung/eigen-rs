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
const MC: usize = 960; // larger panel for 1MB+ L2 (960*256*4 = 960KB)
const KC: usize = 256; //
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
// Thread-local workspace to avoid allocation in hot loops
use std::cell::RefCell;
thread_local! {
    static GEMM_WORKSPACE: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(3 * 1024 * 1024));
}

fn with_gemm_workspace<F, R>(size_bytes: usize, f: F) -> R
where
    F: FnOnce(*mut u8) -> R,
{
    GEMM_WORKSPACE.with(|ws_cell| {
        let mut ws = ws_cell.borrow_mut();
        // Alignment padding (64 bytes)
        let required_cap = size_bytes + 64;
        if ws.capacity() < required_cap {
            let current = ws.capacity();
            let new_cap = std::cmp::max(required_cap, current * 2);
            let current_len = ws.len();
            ws.reserve(new_cap - current_len);
        }

        let ptr = ws.as_mut_ptr();
        let addr = ptr as usize;
        let offset = (64 - (addr % 64)) % 64;
        let aligned_ptr = unsafe { ptr.add(offset) };

        f(aligned_ptr)
    })
}

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
    alpha: T,
) -> Result<(), String>
where
    T: Scalar + Copy + Default + num_traits::One,
    K: GemmKernel<Elem = T>,
{
    // Sizes for packing buffers
    let mc_rounded = MC.div_ceil(K::MR) * K::MR;
    let packed_a_len = mc_rounded * KC;

    let nc_rounded = NC.div_ceil(K::NR) * K::NR;
    let packed_b_len = KC * nc_rounded;

    let tmp_c_len = K::MR * K::NR;

    let size_t = std::mem::size_of::<T>();
    let bytes_a = packed_a_len * size_t;
    let bytes_b = packed_b_len * size_t;
    let bytes_tmp = tmp_c_len * size_t;
    // Alignment padding (64 bytes) + Safety Padding (4096 bytes) to prevent OOB corruption
    let required_cap = bytes_a + bytes_b + bytes_tmp + 64 + 4096;

    with_gemm_workspace(required_cap, |ws_ptr| {
        // Partition workspace
        // A | B | TMP
        let ptr_a = ws_ptr as *mut T;
        let ptr_b = ws_ptr.add(bytes_a) as *mut T;
        let ptr_tmp = ws_ptr.add(bytes_a + bytes_b) as *mut T;

        // Create mutable slices (unsafe alias to workspace)
        // Note: we don't zero-init. packing overwrites. tmp_c overwrites.

        // Macro-kernel loops (GOTO BLAS Style)
        // Loop 5: JC loop (Column blocks of C)
        for jc in (0..n).step_by(NC) {
            let nc_eff = std::cmp::min(n - jc, NC);

            // Loop 4: KC loop (Accumulation blocks via K)
            for pc in (0..k).step_by(KC) {
                let kc_eff = std::cmp::min(k - pc, KC);

                // Pack B (KC x NC)
                let b_sub_pointer = b.offset((pc as isize) * rs_b + (jc as isize) * cs_b);
                K::pack_rhs(
                    kc_eff,
                    nc_eff,
                    b_sub_pointer,
                    rs_b,
                    cs_b,
                    ptr_b, // Packed B buffer
                );

                // Loop 3: IC loop (Row blocks of C)
                for ic in (0..m).step_by(MC) {
                    let mc_eff = std::cmp::min(m - ic, MC);

                    // Pack A (MC x KC)
                    let a_sub_pointer = a.offset((ic as isize) * rs_a + (pc as isize) * cs_a);
                    K::pack_lhs(
                        kc_eff,
                        mc_eff,
                        a_sub_pointer,
                        rs_a,
                        cs_a,
                        ptr_a, // Packed A buffer
                    );

                    // Micro-kernel Loop (jr, ir)
                    for jr in (0..nc_eff).step_by(K::NR) {
                        let nr_curr = std::cmp::min(nc_eff - jr, K::NR);

                        for ir in (0..mc_eff).step_by(K::MR) {
                            let mr_curr = std::cmp::min(mc_eff - ir, K::MR);

                            // Call Micro-kernel
                            // ptr_a is MC*KC. We access block at ir.
                            // ptr_aLayout: [MC_strips]. Strip is KC*MR.
                            // We need to offset carefully.
                            // Check packing.rs: pack_lhs produces contiguous MR-strips.
                            // Strip index = ir / MR.
                            // Offset = (ir / MR) * (KC * MR).
                            // My previous code used: `packed_a.as_ptr().add(ir * kc_eff)`.
                            // Let's verify.
                            // Iter ir (step MR). ir=0, MR, 2MR...
                            // (ir/MR) * KC * MR == ir * KC.
                            // This matches.
                            let a_ptr_k = ptr_a.add(ir * kc_eff);
                            let b_ptr_k = ptr_b.add(jr * kc_eff);

                            if mr_curr == K::MR && nr_curr == K::NR {
                                // Fast path
                                let c_ptr = c.offset(
                                    (ic as isize + ir as isize) * rs_c
                                        + (jc as isize + jr as isize) * cs_c,
                                );
                                K::microkernel(
                                    kc_eff,
                                    alpha,
                                    a_ptr_k,
                                    b_ptr_k,
                                    T::one(),
                                    c_ptr,
                                    rs_c,
                                    cs_c,
                                );
                            } else {
                                // Slow path using tmp_c
                                K::microkernel(
                                    kc_eff,
                                    alpha,
                                    a_ptr_k,
                                    b_ptr_k,
                                    T::default(),
                                    ptr_tmp, // tmp_c
                                    1,
                                    K::MR as isize,
                                );

                                // Accumulate
                                let tmp_slice = std::slice::from_raw_parts(ptr_tmp, K::MR * K::NR);
                                for j in 0..nr_curr {
                                    for i in 0..mr_curr {
                                        let c_idx = (ic as isize + (ir + i) as isize) * rs_c
                                            + (jc as isize + (jr + j) as isize) * cs_c;
                                        *c.offset(c_idx) =
                                            *c.offset(c_idx) + tmp_slice[j * K::MR + i];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })
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

    // Use specialized unsafe scalar fallback for small matrices to avoid packing/workspace overhead.
    if m <= 8 && n <= 8 && k <= 8 {
        // Prepare pointers for unsafe fallback
        let a_ptr = a.storage().data().as_ptr();
        // Since we are inside generic Matrix, we need to handle Strides?
        // Current Matrix impl is dense column-major, DynamicStorage is contiguous.
        // But let's check strides.
        // For DynamicStorage, strides are (1, rows).
        let rs_a = 1;
        let cs_a = a.rows() as isize;

        let b_ptr = b.storage().data().as_ptr();
        let rs_b = 1;
        let cs_b = b.rows() as isize;

        let c_ptr = c.storage_mut().data_mut().as_mut_ptr();
        let rs_c = 1;
        let cs_c = c.rows() as isize;
        
        unsafe {
            gemm_small_unsafe(
                m, k, n,
                a_ptr, rs_a, cs_a,
                b_ptr, rs_b, cs_b,
                c_ptr, rs_c, cs_c,
                T::from_usize(1)
            );
        }
        return Ok(());
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let tid = std::any::TypeId::of::<T>();
        if tid == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
            // Debug: Confirm optimized path
            eprintln!("DEBUG: Using AVX2 FMA Kernel for f32");
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
                    1.0,
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
                    1.0,
                )?;
            }
            return Ok(());
        }
    }

    // Fallback
    gemm_cm_unoptimized_xpr(a, b, c)
}

/// Dispatches GEMM based on runtime feature detection using raw pointers.
///
/// # Safety
/// Pointers must be valid for the specified dimensions and strides.
pub unsafe fn gemm_dispatch_pointers<T: Scalar + Copy + Default>(
    m: usize,
    k: usize,
    n: usize,
    a_ptr: *const T,
    rs_a: isize,
    cs_a: isize,
    b_ptr: *const T,
    rs_b: isize,
    cs_b: isize,
    c_ptr: *mut T,
    rs_c: isize,
    cs_c: isize,
) -> Result<bool, String> {
    // Use scalar fallback for small matrices to avoid packing/workspace overhead.
    if m <= 8 && n <= 8 && k <= 8 {
        gemm_small_unsafe(
            m, k, n,
            a_ptr, rs_a, cs_a,
            b_ptr, rs_b, cs_b,
            c_ptr, rs_c, cs_c,
            T::from_usize(1) // Assuming alpha=1 for dispatch
        );
        return Ok(true); // Handled
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let tid = std::any::TypeId::of::<T>();
        if tid == std::any::TypeId::of::<f32>() && is_x86_feature_detected!("fma") {
            use self::arch::x86::asm_kernel::AsmFmaKernelF32;
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
                1.0,
            )?;
            return Ok(true);
        }
        if tid == std::any::TypeId::of::<f64>() && is_x86_feature_detected!("fma") {
            use self::arch::x86::asm_kernel::AsmFmaKernelF64;
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
                1.0,
            )?;
            return Ok(true);
        }
    }
    Ok(false)
}

/// Unsafe Scalar GEMM for Small Matrices (N <= 16)
/// Avoids overhead of packing, workspace allocation, and bounds checking.
///
/// # Safety
/// Pointers must be valid and bounds `m, k, n` must match.
pub unsafe fn gemm_small_unsafe<T: Scalar + Copy + Default>(
    m: usize,
    k: usize,
    n: usize,
    a_ptr: *const T,
    rs_a: isize,
    cs_a: isize,
    b_ptr: *const T,
    rs_b: isize,
    cs_b: isize,
    c_ptr: *mut T,
    rs_c: isize,
    cs_c: isize,
    alpha: T,
) {
    // J-K-I Loop Order (Column-Major Friendly)
    for j in 0..n {
        let b_col = b_ptr.offset(j as isize * cs_b);
        let c_col = c_ptr.offset(j as isize * cs_c);

        for l in 0..k { // l used for k index to avoid confusion with k size
            let b_val = *b_col.offset(l as isize * rs_b) * alpha;
            let a_col = a_ptr.offset(l as isize * cs_a);

            for i in 0..m {
                let a_val = *a_col.offset(i as isize * rs_a);
                *c_col.offset(i as isize * rs_c) += a_val * b_val;
            }
        }
    }
}
