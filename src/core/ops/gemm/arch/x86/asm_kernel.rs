#[cfg(target_arch = "x86_64")]
use crate::core::ops::gemm::kernel::GemmKernel;
#[cfg(target_arch = "x86_64")]
use std::arch::asm;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct AsmFmaKernelF32;

#[cfg(target_arch = "x86_64")]
impl GemmKernel for AsmFmaKernelF32 {
    type Elem = f32;
    const MR: usize = 24;
    const NR: usize = 4;

    /// Override packing with AVX2 optimized version
    unsafe fn pack_lhs(
        kc: usize,
        mc: usize,
        a: *const f32,
        rs: isize,
        cs: isize,
        packed: *mut f32,
    ) {
        crate::core::ops::gemm::packing::pack_lhs_f32_avx(Self::MR, kc, mc, a, rs, cs, packed)
    }

    /// Override packing with AVX2 optimized version
    unsafe fn pack_rhs(
        kc: usize,
        nc: usize,
        b: *const f32,
        rs: isize,
        cs: isize,
        packed: *mut f32,
    ) {
        crate::core::ops::gemm::packing::pack_rhs_f32_avx(Self::NR, kc, nc, b, rs, cs, packed)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn microkernel(
        kc: usize,
        alpha: f32,
        a: *const f32,
        b: *const f32,
        beta: f32,
        c: *mut f32,
        rs_c: isize,
        cs_c: isize,
    ) {
        // Accumulators: 24 rows (3 regs) x 4 cols = 12 regs
        // c00, c01, c02, c03 (rows 0..7)
        // c10, c11, c12, c13 (rows 8..15)
        // c20, c21, c22, c23 (rows 16..23)

        let mut c00: __m256;
        let mut c01: __m256;
        let mut c02: __m256;
        let mut c03: __m256;
        let mut c10: __m256;
        let mut c11: __m256;
        let mut c12: __m256;
        let mut c13: __m256;
        let mut c20: __m256;
        let mut c21: __m256;
        let mut c22: __m256;
        let mut c23: __m256;

        // Initialize accumulators to zero
        asm!(
            "vxorps {c00}, {c00}, {c00}", "vxorps {c01}, {c01}, {c01}", "vxorps {c02}, {c02}, {c02}", "vxorps {c03}, {c03}, {c03}",
            "vxorps {c10}, {c10}, {c10}", "vxorps {c11}, {c11}, {c11}", "vxorps {c12}, {c12}, {c12}", "vxorps {c13}, {c13}, {c13}",
            "vxorps {c20}, {c20}, {c20}", "vxorps {c21}, {c21}, {c21}", "vxorps {c22}, {c22}, {c22}", "vxorps {c23}, {c23}, {c23}",
            c00 = out(ymm_reg) c00, c01 = out(ymm_reg) c01, c02 = out(ymm_reg) c02, c03 = out(ymm_reg) c03,
            c10 = out(ymm_reg) c10, c11 = out(ymm_reg) c11, c12 = out(ymm_reg) c12, c13 = out(ymm_reg) c13,
            c20 = out(ymm_reg) c20, c21 = out(ymm_reg) c21, c22 = out(ymm_reg) c22, c23 = out(ymm_reg) c23,
        );

        let mut a_ptr = a;
        let mut b_ptr = b;
        let mut k = kc;

        // Pointers to help asm
        // asm loop
        asm!(
            // ASM Loop (Fused K-loop, Unrolled 4x + Tail)
            "cmp {k}, 4",
            "jl 3f", // Jump to tail if k < 4

            "2:", // Loop label 4x

            // Iteration 0
            "vmovups {a0}, [{a_ptr}]",
            "vmovups {a1}, [{a_ptr} + 32]",
            "vmovups {a2}, [{a_ptr} + 64]",

            "vbroadcastss {b_val}, [{b_ptr}]",
            "vfmadd231ps {c00}, {a0}, {b_val}",
            "vfmadd231ps {c10}, {a1}, {b_val}",
            "vfmadd231ps {c20}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 4]",
            "vfmadd231ps {c01}, {a0}, {b_val}",
            "vfmadd231ps {c11}, {a1}, {b_val}",
            "vfmadd231ps {c21}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 8]",
            "vfmadd231ps {c02}, {a0}, {b_val}",
            "vfmadd231ps {c12}, {a1}, {b_val}",
            "vfmadd231ps {c22}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 12]",
            "vfmadd231ps {c03}, {a0}, {b_val}",
            "vfmadd231ps {c13}, {a1}, {b_val}",
            "vfmadd231ps {c23}, {a2}, {b_val}",

            // Iteration 1 (+96A, +16B)
            "vmovups {a0}, [{a_ptr} + 96]",
            "vmovups {a1}, [{a_ptr} + 128]",
            "vmovups {a2}, [{a_ptr} + 160]",

            "vbroadcastss {b_val}, [{b_ptr} + 16]",
            "vfmadd231ps {c00}, {a0}, {b_val}",
            "vfmadd231ps {c10}, {a1}, {b_val}",
            "vfmadd231ps {c20}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 20]",
            "vfmadd231ps {c01}, {a0}, {b_val}",
            "vfmadd231ps {c11}, {a1}, {b_val}",
            "vfmadd231ps {c21}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 24]",
            "vfmadd231ps {c02}, {a0}, {b_val}",
            "vfmadd231ps {c12}, {a1}, {b_val}",
            "vfmadd231ps {c22}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 28]",
            "vfmadd231ps {c03}, {a0}, {b_val}",
            "vfmadd231ps {c13}, {a1}, {b_val}",
            "vfmadd231ps {c23}, {a2}, {b_val}",

            // Iteration 2 (+192A, +32B)
            "vmovups {a0}, [{a_ptr} + 192]",
            "vmovups {a1}, [{a_ptr} + 224]",
            "vmovups {a2}, [{a_ptr} + 256]",

            "vbroadcastss {b_val}, [{b_ptr} + 32]",
            "vfmadd231ps {c00}, {a0}, {b_val}",
            "vfmadd231ps {c10}, {a1}, {b_val}",
            "vfmadd231ps {c20}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 36]",
            "vfmadd231ps {c01}, {a0}, {b_val}",
            "vfmadd231ps {c11}, {a1}, {b_val}",
            "vfmadd231ps {c21}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 40]",
            "vfmadd231ps {c02}, {a0}, {b_val}",
            "vfmadd231ps {c12}, {a1}, {b_val}",
            "vfmadd231ps {c22}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 44]",
            "vfmadd231ps {c03}, {a0}, {b_val}",
            "vfmadd231ps {c13}, {a1}, {b_val}",
            "vfmadd231ps {c23}, {a2}, {b_val}",

            // Iteration 3 (+288A, +48B)
            "vmovups {a0}, [{a_ptr} + 288]",
            "vmovups {a1}, [{a_ptr} + 320]",
            "vmovups {a2}, [{a_ptr} + 352]",

            "vbroadcastss {b_val}, [{b_ptr} + 48]",
            "vfmadd231ps {c00}, {a0}, {b_val}",
            "vfmadd231ps {c10}, {a1}, {b_val}",
            "vfmadd231ps {c20}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 52]",
            "vfmadd231ps {c01}, {a0}, {b_val}",
            "vfmadd231ps {c11}, {a1}, {b_val}",
            "vfmadd231ps {c21}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 56]",
            "vfmadd231ps {c02}, {a0}, {b_val}",
            "vfmadd231ps {c12}, {a1}, {b_val}",
            "vfmadd231ps {c22}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 60]",
            "vfmadd231ps {c03}, {a0}, {b_val}",
            "vfmadd231ps {c13}, {a1}, {b_val}",
            "vfmadd231ps {c23}, {a2}, {b_val}",


            // Update pointers (4x)
            "add {a_ptr}, 384",
            "add {b_ptr}, 64",
            "sub {k}, 4",
            "cmp {k}, 4",
            "jge 2b",

            "3:", // Tail
            "test {k}, {k}",
            "jz 5f", // Exit if k=0

            "4:", // Tail loop
            "vmovups {a0}, [{a_ptr}]",
            "vmovups {a1}, [{a_ptr} + 32]",
            "vmovups {a2}, [{a_ptr} + 64]",

            "vbroadcastss {b_val}, [{b_ptr}]",
            "vfmadd231ps {c00}, {a0}, {b_val}",
            "vfmadd231ps {c10}, {a1}, {b_val}",
            "vfmadd231ps {c20}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 4]",
            "vfmadd231ps {c01}, {a0}, {b_val}",
            "vfmadd231ps {c11}, {a1}, {b_val}",
            "vfmadd231ps {c21}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 8]",
            "vfmadd231ps {c02}, {a0}, {b_val}",
            "vfmadd231ps {c12}, {a1}, {b_val}",
            "vfmadd231ps {c22}, {a2}, {b_val}",

            "vbroadcastss {b_val}, [{b_ptr} + 12]",
            "vfmadd231ps {c03}, {a0}, {b_val}",
            "vfmadd231ps {c13}, {a1}, {b_val}",
            "vfmadd231ps {c23}, {a2}, {b_val}",

            "add {a_ptr}, 96",
            "add {b_ptr}, 16",
            "sub {k}, 1",
            "jnz 4b",

            "5:", // End
            a_ptr = inout(reg) a_ptr,
            b_ptr = inout(reg) b_ptr,
            k = inout(reg) k,

            c00 = inout(ymm_reg) c00, c01 = inout(ymm_reg) c01, c02 = inout(ymm_reg) c02, c03 = inout(ymm_reg) c03,
            c10 = inout(ymm_reg) c10, c11 = inout(ymm_reg) c11, c12 = inout(ymm_reg) c12, c13 = inout(ymm_reg) c13,
            c20 = inout(ymm_reg) c20, c21 = inout(ymm_reg) c21, c22 = inout(ymm_reg) c22, c23 = inout(ymm_reg) c23,

            a0 = out(ymm_reg) _,
            a1 = out(ymm_reg) _,
            a2 = out(ymm_reg) _,
            b_val = out(ymm_reg) _,
        );

        // Prefetch hint for next iteration of microkernel or next k-block
        // This is heuristic.
        asm!(
            "prefetcht0 [{a_ptr} + 384]", // Prefetch next block of A
            "prefetcht0 [{b_ptr} + 64]",  // Prefetch next block of B
            a_ptr = in(reg) a_ptr,
            b_ptr = in(reg) b_ptr,
        );

        // Store results
        // Use intrinsic logic for update: C = alpha*Acc + beta*C
        // We reuse the FMA intrinsics structure but use the computed accumulators

        let alphav = _mm256_set1_ps(alpha);

        // Helper to store: *ptr = alpha * val + beta * *ptr
        let store_col = |j: isize, c0_a: __m256, c1_a: __m256, c2_a: __m256| {
            let ptr0 = c.offset(j * cs_c);
            let ptr1 = c.offset(8 * rs_c + j * cs_c);
            let ptr2 = c.offset(16 * rs_c + j * cs_c);

            if beta == 0.0 {
                _mm256_storeu_ps(ptr0, _mm256_mul_ps(c0_a, alphav));
                _mm256_storeu_ps(ptr1, _mm256_mul_ps(c1_a, alphav));
                _mm256_storeu_ps(ptr2, _mm256_mul_ps(c2_a, alphav));
            } else {
                let betav = _mm256_set1_ps(beta);
                let existing0 = _mm256_loadu_ps(ptr0);
                let existing1 = _mm256_loadu_ps(ptr1);
                let existing2 = _mm256_loadu_ps(ptr2);

                let res0 = _mm256_fmadd_ps(c0_a, alphav, _mm256_mul_ps(existing0, betav));
                let res1 = _mm256_fmadd_ps(c1_a, alphav, _mm256_mul_ps(existing1, betav));
                let res2 = _mm256_fmadd_ps(c2_a, alphav, _mm256_mul_ps(existing2, betav));

                _mm256_storeu_ps(ptr0, res0);
                _mm256_storeu_ps(ptr1, res1);
                _mm256_storeu_ps(ptr2, res2);
            }
        };

        store_col(0, c00, c10, c20);
        store_col(1, c01, c11, c21);
        store_col(2, c02, c12, c22);
        store_col(3, c03, c13, c23);
    }
}

pub struct AsmFmaKernelF64;

#[cfg(target_arch = "x86_64")]
impl GemmKernel for AsmFmaKernelF64 {
    type Elem = f64;
    const MR: usize = 8;
    const NR: usize = 4;

    /// Override packing with AVX2 optimized version
    unsafe fn pack_lhs(
        kc: usize,
        mc: usize,
        a: *const f64,
        rs: isize,
        cs: isize,
        packed: *mut f64,
    ) {
        crate::core::ops::gemm::packing::pack_lhs_f64_avx(Self::MR, kc, mc, a, rs, cs, packed)
    }

    /// Override packing with AVX2 optimized version
    unsafe fn pack_rhs(
        kc: usize,
        nc: usize,
        b: *const f64,
        rs: isize,
        cs: isize,
        packed: *mut f64,
    ) {
        crate::core::ops::gemm::packing::pack_rhs_f64_avx(Self::NR, kc, nc, b, rs, cs, packed)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    unsafe fn microkernel(
        kc: usize,
        alpha: f64,
        a: *const f64,
        b: *const f64,
        beta: f64,
        c: *mut f64,
        rs_c: isize,
        cs_c: isize,
    ) {
        // Accumulators: 8 rows (2 regs) x 4 cols = 8 regs
        // c00, c01, c02, c03 (rows 0..3)
        // c10, c11, c12, c13 (rows 4..7)

        let mut c00: __m256d;
        let mut c01: __m256d;
        let mut c02: __m256d;
        let mut c03: __m256d;
        let mut c10: __m256d;
        let mut c11: __m256d;
        let mut c12: __m256d;
        let mut c13: __m256d;

        asm!(
            "vxorps {c00}, {c00}, {c00}", "vxorps {c01}, {c01}, {c01}",
            "vxorps {c02}, {c02}, {c02}", "vxorps {c03}, {c03}, {c03}",
            "vxorps {c10}, {c10}, {c10}", "vxorps {c11}, {c11}, {c11}",
            "vxorps {c12}, {c12}, {c12}", "vxorps {c13}, {c13}, {c13}",
            c00 = out(ymm_reg) c00, c01 = out(ymm_reg) c01,
            c02 = out(ymm_reg) c02, c03 = out(ymm_reg) c03,
            c10 = out(ymm_reg) c10, c11 = out(ymm_reg) c11,
            c12 = out(ymm_reg) c12, c13 = out(ymm_reg) c13,
        );

        let a_ptr = a;
        let b_ptr = b;
        let k_loop = kc;

        asm!(
            // ASM Loop (Fused K-loop, Unrolled 4x + Tail)
            "cmp {k}, 4",
            "jl 3f",

            "2:",

            // Iteration 0
            "vmovupd {a0}, [{a_ptr}]",
            "vmovupd {a1}, [{a_ptr} + 32]",
            "vbroadcastsd {b_val}, [{b_ptr}]",
            "vfmadd231pd {c00}, {a0}, {b_val}",
            "vfmadd231pd {c10}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 8]",
            "vfmadd231pd {c01}, {a0}, {b_val}",
            "vfmadd231pd {c11}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 16]",
            "vfmadd231pd {c02}, {a0}, {b_val}",
            "vfmadd231pd {c12}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 24]",
            "vfmadd231pd {c03}, {a0}, {b_val}",
            "vfmadd231pd {c13}, {a1}, {b_val}",

            // Iteration 1 (+64A, +32B)
            "vmovupd {a0}, [{a_ptr} + 64]",
            "vmovupd {a1}, [{a_ptr} + 96]",
            "vbroadcastsd {b_val}, [{b_ptr} + 32]",
            "vfmadd231pd {c00}, {a0}, {b_val}",
            "vfmadd231pd {c10}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 40]",
            "vfmadd231pd {c01}, {a0}, {b_val}",
            "vfmadd231pd {c11}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 48]",
            "vfmadd231pd {c02}, {a0}, {b_val}",
            "vfmadd231pd {c12}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 56]",
            "vfmadd231pd {c03}, {a0}, {b_val}",
            "vfmadd231pd {c13}, {a1}, {b_val}",

            // Iteration 2 (+128A, +64B)
            "vmovupd {a0}, [{a_ptr} + 128]",
            "vmovupd {a1}, [{a_ptr} + 160]",
            "vbroadcastsd {b_val}, [{b_ptr} + 64]",
            "vfmadd231pd {c00}, {a0}, {b_val}",
            "vfmadd231pd {c10}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 72]",
            "vfmadd231pd {c01}, {a0}, {b_val}",
            "vfmadd231pd {c11}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 80]",
            "vfmadd231pd {c02}, {a0}, {b_val}",
            "vfmadd231pd {c12}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 88]",
            "vfmadd231pd {c03}, {a0}, {b_val}",
            "vfmadd231pd {c13}, {a1}, {b_val}",

            // Iteration 3 (+192A, +96B)
            "vmovupd {a0}, [{a_ptr} + 192]",
            "vmovupd {a1}, [{a_ptr} + 224]",
            "vbroadcastsd {b_val}, [{b_ptr} + 96]",
            "vfmadd231pd {c00}, {a0}, {b_val}",
            "vfmadd231pd {c10}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 104]",
            "vfmadd231pd {c01}, {a0}, {b_val}",
            "vfmadd231pd {c11}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 112]",
            "vfmadd231pd {c02}, {a0}, {b_val}",
            "vfmadd231pd {c12}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 120]",
            "vfmadd231pd {c03}, {a0}, {b_val}",
            "vfmadd231pd {c13}, {a1}, {b_val}",

            // Update pointers (4x)
            "add {a_ptr}, 256",
            "add {b_ptr}, 128",
            "sub {k}, 4",
            "cmp {k}, 4",
            "jge 2b",

            "3:",
            "test {k}, {k}",
            "jz 5f",

            "4:", // Tail 1x
            "vmovupd {a0}, [{a_ptr}]",
            "vmovupd {a1}, [{a_ptr} + 32]",
            "vbroadcastsd {b_val}, [{b_ptr}]",
            "vfmadd231pd {c00}, {a0}, {b_val}",
            "vfmadd231pd {c10}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 8]",
            "vfmadd231pd {c01}, {a0}, {b_val}",
            "vfmadd231pd {c11}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 16]",
            "vfmadd231pd {c02}, {a0}, {b_val}",
            "vfmadd231pd {c12}, {a1}, {b_val}",
            "vbroadcastsd {b_val}, [{b_ptr} + 24]",
            "vfmadd231pd {c03}, {a0}, {b_val}",
            "vfmadd231pd {c13}, {a1}, {b_val}",

            "add {a_ptr}, 64",
            "add {b_ptr}, 32",
            "sub {k}, 1",
            "jnz 4b",

            "5:", // End

            a_ptr = inout(reg) a_ptr => _,
            b_ptr = inout(reg) b_ptr => _,
            k = inout(reg) k_loop => _,

            c00 = inout(ymm_reg) c00, c01 = inout(ymm_reg) c01,
            c02 = inout(ymm_reg) c02, c03 = inout(ymm_reg) c03,
            c10 = inout(ymm_reg) c10, c11 = inout(ymm_reg) c11,
            c12 = inout(ymm_reg) c12, c13 = inout(ymm_reg) c13,

            a0 = out(ymm_reg) _,
            a1 = out(ymm_reg) _,
            b_val = out(ymm_reg) _,
        );

        let alphav = _mm256_set1_pd(alpha);

        let store_col = |j: isize, c_alias: __m256d, c1_alias: __m256d| {
            let ptr0 = c.offset(j * cs_c);
            let ptr1 = c.offset(4 * rs_c + j * cs_c);

            if beta == 0.0 {
                _mm256_storeu_pd(ptr0, _mm256_mul_pd(c_alias, alphav));
                _mm256_storeu_pd(ptr1, _mm256_mul_pd(c1_alias, alphav));
            } else {
                let betav = _mm256_set1_pd(beta);
                let existing0 = _mm256_loadu_pd(ptr0);
                let existing1 = _mm256_loadu_pd(ptr1);

                let res0 = _mm256_fmadd_pd(c_alias, alphav, _mm256_mul_pd(existing0, betav));
                let res1 = _mm256_fmadd_pd(c1_alias, alphav, _mm256_mul_pd(existing1, betav));

                _mm256_storeu_pd(ptr0, res0);
                _mm256_storeu_pd(ptr1, res1);
            }
        };

        store_col(0, c00, c10);
        store_col(1, c01, c11);
        store_col(2, c02, c12);
        store_col(3, c03, c13);
    }
}
