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
    const MR: usize = 16; // Matching implementation (2 YMM per col)
                          // Wait, AVX2 has 16 YMM registers.
                          // If we use 6 columns (NR=6) and 16 rows (MR=16):
                          // 16 rows / 8 per reg = 2 regs per column.
                          // 2 regs * 6 cols = 12 accumulators.
                          // 4 regs for B (broadcasts) + A loads.
                          // Total 16 regs. This fits!
                          // Let's stick to MR=16, NR=6 for f32 to start, matching intrinsic version but with explicit asm.
    const NR: usize = 6;

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
        // Accumulators: 16 rows (2 regs) x 6 cols = 12 regs
        // c00, c01, c02, c03, c04, c05 (rows 0..7)
        // c10, c11, c12, c13, c14, c15 (rows 8..15)

        let mut c00: __m256;
        let mut c01: __m256;
        let mut c02: __m256;
        let mut c03: __m256;
        let mut c04: __m256;
        let mut c05: __m256;
        let mut c10: __m256;
        let mut c11: __m256;
        let mut c12: __m256;
        let mut c13: __m256;
        let mut c14: __m256;
        let mut c15: __m256;

        // Initialize accumulators to zero
        asm!(
            "vxorps {c00}, {c00}, {c00}", "vxorps {c01}, {c01}, {c01}", "vxorps {c02}, {c02}, {c02}",
            "vxorps {c03}, {c03}, {c03}", "vxorps {c04}, {c04}, {c04}", "vxorps {c05}, {c05}, {c05}",
            "vxorps {c10}, {c10}, {c10}", "vxorps {c11}, {c11}, {c11}", "vxorps {c12}, {c12}, {c12}",
            "vxorps {c13}, {c13}, {c13}", "vxorps {c14}, {c14}, {c14}", "vxorps {c15}, {c15}, {c15}",
            c00 = out(ymm_reg) c00, c01 = out(ymm_reg) c01, c02 = out(ymm_reg) c02,
            c03 = out(ymm_reg) c03, c04 = out(ymm_reg) c04, c05 = out(ymm_reg) c05,
            c10 = out(ymm_reg) c10, c11 = out(ymm_reg) c11, c12 = out(ymm_reg) c12,
            c13 = out(ymm_reg) c13, c14 = out(ymm_reg) c14, c15 = out(ymm_reg) c15,
        );

        let mut a_ptr = a;
        let mut b_ptr = b;
        let mut k_loop = kc;

        // Pointers to help asm
        // asm loop
        while k_loop > 0 {
            asm!(
                // Load A columns (16 rows) -> 2 regs (a0, a1)
                // A is [16 x KC] packed.
                "vmovups {a0}, [{a_ptr}]",
                "vmovups {a1}, [{a_ptr} + 32]",

                // Load B scalars, broadcast and FMA
                // B is [KC x 6] packed.
                // Col 0
                "vbroadcastss {b_val}, [{b_ptr}]",
                "vfmadd231ps {c00}, {a0}, {b_val}",
                "vfmadd231ps {c10}, {a1}, {b_val}",

                // Col 1
                "vbroadcastss {b_val}, [{b_ptr} + 4]",
                "vfmadd231ps {c01}, {a0}, {b_val}",
                "vfmadd231ps {c11}, {a1}, {b_val}",

                // Col 2
                "vbroadcastss {b_val}, [{b_ptr} + 8]",
                "vfmadd231ps {c02}, {a0}, {b_val}",
                "vfmadd231ps {c12}, {a1}, {b_val}",

                // Col 3
                "vbroadcastss {b_val}, [{b_ptr} + 12]",
                "vfmadd231ps {c03}, {a0}, {b_val}",
                "vfmadd231ps {c13}, {a1}, {b_val}",

                // Col 4
                "vbroadcastss {b_val}, [{b_ptr} + 16]",
                "vfmadd231ps {c04}, {a0}, {b_val}",
                "vfmadd231ps {c14}, {a1}, {b_val}",

                // Col 5
                "vbroadcastss {b_val}, [{b_ptr} + 20]",
                "vfmadd231ps {c05}, {a0}, {b_val}",
                "vfmadd231ps {c15}, {a1}, {b_val}",

                // Update pointers
                // a += 16 * 4 bytes = 64
                // b += 6 * 4 bytes = 24

                a_ptr = in(reg) a_ptr,
                b_ptr = in(reg) b_ptr,

                c00 = inout(ymm_reg) c00, c01 = inout(ymm_reg) c01, c02 = inout(ymm_reg) c02,
                c03 = inout(ymm_reg) c03, c04 = inout(ymm_reg) c04, c05 = inout(ymm_reg) c05,
                c10 = inout(ymm_reg) c10, c11 = inout(ymm_reg) c11, c12 = inout(ymm_reg) c12,
                c13 = inout(ymm_reg) c13, c14 = inout(ymm_reg) c14, c15 = inout(ymm_reg) c15,

                a0 = out(ymm_reg) _,
                a1 = out(ymm_reg) _,
                b_val = out(ymm_reg) _,
            );

            a_ptr = a_ptr.add(16);
            b_ptr = b_ptr.add(6);
            k_loop -= 1;
        }

        // Store results
        // Use intrinsic logic for update: C = alpha*Acc + beta*C
        // We reuse the FMA intrinsics structure but use the computed accumulators

        let alphav = _mm256_set1_ps(alpha);

        // Helper to store: *ptr = alpha * val + beta * *ptr
        let store_col = |j: isize, c_alias: __m256, c1_alias: __m256| {
            let ptr0 = c.offset(j * cs_c);
            let ptr1 = c.offset(8 * rs_c + j * cs_c);

            if beta == 0.0 {
                _mm256_storeu_ps(ptr0, _mm256_mul_ps(c_alias, alphav));
                _mm256_storeu_ps(ptr1, _mm256_mul_ps(c1_alias, alphav));
            } else {
                let betav = _mm256_set1_ps(beta);
                let existing0 = _mm256_loadu_ps(ptr0);
                let existing1 = _mm256_loadu_ps(ptr1);

                let res0 = _mm256_fmadd_ps(c_alias, alphav, _mm256_mul_ps(existing0, betav));
                let res1 = _mm256_fmadd_ps(c1_alias, alphav, _mm256_mul_ps(existing1, betav));

                _mm256_storeu_ps(ptr0, res0);
                _mm256_storeu_ps(ptr1, res1);
            }
        };

        store_col(0, c00, c10);
        store_col(1, c01, c11);
        store_col(2, c02, c12);
        store_col(3, c03, c13);
        store_col(4, c04, c14);
        store_col(5, c05, c15);
    }
}

pub struct AsmFmaKernelF64;

#[cfg(target_arch = "x86_64")]
impl GemmKernel for AsmFmaKernelF64 {
    type Elem = f64;
    const MR: usize = 8;
    const NR: usize = 4;

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

        let mut a_ptr = a;
        let mut b_ptr = b;
        let mut k_loop = kc;

        while k_loop > 0 {
            asm!(
                // Load A columns (8 rows) -> 2 regs (a0, a1)
                // A is [8 x KC] packed.
                "vmovupd {a0}, [{a_ptr}]",
                "vmovupd {a1}, [{a_ptr} + 32]",

                // Load B scalars, broadcast and FWA
                // B is [KC x 4] packed
                // Col 0
                "vbroadcastsd {b_val}, [{b_ptr}]",
                "vfmadd231pd {c00}, {a0}, {b_val}",
                "vfmadd231pd {c10}, {a1}, {b_val}",

                // Col 1
                "vbroadcastsd {b_val}, [{b_ptr} + 8]",
                "vfmadd231pd {c01}, {a0}, {b_val}",
                "vfmadd231pd {c11}, {a1}, {b_val}",

                // Col 2
                "vbroadcastsd {b_val}, [{b_ptr} + 16]",
                "vfmadd231pd {c02}, {a0}, {b_val}",
                "vfmadd231pd {c12}, {a1}, {b_val}",

                // Col 3
                "vbroadcastsd {b_val}, [{b_ptr} + 24]",
                "vfmadd231pd {c03}, {a0}, {b_val}",
                "vfmadd231pd {c13}, {a1}, {b_val}",

                // Update pointers
                // a += 8 * 8 bytes = 64
                // b += 4 * 8 bytes = 32

                a_ptr = in(reg) a_ptr,
                b_ptr = in(reg) b_ptr,

                c00 = inout(ymm_reg) c00, c01 = inout(ymm_reg) c01,
                c02 = inout(ymm_reg) c02, c03 = inout(ymm_reg) c03,
                c10 = inout(ymm_reg) c10, c11 = inout(ymm_reg) c11,
                c12 = inout(ymm_reg) c12, c13 = inout(ymm_reg) c13,

                a0 = out(ymm_reg) _,
                a1 = out(ymm_reg) _,
                b_val = out(ymm_reg) _,
            );
            a_ptr = a_ptr.add(8);
            b_ptr = b_ptr.add(4);
            k_loop -= 1;
        }

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
