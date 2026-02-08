#[cfg(target_arch = "x86_64")]
use crate::core::ops::gemm::kernel::GemmKernel;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct FmaKernelF32;

#[cfg(target_arch = "x86_64")]
impl GemmKernel for FmaKernelF32 {
    type Elem = f32;
    const MR: usize = 16;
    const NR: usize = 6;

    #[target_feature(enable = "avx,fma")]
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
        let mut av = a;
        let mut bv = b;

        // Accumulators (16 rows x 6 cols) -> 2 ymm x 6 = 12 registers
        // rows 0..7 (ymm0..5), rows 8..15 (ymm6..11)
        let mut c00 = _mm256_setzero_ps(); let mut c01 = _mm256_setzero_ps();
        let mut c02 = _mm256_setzero_ps(); let mut c03 = _mm256_setzero_ps();
        let mut c04 = _mm256_setzero_ps(); let mut c05 = _mm256_setzero_ps();

        let mut c10 = _mm256_setzero_ps(); let mut c11 = _mm256_setzero_ps();
        let mut c12 = _mm256_setzero_ps(); let mut c13 = _mm256_setzero_ps();
        let mut c14 = _mm256_setzero_ps(); let mut c15 = _mm256_setzero_ps();

        for _ in 0..kc {
            let b0 = _mm256_broadcast_ss(&*bv.add(0));
            let b1 = _mm256_broadcast_ss(&*bv.add(1));
            let b2 = _mm256_broadcast_ss(&*bv.add(2));
            let b3 = _mm256_broadcast_ss(&*bv.add(3));
            let b4 = _mm256_broadcast_ss(&*bv.add(4));
            let b5 = _mm256_broadcast_ss(&*bv.add(5));

            let a0 = _mm256_loadu_ps(av.add(0)); // rows 0..7
            let a1 = _mm256_loadu_ps(av.add(8)); // rows 8..15

            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c10 = _mm256_fmadd_ps(a1, b0, c10);

            c01 = _mm256_fmadd_ps(a0, b1, c01);
            c11 = _mm256_fmadd_ps(a1, b1, c11);

            c02 = _mm256_fmadd_ps(a0, b2, c02);
            c12 = _mm256_fmadd_ps(a1, b2, c12);

            c03 = _mm256_fmadd_ps(a0, b3, c03);
            c13 = _mm256_fmadd_ps(a1, b3, c13);

            c04 = _mm256_fmadd_ps(a0, b4, c04);
            c14 = _mm256_fmadd_ps(a1, b4, c14);

            c05 = _mm256_fmadd_ps(a0, b5, c05);
            c15 = _mm256_fmadd_ps(a1, b5, c15);

            av = av.add(16);
            bv = bv.add(6);
        }

        // Store results with beta and alpha scaling
        // c = alpha * acc + beta * c
        
        let alphav = _mm256_set1_ps(alpha);
        
        // Helper to store: *ptr = alpha * val + beta * *ptr
        // If beta == 0, overwrite.
        
        let store_col = |j: isize, c_alias: __m256, c1_alias: __m256| {
             let ptr0 = c.offset(0 * rs_c + j * cs_c);
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

pub struct FmaKernelF64;

#[cfg(target_arch = "x86_64")]
impl GemmKernel for FmaKernelF64 {
    type Elem = f64;
    const MR: usize = 8;
    const NR: usize = 4;

    #[target_feature(enable = "avx,fma")]
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
        let mut av = a;
        let mut bv = b;

        // Accumulators (8 rows x 4 cols) -> 2 ymm x 4 = 8 registers
        // rows 0..3 (ymm0..3), rows 4..7 (ymm4..7)
        let mut c00 = _mm256_setzero_pd(); let mut c01 = _mm256_setzero_pd();
        let mut c02 = _mm256_setzero_pd(); let mut c03 = _mm256_setzero_pd();

        let mut c10 = _mm256_setzero_pd(); let mut c11 = _mm256_setzero_pd();
        let mut c12 = _mm256_setzero_pd(); let mut c13 = _mm256_setzero_pd();

        for _ in 0..kc {
            let b0 = _mm256_broadcast_sd(&*bv.add(0));
            let b1 = _mm256_broadcast_sd(&*bv.add(1));
            let b2 = _mm256_broadcast_sd(&*bv.add(2));
            let b3 = _mm256_broadcast_sd(&*bv.add(3));

            let a0 = _mm256_loadu_pd(av.add(0)); // rows 0..3
            let a1 = _mm256_loadu_pd(av.add(4)); // rows 4..7

            c00 = _mm256_fmadd_pd(a0, b0, c00);
            c10 = _mm256_fmadd_pd(a1, b0, c10);

            c01 = _mm256_fmadd_pd(a0, b1, c01);
            c11 = _mm256_fmadd_pd(a1, b1, c11);

            c02 = _mm256_fmadd_pd(a0, b2, c02);
            c12 = _mm256_fmadd_pd(a1, b2, c12);

            c03 = _mm256_fmadd_pd(a0, b3, c03);
            c13 = _mm256_fmadd_pd(a1, b3, c13);

            av = av.add(8);
            bv = bv.add(4);
        }

        let alphav = _mm256_set1_pd(alpha);
        
        let store_col = |j: isize, c_alias: __m256d, c1_alias: __m256d| {
             let ptr0 = c.offset(0 * rs_c + j * cs_c);
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
