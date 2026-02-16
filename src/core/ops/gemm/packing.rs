use crate::core::scalar::Scalar;

// Pack LHS (A) from (MC x KC) into packed buffer (MC-major strips of MR rows)
// A: (m x k) matrix
// Packed: contiguous
// For each micropanel of width MR:
//   For each k:
//     Store MR elements
/// # Safety
/// Pointers must be valid.
pub unsafe fn pack_lhs<T: Scalar + Copy>(
    mr: usize,
    kc: usize,
    mc: usize,
    a: *const T,
    rs: isize,
    cs: isize,
    packed: *mut T,
) {
    let mut packed_ptr = packed;

    // Iterate over micropanels of height MR
    for i in (0..mc).step_by(mr) {
        let mr_eff = std::cmp::min(mr, mc - i);

        for k in 0..kc {
            // software prefetch for A (next lines)
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                _mm_prefetch(
                    a.offset((i as isize + mr as isize) * rs + (k as isize) * cs) as *const i8,
                    _MM_HINT_T0,
                );
            }

            for r in 0..mr_eff {
                // Access A(i+r, k) = a[ (i+r)*rs + k*cs ]
                let val = *a.offset((i as isize + r as isize) * rs + (k as isize) * cs);
                *packed_ptr.add(r) = val;
            }
            // If mr_eff < mr, pad with zeros
            for r in mr_eff..mr {
                *packed_ptr.add(r) = T::default();
            }
            packed_ptr = packed_ptr.add(mr);
        }
    }
}

// Pack RHS (B) from (KC x NC) into packed buffer (KC-major strips of NR columns)
// B: (k x n) matrix
// Packed: contiguous
// For each micropanel of NR cols:
//   For each k:
//     Store NR elements
/// # Safety
/// Pointers must be valid.
pub unsafe fn pack_rhs<T: Scalar + Copy>(
    nr: usize,
    kc: usize,
    nc: usize,
    b: *const T,
    rs: isize,
    cs: isize,
    packed: *mut T,
) {
    let mut packed_ptr = packed;

    // Iterate over micropanels of width NR
    for j in (0..nc).step_by(nr) {
        let nr_eff = std::cmp::min(nr, nc - j);

        for k in 0..kc {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                _mm_prefetch(
                    b.offset((k as isize) * rs + (j as isize + nr as isize) * cs) as *const i8,
                    _MM_HINT_T0,
                );
            }

            for c in 0..nr_eff {
                // Access B(k, j+c) = b[ k*rs + (j+c)*cs ]
                let val = *b.offset((k as isize) * rs + (j as isize + c as isize) * cs);
                *packed_ptr.add(c) = val;
            }
            // If nr_eff < nr, pad
            for c in nr_eff..nr {
                *packed_ptr.add(c) = T::default();
            }
            packed_ptr = packed_ptr.add(nr);
        }
    }
}

/// # Safety
/// AVX2 must be enabled. Pointers valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "avx2", enable = "fma")]
pub unsafe fn pack_lhs_f32_avx(
    mr: usize,
    kc: usize,
    mc: usize,
    a: *const f32,
    rs: isize,
    cs: isize,
    packed: *mut f32,
) {
    // If layout matches optimized path
    if mr == 24 && rs == 1 {
        let mut packed_ptr = packed;

        for i in (0..mc).step_by(mr) {
            let mr_eff = std::cmp::min(mr, mc - i);

            for k in 0..kc {
                let a_ptr_k = a.offset((i as isize) + (k as isize) * cs);

                if mr_eff == 24 {
                    // Fast path: Memcpy 24 floats
                    std::ptr::copy_nonoverlapping(a_ptr_k, packed_ptr, 24);
                } else {
                    // Partial pack
                    for r in 0..mr_eff {
                        *packed_ptr.add(r) = *a_ptr_k.add(r);
                    }
                    for r in mr_eff..mr {
                        *packed_ptr.add(r) = 0.0;
                    }
                }
                packed_ptr = packed_ptr.add(mr);
            }
        }
        return;
    }

    // Fallback to generic `pack_lhs` if parameters don't match
    pack_lhs(mr, kc, mc, a, rs, cs, packed);
}

/// # Safety
/// AVX2 enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "avx2", enable = "fma")]
pub unsafe fn pack_rhs_f32_avx(
    nr: usize,
    kc: usize,
    nc: usize,
    b: *const f32,
    rs: isize,
    cs: isize,
    packed: *mut f32,
) {
    // Optimized for NR=4, Col-Major B (rs=1).
    if nr == 4 && rs == 1 {
        use std::arch::x86_64::*;
        let mut packed_ptr = packed;

        for j in (0..nc).step_by(nr) {
            let nr_eff = std::cmp::min(nr, nc - j);

            if nr_eff == 4 {
                let mut k = 0;
                // Unroll K=4
                while k + 4 <= kc {
                    let b0_ptr = b.offset((k as isize) + (j as isize) * cs);
                    let b1_ptr = b.offset((k as isize) + (j as isize + 1) * cs);
                    let b2_ptr = b.offset((k as isize) + (j as isize + 2) * cs);
                    let b3_ptr = b.offset((k as isize) + (j as isize + 3) * cs);

                    let mut v0 = _mm_loadu_ps(b0_ptr);
                    let mut v1 = _mm_loadu_ps(b1_ptr);
                    let mut v2 = _mm_loadu_ps(b2_ptr);
                    let mut v3 = _mm_loadu_ps(b3_ptr);

                    // 4x4 Transpose
                    _MM_TRANSPOSE4_PS(&mut v0, &mut v1, &mut v2, &mut v3);

                    _mm_storeu_ps(packed_ptr, v0);
                    packed_ptr = packed_ptr.add(4);

                    _mm_storeu_ps(packed_ptr, v1);
                    packed_ptr = packed_ptr.add(4);

                    _mm_storeu_ps(packed_ptr, v2);
                    packed_ptr = packed_ptr.add(4);

                    _mm_storeu_ps(packed_ptr, v3);
                    packed_ptr = packed_ptr.add(4);

                    k += 4;
                }

                for kk in k..kc {
                    for c in 0..4 {
                        *packed_ptr.add(c) =
                            *b.offset((kk as isize) + (j as isize + c as isize) * cs);
                    }
                    packed_ptr = packed_ptr.add(4);
                }
            } else {
                for k in 0..kc {
                    for c in 0..nr_eff {
                        *packed_ptr.add(c) =
                            *b.offset((k as isize) + (j as isize + c as isize) * cs);
                    }
                    for c in nr_eff..nr {
                        *packed_ptr.add(c) = 0.0;
                    }
                    packed_ptr = packed_ptr.add(nr);
                }
            }
        }
        return;
    }

    // Fallback
    pack_rhs(nr, kc, nc, b, rs, cs, packed);
}

/// # Safety
/// AVX2 must be enabled. Pointers valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "avx2", enable = "fma")]
pub unsafe fn pack_lhs_f64_avx(
    mr: usize,
    kc: usize,
    mc: usize,
    a: *const f64,
    rs: isize,
    cs: isize,
    packed: *mut f64,
) {
    // If layout matches optimized path (Col-Major A, rs=1)
    if mr == 8 && rs == 1 {
        let mut packed_ptr = packed;

        for i in (0..mc).step_by(mr) {
            let mr_eff = std::cmp::min(mr, mc - i);

            for k in 0..kc {
                let a_ptr_k = a.offset((i as isize) + (k as isize) * cs);

                if mr_eff == 8 {
                    // Fast path: Memcpy 8 doubles (64 bytes)
                    std::ptr::copy_nonoverlapping(a_ptr_k, packed_ptr, 8);
                } else {
                    // Partial pack
                    for r in 0..mr_eff {
                        *packed_ptr.add(r) = *a_ptr_k.add(r);
                    }
                    for r in mr_eff..mr {
                        *packed_ptr.add(r) = 0.0;
                    }
                }
                packed_ptr = packed_ptr.add(mr);
            }
        }
        return;
    }

    // Fallback to generic `pack_lhs` if parameters don't match
    pack_lhs(mr, kc, mc, a, rs, cs, packed);
}

/// # Safety
/// AVX2 enabled.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "avx2", enable = "fma")]
pub unsafe fn pack_rhs_f64_avx(
    nr: usize,
    kc: usize,
    nc: usize,
    b: *const f64,
    rs: isize,
    cs: isize,
    packed: *mut f64,
) {
    // Optimized for NR=4, Col-Major B (rs=1).
    if nr == 4 && rs == 1 {
        use std::arch::x86_64::*;
        let mut packed_ptr = packed;

        for j in (0..nc).step_by(nr) {
            let nr_eff = std::cmp::min(nr, nc - j);

            if nr_eff == 4 {
                let mut k = 0;
                // Unroll K=4
                while k + 4 <= kc {
                    let b0_ptr = b.offset((k as isize) + (j as isize) * cs);
                    let b1_ptr = b.offset((k as isize) + (j as isize + 1) * cs);
                    let b2_ptr = b.offset((k as isize) + (j as isize + 2) * cs);
                    let b3_ptr = b.offset((k as isize) + (j as isize + 3) * cs);

                    let v0 = _mm256_loadu_pd(b0_ptr);
                    let v1 = _mm256_loadu_pd(b1_ptr);
                    let v2 = _mm256_loadu_pd(b2_ptr);
                    let v3 = _mm256_loadu_pd(b3_ptr);

                    // Transpose 4x4 doubles
                    // v0: a0 a1 a2 a3
                    // v1: b0 b1 b2 b3
                    // v2: c0 c1 c2 c3
                    // v3: d0 d1 d2 d3

                    // t0: a0 b0 a1 b1
                    // t1: a2 b2 a3 b3
                    let t0 = _mm256_unpacklo_pd(v0, v1);
                    let t1 = _mm256_unpackhi_pd(v0, v1);

                    // t2: c0 d0 c1 d1
                    // t3: c2 d2 c3 d3
                    let t2 = _mm256_unpacklo_pd(v2, v3);
                    let t3 = _mm256_unpackhi_pd(v2, v3);

                    // r0: a0 b0 c0 d0 (permute 128)
                    let r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                    // r1: a1 b1 c1 d1
                    let r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                    // r2: a2 b2 c2 d2
                    let r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                    // r3: a3 b3 c3 d3
                    let r3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                    _mm256_storeu_pd(packed_ptr, r0);
                    packed_ptr = packed_ptr.add(4);

                    _mm256_storeu_pd(packed_ptr, r1);
                    packed_ptr = packed_ptr.add(4);

                    _mm256_storeu_pd(packed_ptr, r2);
                    packed_ptr = packed_ptr.add(4);

                    _mm256_storeu_pd(packed_ptr, r3);
                    packed_ptr = packed_ptr.add(4);

                    k += 4;
                }

                for kk in k..kc {
                    for c in 0..4 {
                        *packed_ptr.add(c) =
                            *b.offset((kk as isize) + (j as isize + c as isize) * cs);
                    }
                    packed_ptr = packed_ptr.add(4);
                }
            } else {
                for k in 0..kc {
                    for c in 0..nr_eff {
                        *packed_ptr.add(c) =
                            *b.offset((k as isize) + (j as isize + c as isize) * cs);
                    }
                    for c in nr_eff..nr {
                        *packed_ptr.add(c) = 0.0;
                    }
                    packed_ptr = packed_ptr.add(nr);
                }
            }
        }
        return;
    }

    // Fallback
    pack_rhs(nr, kc, nc, b, rs, cs, packed);
}
