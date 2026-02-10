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
