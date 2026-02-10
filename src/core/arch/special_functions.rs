//! SIMD approximation for elementary functions.
//! Based on Cephes Mathematical Library (http://www.netlib.org/cephes/).
//!
//! Currently supports float (f32) SIMD packets using SSE.
//! Implementations are now directly on the Packet trait in mod.rs or via extension.
//! This file contains the implementation logic.

// We need to implement the methods for specific packets.
// Since the methods are on Packet trait, we need to implement them in the impl block of Packet for SsePacketF32 etc.
// But those impl blocks are in mod.rs.
// We can't implement trait methods in a different module unless we use an extension trait,
// OR we just put the logic here and call it from mod.rs.
//
// Let's use the second approach: Keep the logic here as standalone functions or a helper trait,
// and call them from mod.rs.
// Actually, to avoid circular deps or complex organization, let's just move the implementation to mod.rs
// or keep this file as a module that `mod.rs` uses.
//
// Better: Define a macro or just public functions that accept the packet buffer.
//
// Let's try to keep the `PacketMath` trait but as a private/internal helper if needed,
// OR just implement `Packet` methods in `mod.rs` by importing logic from here.
//
// Wait, `mod.rs` defines `SsePacketF32`. `special_functions.rs` is a child module.
// We can implement `Packet<f32>` for `SsePacketF32` in `mod.rs`.
// But the code is large.
//
// Best approach:
// Define the logic in `special_functions.rs` as generic functions or specific functions accepting `__m128`.
// Then call them from `mod.rs`.

use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub mod sse_f32 {
    use super::*;

    const FOPI: f32 = 1.273_239_5;
    const DP1: f32 = -0.78515625;
    const DP2: f32 = -2.418_756_5e-4;
    const DP3: f32 = -3.774_895e-8;
    const SINCOF_P0: f32 = -1.951_529_6E-4;
    const SINCOF_P1: f32 = 8.332_161E-3;
    const SINCOF_P2: f32 = -1.666_665_5E-1;
    const COSCOF_P0: f32 = 2.443_315_7E-5;
    const COSCOF_P1: f32 = -1.388_731_6E-3;
    const COSCOF_P2: f32 = 4.166_664_6E-2;

    const EXP_HI: f32 = 88.376_26;
    const EXP_LO: f32 = -88.376_26;
    const LOG2EF: f32 = std::f32::consts::LOG2_E;
    const EXP_C1: f32 = 0.693_359_4;
    const EXP_C2: f32 = -2.121_944_4e-4;
    const EXP_P0: f32 = 1.987_569_1E-4;
    const EXP_P1: f32 = 1.398_199_9E-3;
    const EXP_P2: f32 = 8.333_452E-3;
    const EXP_P3: f32 = 4.166_579_6E-2;
    const EXP_P4: f32 = 1.666_666_6E-1;
    const EXP_P5: f32 = 5E-1;

    const LOG_P0: f32 = 7.037_683_6E-2;
    const LOG_P1: f32 = -1.151_461E-1;
    const LOG_P2: f32 = 1.167_699_84E-1;
    const LOG_P3: f32 = -1.242_014_1E-1;
    const LOG_P4: f32 = 1.424_932_3E-1;
    const LOG_P5: f32 = -1.666_805_7E-1;
    const LOG_P6: f32 = 2.000_071_4E-1;
    const LOG_P7: f32 = -2.499_999_4E-1;
    const LOG_P8: f32 = 3.333_333E-1;
    const LOG_Q1: f32 = -2.121_944_4e-4;
    const LOG_Q2: f32 = 0.693_359_4;

    unsafe fn set1(v: f32) -> __m128 {
        unsafe { _mm_set1_ps(v) }
    }

    pub unsafe fn psin(x_in: __m128) -> __m128 {
        unsafe {
            let x = x_in;
            let sign_bit = _mm_castsi128_ps(_mm_set1_epi32(0x80000000u32 as i32));
            let sign_mask = _mm_and_ps(sign_bit, x);
            let abs_x = _mm_andnot_ps(sign_bit, x);

            let y = _mm_mul_ps(abs_x, set1(FOPI));
            let emm2 = _mm_cvttps_epi32(_mm_add_ps(y, set1(0.5)));
            let y = _mm_cvtepi32_ps(emm2);

            let emm2_and_1 = _mm_and_si128(emm2, _mm_set1_epi32(1));
            let poly_sign_mask = _mm_cmpeq_epi32(emm2_and_1, _mm_setzero_si128());
            let poly_sign = _mm_castsi128_ps(poly_sign_mask);

            let x = _mm_add_ps(
                _mm_add_ps(
                    _mm_add_ps(abs_x, _mm_mul_ps(y, set1(DP1))),
                    _mm_mul_ps(y, set1(DP2)),
                ),
                _mm_mul_ps(y, set1(DP3)),
            );

            let z = _mm_mul_ps(x, x);

            let mut v = set1(SINCOF_P0);
            v = _mm_add_ps(_mm_mul_ps(v, z), set1(SINCOF_P1));
            v = _mm_add_ps(_mm_mul_ps(v, z), set1(SINCOF_P2));
            v = _mm_mul_ps(_mm_mul_ps(v, z), x);
            v = _mm_add_ps(v, x);

            let mut c = set1(COSCOF_P0);
            c = _mm_add_ps(_mm_mul_ps(c, z), set1(COSCOF_P1));
            c = _mm_add_ps(_mm_mul_ps(c, z), set1(COSCOF_P2));
            c = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(c, z), z), _mm_mul_ps(z, set1(-0.5)));
            c = _mm_add_ps(c, set1(1.0));

            let select_sin = poly_sign;
            let result = _mm_or_ps(_mm_and_ps(select_sin, v), _mm_andnot_ps(select_sin, c));

            let emm2_and_2 = _mm_and_si128(emm2, _mm_set1_epi32(2));
            let sign_flip_mask = _mm_castsi128_ps(_mm_cmpeq_epi32(emm2_and_2, _mm_setzero_si128()));
            let combined_sign =
                _mm_xor_ps(sign_mask, _mm_andnot_ps(sign_flip_mask, _mm_set1_ps(-0.0)));

            _mm_xor_ps(result, combined_sign)
        }
    }

    pub unsafe fn pcos(x_in: __m128) -> __m128 {
        let half_pi = unsafe { set1(1.570_796_4) };
        unsafe {
            let x_plus_half_pi = _mm_add_ps(x_in, half_pi);
            psin(x_plus_half_pi)
        }
    }

    pub unsafe fn pexp(x_in: __m128) -> __m128 {
        unsafe {
            let x = x_in;
            let min_val = set1(EXP_LO);
            let max_val = set1(EXP_HI);

            let x = _mm_min_ps(x, max_val);
            let x = _mm_max_ps(x, min_val);

            let fx = _mm_add_ps(_mm_mul_ps(x, set1(LOG2EF)), set1(0.5));
            let mut emm0 = _mm_cvttps_epi32(fx);
            let tmp = _mm_cvtepi32_ps(emm0);

            let z = _mm_mul_ps(tmp, set1(EXP_C1));
            let x = _mm_sub_ps(x, z);
            let z = _mm_mul_ps(tmp, set1(EXP_C2));
            let x = _mm_sub_ps(x, z);

            let z = _mm_mul_ps(x, x);
            let mut y = set1(EXP_P0);
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(EXP_P1));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(EXP_P2));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(EXP_P3));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(EXP_P4));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(EXP_P5));
            y = _mm_add_ps(_mm_mul_ps(y, z), x);
            y = _mm_add_ps(y, set1(1.0));

            emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(0x7f));
            emm0 = _mm_slli_epi32(emm0, 23);
            let pow2n = _mm_castsi128_ps(emm0);

            _mm_mul_ps(y, pow2n)
        }
    }

    pub unsafe fn plog(x_in: __m128) -> __m128 {
        unsafe {
            let mut x = x_in;
            let invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());
            x = _mm_max_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x00800000)));

            let emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
            let sub_emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
            let e = _mm_cvtepi32_ps(sub_emm0);

            let mut emm0 = _mm_castps_si128(x);
            emm0 = _mm_and_si128(emm0, _mm_set1_epi32(0x7fffff));
            emm0 = _mm_or_si128(emm0, _mm_set1_epi32(0x3f000000));
            x = _mm_castsi128_ps(emm0);

            let mask = _mm_cmplt_ps(x, set1(0.707_106_77));
            let tmp = _mm_and_ps(x, mask);
            x = _mm_sub_ps(x, set1(1.0));
            let e = _mm_sub_ps(e, _mm_and_ps(set1(1.0), mask));
            x = _mm_add_ps(x, tmp);

            let z = _mm_mul_ps(x, x);
            let mut y = set1(LOG_P0);
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(LOG_P1));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(LOG_P2));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(LOG_P3));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(LOG_P4));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(LOG_P5));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(LOG_P6));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(LOG_P7));
            y = _mm_add_ps(_mm_mul_ps(y, x), set1(LOG_P8));
            y = _mm_mul_ps(_mm_mul_ps(y, x), z);

            y = _mm_add_ps(y, _mm_mul_ps(e, set1(LOG_Q1)));
            y = _mm_sub_ps(y, _mm_mul_ps(z, set1(0.5)));
            x = _mm_add_ps(x, y);
            x = _mm_add_ps(x, _mm_mul_ps(e, set1(LOG_Q2)));

            _mm_or_ps(x, invalid_mask)
        }
    }
}
