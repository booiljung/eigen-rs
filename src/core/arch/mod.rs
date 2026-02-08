//! Architecture-specific SIMD abstractions.
//! Defines the Packet trait and specialization for different SIMD levels.

use crate::core::scalar::Scalar;
use std::ops::{Add, Sub, Mul};

/// Trait representing a SIMD packet of a specific scalar type.
pub trait Packet<T: Scalar>: 
    Copy + 
    Add<Output = Self> + 
    Sub<Output = Self> + 
    Mul<Output = Self> 
{
    /// Number of scalars in this packet.
    const SIZE: usize;

    /// Loads a packet from a memory pointer.
    /// 
    /// # Safety
    /// The pointer must be valid for reading `SIZE` elements.
    unsafe fn load(ptr: *const T) -> Self;

    /// Stores a packet to a memory pointer.
    /// 
    /// # Safety
    /// The pointer must be valid for writing `SIZE` elements.
    unsafe fn store(self, ptr: *mut T);

    /// Sets all elements in the packet to a single scalar value.
    fn set1(val: T) -> Self;

    /// Fused Add-Multiply: self = self + (a * b)
    /// Fused Add-Multiply: self = self + (a * b)
    fn fused_add_mul(&mut self, a: Self, b: Self);

    // Special Functions (default unimplemented)
    fn psin(self) -> Self { unimplemented!("psin not implemented for this packet") }
    fn pcos(self) -> Self { unimplemented!("pcos not implemented for this packet") }
    fn pexp(self) -> Self { unimplemented!("pexp not implemented for this packet") }
    fn plog(self) -> Self { unimplemented!("plog not implemented for this packet") }
    fn psqrt(self) -> Self { unimplemented!("psqrt not implemented for this packet") }
    fn prsqrt(self) -> Self { unimplemented!("prsqrt not implemented for this packet") }
    fn pabs(self) -> Self { unimplemented!("pabs not implemented for this packet") }
}

/// Fallback scalar packet (SIMD size 1).
#[derive(Debug, Clone, Copy)]
pub struct ScalarPacket<T: Scalar>(pub T);

impl<T: Scalar> Add for ScalarPacket<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Self(self.0 + rhs.0) }
}

impl<T: Scalar> Sub for ScalarPacket<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Self(self.0 - rhs.0) }
}

impl<T: Scalar> Mul for ScalarPacket<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { Self(self.0 * rhs.0) }
}

impl<T: Scalar> Packet<T> for ScalarPacket<T> {
    const SIZE: usize = 1;

    unsafe fn load(ptr: *const T) -> Self {
        unsafe { Self(*ptr) }
    }

    unsafe fn store(self, ptr: *mut T) {
        unsafe { *ptr = self.0; }
    }

    fn set1(val: T) -> Self {
        Self(val)
    }

    fn fused_add_mul(&mut self, a: Self, b: Self) {
        self.0 += a.0 * b.0;
    }

    fn psin(self) -> Self { Self(self.0.sin()) }
    fn pcos(self) -> Self { Self(self.0.cos()) }
    fn pexp(self) -> Self { Self(self.0.exp()) }
    fn plog(self) -> Self { Self(self.0.ln()) }
    fn psqrt(self) -> Self { Self(self.0.sqrt()) }
    fn prsqrt(self) -> Self { Self(T::from_f64(1.0) / self.0.sqrt()) }
    fn pabs(self) -> Self { Self(self.0.abs()) }
}

// x86_64 Specializations
#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    #[derive(Debug, Clone, Copy)]
    pub struct SsePacketF32(pub __m128);

    impl Add for SsePacketF32 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self { unsafe { Self(_mm_add_ps(self.0, rhs.0)) } }
    }
    impl Sub for SsePacketF32 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm_sub_ps(self.0, rhs.0)) } }
    }
    impl Mul for SsePacketF32 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self { unsafe { Self(_mm_mul_ps(self.0, rhs.0)) } }
    }

    impl Packet<f32> for SsePacketF32 {
        const SIZE: usize = 4;
        unsafe fn load(ptr: *const f32) -> Self { unsafe { Self(_mm_loadu_ps(ptr)) } }
        unsafe fn store(self, ptr: *mut f32) { unsafe { _mm_storeu_ps(ptr, self.0); } }
        fn set1(val: f32) -> Self { unsafe { Self(_mm_set1_ps(val)) } }
        fn fused_add_mul(&mut self, a: Self, b: Self) {
            unsafe { self.0 = _mm_add_ps(self.0, _mm_mul_ps(a.0, b.0)); }
        }

        fn psin(self) -> Self { unsafe { Self(special_functions::sse_f32::psin(self.0)) } }
        fn pcos(self) -> Self { unsafe { Self(special_functions::sse_f32::pcos(self.0)) } }
        fn pexp(self) -> Self { unsafe { Self(special_functions::sse_f32::pexp(self.0)) } }
        fn plog(self) -> Self { unsafe { Self(special_functions::sse_f32::plog(self.0)) } }
        fn psqrt(self) -> Self { unsafe { Self(_mm_sqrt_ps(self.0)) } }
        fn prsqrt(self) -> Self { unsafe { Self(_mm_rsqrt_ps(self.0)) } }
        fn pabs(self) -> Self {
             unsafe {
                 let sign_bit = _mm_castsi128_ps(_mm_set1_epi32(0x80000000u32 as i32));
                 Self(_mm_andnot_ps(sign_bit, self.0))
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct AvxPacketF32(pub __m256);

    impl Add for AvxPacketF32 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self { unsafe { Self(_mm256_add_ps(self.0, rhs.0)) } }
    }
    impl Sub for AvxPacketF32 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm256_sub_ps(self.0, rhs.0)) } }
    }
    impl Mul for AvxPacketF32 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self { unsafe { Self(_mm256_mul_ps(self.0, rhs.0)) } }
    }

    impl Packet<f32> for AvxPacketF32 {
        const SIZE: usize = 8;
        unsafe fn load(ptr: *const f32) -> Self { unsafe { Self(_mm256_loadu_ps(ptr)) } }
        unsafe fn store(self, ptr: *mut f32) { unsafe { _mm256_storeu_ps(ptr, self.0); } }
        fn set1(val: f32) -> Self { unsafe { Self(_mm256_set1_ps(val)) } }
        fn fused_add_mul(&mut self, a: Self, b: Self) {
            unsafe {
                self.0 = _mm256_add_ps(self.0, _mm256_mul_ps(a.0, b.0));
            }
        }

        fn psin(self) -> Self { unimplemented!("AVX psin") }
        fn pcos(self) -> Self { unimplemented!("AVX pcos") }
        fn pexp(self) -> Self { unimplemented!("AVX pexp") }
        fn plog(self) -> Self { unimplemented!("AVX plog") }
        fn psqrt(self) -> Self { unsafe { Self(_mm256_sqrt_ps(self.0)) } }
        fn prsqrt(self) -> Self { unsafe { Self(_mm256_rsqrt_ps(self.0)) } }
        fn pabs(self) -> Self { 
            unsafe {
                let sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u32 as i32));
                Self(_mm256_andnot_ps(sign_bit, self.0))
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct AvxFmaPacketF32(pub __m256);

    impl Add for AvxFmaPacketF32 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self { unsafe { Self(_mm256_add_ps(self.0, rhs.0)) } }
    }
    impl Sub for AvxFmaPacketF32 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm256_sub_ps(self.0, rhs.0)) } }
    }
    impl Mul for AvxFmaPacketF32 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self { unsafe { Self(_mm256_mul_ps(self.0, rhs.0)) } }
    }

    impl Packet<f32> for AvxFmaPacketF32 {
        const SIZE: usize = 8;
        unsafe fn load(ptr: *const f32) -> Self { unsafe { Self(_mm256_loadu_ps(ptr)) } }
        unsafe fn store(self, ptr: *mut f32) { unsafe { _mm256_storeu_ps(ptr, self.0); } }
        fn set1(val: f32) -> Self { unsafe { Self(_mm256_set1_ps(val)) } }
        fn fused_add_mul(&mut self, a: Self, b: Self) {
            unsafe { self.0 = _mm256_fmadd_ps(a.0, b.0, self.0); }
        }

        fn psin(self) -> Self { unimplemented!("AVX FMA psin") }
        fn pcos(self) -> Self { unimplemented!("AVX FMA pcos") }
        fn pexp(self) -> Self { unimplemented!("AVX FMA pexp") }
        fn plog(self) -> Self { unimplemented!("AVX FMA plog") }
        fn psqrt(self) -> Self { unsafe { Self(_mm256_sqrt_ps(self.0)) } }
        fn prsqrt(self) -> Self { unsafe { Self(_mm256_rsqrt_ps(self.0)) } }
        fn pabs(self) -> Self { 
            unsafe {
                let sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u32 as i32));
                Self(_mm256_andnot_ps(sign_bit, self.0))
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct SsePacketF64(pub __m128d);

    impl Add for SsePacketF64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self { unsafe { Self(_mm_add_pd(self.0, rhs.0)) } }
    }
    impl Sub for SsePacketF64 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm_sub_pd(self.0, rhs.0)) } }
    }
    impl Mul for SsePacketF64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self { unsafe { Self(_mm_mul_pd(self.0, rhs.0)) } }
    }

    impl Packet<f64> for SsePacketF64 {
        const SIZE: usize = 2;
        unsafe fn load(ptr: *const f64) -> Self { unsafe { Self(_mm_loadu_pd(ptr)) } }
        unsafe fn store(self, ptr: *mut f64) { unsafe { _mm_storeu_pd(ptr, self.0); } }
        fn set1(val: f64) -> Self { unsafe { Self(_mm_set1_pd(val)) } }
        fn fused_add_mul(&mut self, a: Self, b: Self) {
            unsafe { self.0 = _mm_add_pd(self.0, _mm_mul_pd(a.0, b.0)); }
        }

        fn psin(self) -> Self { unimplemented!("SSE f64 psin") }
        fn pcos(self) -> Self { unimplemented!("SSE f64 pcos") }
        fn pexp(self) -> Self { unimplemented!("SSE f64 pexp") }
        fn plog(self) -> Self { unimplemented!("SSE f64 plog") }
        fn psqrt(self) -> Self { unsafe { Self(_mm_sqrt_pd(self.0)) } }
        fn prsqrt(self) -> Self { unsafe { Self(_mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(self.0))) } } // rsqrt_pd is approx, do 1/sqrt
        fn pabs(self) -> Self { 
            unsafe {
                let sign_bit = _mm_castsi128_pd(_mm_set1_epi64x(i64::MIN)); // i64::MIN is 0x8000...
                // set1_epi64x is SSE2.
                 Self(_mm_andnot_pd(sign_bit, self.0))
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct AvxPacketF64(pub __m256d);

    impl Add for AvxPacketF64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self { unsafe { Self(_mm256_add_pd(self.0, rhs.0)) } }
    }
    impl Sub for AvxPacketF64 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm256_sub_pd(self.0, rhs.0)) } }
    }
    impl Mul for AvxPacketF64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self { unsafe { Self(_mm256_mul_pd(self.0, rhs.0)) } }
    }

    impl Packet<f64> for AvxPacketF64 {
        const SIZE: usize = 4;
        unsafe fn load(ptr: *const f64) -> Self { unsafe { Self(_mm256_loadu_pd(ptr)) } }
        unsafe fn store(self, ptr: *mut f64) { unsafe { _mm256_storeu_pd(ptr, self.0); } }
        fn set1(val: f64) -> Self { unsafe { Self(_mm256_set1_pd(val)) } }
        fn fused_add_mul(&mut self, a: Self, b: Self) {
            unsafe { self.0 = _mm256_add_pd(self.0, _mm256_mul_pd(a.0, b.0)); }
        }

        fn psin(self) -> Self { unimplemented!("AVX f64 psin") }
        fn pcos(self) -> Self { unimplemented!("AVX f64 pcos") }
        fn pexp(self) -> Self { unimplemented!("AVX f64 pexp") }
        fn plog(self) -> Self { unimplemented!("AVX f64 plog") }
        fn psqrt(self) -> Self { unsafe { Self(_mm256_sqrt_pd(self.0)) } }
        fn prsqrt(self) -> Self { unsafe { Self(_mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(self.0))) } } 
        fn pabs(self) -> Self { 
            unsafe {
                let sign_bit = _mm256_castsi256_pd(_mm256_set1_epi64x(i64::MIN));
                 Self(_mm256_andnot_pd(sign_bit, self.0))
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct AvxFmaPacketF64(pub __m256d);

    impl Add for AvxFmaPacketF64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self { unsafe { Self(_mm256_add_pd(self.0, rhs.0)) } }
    }
    impl Sub for AvxFmaPacketF64 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self { unsafe { Self(_mm256_sub_pd(self.0, rhs.0)) } }
    }
    impl Mul for AvxFmaPacketF64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self { unsafe { Self(_mm256_mul_pd(self.0, rhs.0)) } }
    }

    impl Packet<f64> for AvxFmaPacketF64 {
        const SIZE: usize = 4;
        unsafe fn load(ptr: *const f64) -> Self { unsafe { Self(_mm256_loadu_pd(ptr)) } }
        unsafe fn store(self, ptr: *mut f64) { unsafe { _mm256_storeu_pd(ptr, self.0); } }
        fn set1(val: f64) -> Self { unsafe { Self(_mm256_set1_pd(val)) } }
        fn fused_add_mul(&mut self, a: Self, b: Self) {
            unsafe { self.0 = _mm256_fmadd_pd(a.0, b.0, self.0); }
        }

        fn psin(self) -> Self { unimplemented!("AVX FMA f64 psin") }
        fn pcos(self) -> Self { unimplemented!("AVX FMA f64 pcos") }
        fn pexp(self) -> Self { unimplemented!("AVX FMA f64 pexp") }
        fn plog(self) -> Self { unimplemented!("AVX FMA f64 plog") }
        fn psqrt(self) -> Self { unsafe { Self(_mm256_sqrt_pd(self.0)) } }
        fn prsqrt(self) -> Self { unsafe { Self(_mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(self.0))) } } 
        fn pabs(self) -> Self { 
            unsafe {
                let sign_bit = _mm256_castsi256_pd(_mm256_set1_epi64x(i64::MIN));
                 Self(_mm256_andnot_pd(sign_bit, self.0))
            }
        }
    }
}

pub mod special_functions;
pub use special_functions::sse_f32; // Expose module if needed, or just let it be used internally
