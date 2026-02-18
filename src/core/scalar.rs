//! Scalar trait for eigen-rs.
//! Defines requirements for types that can be used as matrix elements.

use crate::core::xpr::MatrixXpr;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use num_traits::Float;

/// Marker trait for scalar types supported by eigen-rs.
pub trait Scalar:
    Default
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Neg<Output = Self>
    + PartialEq
    + Neg<Output = Self>
    + PartialEq
    // + PartialOrd // Removed to support Complex
    + std::fmt::Debug
    + std::fmt::Display
    + Send
    + Sync
    + 'static
    + num_traits::Zero
{
    /// The Real part type (e.g., f32 for Complex<f32>).
    /// Must be PartialOrd to allow pivoting/comparisons.
    type Real: Scalar + PartialOrd;

    fn from_usize(v: usize) -> Self;
    fn from_f64(v: f64) -> Self;
    fn from_real(v: Self::Real) -> Self;
    fn real(self) -> Self::Real;
    fn imag(self) -> Self::Real;
    
    fn abs(self) -> Self::Real;
    fn sqrt(self) -> Self;
    fn recip(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn epsilon() -> Self::Real;
    fn conj(self) -> Self;
    fn norm_sq(self) -> Self::Real;
    fn to_f64(self) -> f64;

    /// Vectorized assignment helper.
    /// Returns true if vectorized assignment was performed.
    fn assign_vectorized<S, X>(_mat: &mut crate::core::matrix::Matrix<Self, S>, _xpr: &X) -> bool
    where
        S: crate::core::storage::Storage<Self>,
        X: crate::core::xpr::MatrixXpr<Self>,
    {
        false
    }

    /// Vectorized dot product helper.
    fn dot_vectorized<S1, S2>(
        _lhs: &crate::core::matrix::Matrix<Self, S1>,
        _rhs: &crate::core::matrix::Matrix<Self, S2>,
    ) -> Option<Self>
    where
        S1: crate::core::storage::Storage<Self>,
        S2: crate::core::storage::Storage<Self>,
    {
        None
    }

    /// Vectorized scale helper.
    fn scale_vectorized<S>(_mat: &mut crate::core::matrix::Matrix<Self, S>, _factor: Self) -> bool
    where
        S: crate::core::storage::Storage<Self>,
    {
        false
    }

    /// Vectorized squared norm helper.
    /// Vectorized squared norm helper.
    fn squared_norm_vectorized<S>(_mat: &crate::core::matrix::Matrix<Self, S>) -> Option<Self::Real>
    where
        S: crate::core::storage::Storage<Self>,
    {
        None
    }
}

impl Scalar for f32 {
    type Real = f32;

    fn from_usize(v: usize) -> Self {
        v as f32
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn from_real(v: Self::Real) -> Self {
        v
    }
    fn real(self) -> Self::Real {
        self
    }
    fn imag(self) -> Self::Real {
        0.0
    }
    fn abs(self) -> Self::Real {
        self.abs()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn recip(self) -> Self {
        1.0 / self
    }
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn asin(self) -> Self {
        self.asin()
    }
    fn acos(self) -> Self {
        self.acos()
    }
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
    }
    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn epsilon() -> Self::Real {
        f32::EPSILON
    }
    fn conj(self) -> Self {
        self
    }
    fn norm_sq(self) -> Self::Real {
        self * self
    }
    fn to_f64(self) -> f64 {
        self as f64
    }

    fn assign_vectorized<S, X>(mat: &mut crate::core::matrix::Matrix<Self, S>, xpr: &X) -> bool
    where
        S: crate::core::storage::Storage<Self>,
        X: crate::core::xpr::MatrixXpr<Self>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::core::arch::x86::AvxPacketF32;
                use crate::core::arch::Packet;

                let rows = mat.rows();
                let cols = mat.cols();
                let size = rows * cols;
                let data_ptr = mat.storage_mut().data_mut().as_mut_ptr();

                // 1. Linear Access Optimization
                if mat.has_linear_access() && xpr.has_linear_access() {
                    // eprintln!("DEBUG: assign_vectorized linear path taken");
                    if xpr.try_eval_to::<AvxPacketF32>(data_ptr, size) {
                        // eprintln!("DEBUG: try_eval_to succeeded");
                        return true;
                    }
                    // eprintln!("DEBUG: try_eval_to failed, using unrolled loop");

                    let mut i = 0;
                    while i + 64 <= size {
                        unsafe {
                            let p0 = xpr.packet_eval_linear::<AvxPacketF32>(i);
                            let p1 = xpr.packet_eval_linear::<AvxPacketF32>(i + 8);
                            let p2 = xpr.packet_eval_linear::<AvxPacketF32>(i + 16);
                            let p3 = xpr.packet_eval_linear::<AvxPacketF32>(i + 24);
                            let p4 = xpr.packet_eval_linear::<AvxPacketF32>(i + 32);
                            let p5 = xpr.packet_eval_linear::<AvxPacketF32>(i + 40);
                            let p6 = xpr.packet_eval_linear::<AvxPacketF32>(i + 48);
                            let p7 = xpr.packet_eval_linear::<AvxPacketF32>(i + 56);

                            p0.store(data_ptr.add(i));
                            p1.store(data_ptr.add(i + 8));
                            p2.store(data_ptr.add(i + 16));
                            p3.store(data_ptr.add(i + 24));
                            p4.store(data_ptr.add(i + 32));
                            p5.store(data_ptr.add(i + 40));
                            p6.store(data_ptr.add(i + 48));
                            p7.store(data_ptr.add(i + 56));
                        }
                        i += 64;
                    }
                    while i + 8 <= size {
                        unsafe {
                            let packet = xpr.packet_eval_linear::<AvxPacketF32>(i);
                            packet.store(data_ptr.add(i));
                        }
                        i += 8;
                    }
                    while i < size {
                        unsafe {
                            *data_ptr.add(i) = xpr.eval_linear(i);
                        }
                        i += 1;
                    }
                    return true;
                }

                // 2. Fallback: Column-Major Vectorization
                for c in 0..cols {
                    let col_offset = c * rows;
                    let mut r = 0;

                    // Vectorized Loop
                    while r + 8 <= rows {
                        unsafe {
                            let packet = xpr.packet_eval::<AvxPacketF32>(r, c);
                            let dest_ptr = data_ptr.add(col_offset + r);
                            packet.store(dest_ptr);
                        }
                        r += 8;
                    }
                    // Scalar cleanup
                    while r < rows {
                        unsafe {
                            let val = xpr.eval(r, c);
                            *data_ptr.add(col_offset + r) = val;
                        }
                        r += 1;
                    }
                }
                return true;
            }
        }
        false
    }

    fn dot_vectorized<S1, S2>(
        lhs: &crate::core::matrix::Matrix<Self, S1>,
        rhs: &crate::core::matrix::Matrix<Self, S2>,
    ) -> Option<Self>
    where
        S1: crate::core::storage::Storage<Self>,
        S2: crate::core::storage::Storage<Self>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_feature = "avx2")]
            {
                return unsafe { dot_vectorized_avx2_f32(lhs, rhs) };
            }
            #[cfg(not(target_feature = "avx2"))]
            if is_x86_feature_detected!("avx2") {
                return unsafe { dot_vectorized_avx2_f32(lhs, rhs) };
            }
        }
        None
    }

    fn squared_norm_vectorized<S>(mat: &crate::core::matrix::Matrix<Self, S>) -> Option<Self::Real>
    where
        S: crate::core::storage::Storage<Self>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::core::arch::x86::AvxFmaPacketF32;
                use crate::core::arch::Packet;

                let size = mat.size();

                if mat.has_linear_access() {
                    let ptr = mat.storage().data().as_ptr();
                    let packet_size = AvxFmaPacketF32::SIZE;
                    let mut i = 0;
                    let mut sum0 = AvxFmaPacketF32::set1(0.0);
                    let mut sum1 = AvxFmaPacketF32::set1(0.0);
                    let mut sum2 = AvxFmaPacketF32::set1(0.0);
                    let mut sum3 = AvxFmaPacketF32::set1(0.0);
                    let mut sum4 = AvxFmaPacketF32::set1(0.0);
                    let mut sum5 = AvxFmaPacketF32::set1(0.0);
                    let mut sum6 = AvxFmaPacketF32::set1(0.0);
                    let mut sum7 = AvxFmaPacketF32::set1(0.0);

                    unsafe {
                        while i + 64 <= size {
                            let v0 = AvxFmaPacketF32::load(ptr.add(i));
                            let v1 = AvxFmaPacketF32::load(ptr.add(i + 8));
                            let v2 = AvxFmaPacketF32::load(ptr.add(i + 16));
                            let v3 = AvxFmaPacketF32::load(ptr.add(i + 24));
                            let v4 = AvxFmaPacketF32::load(ptr.add(i + 32));
                            let v5 = AvxFmaPacketF32::load(ptr.add(i + 40));
                            let v6 = AvxFmaPacketF32::load(ptr.add(i + 48));
                            let v7 = AvxFmaPacketF32::load(ptr.add(i + 56));

                            sum0.fused_add_mul(v0, v0);
                            sum1.fused_add_mul(v1, v1);
                            sum2.fused_add_mul(v2, v2);
                            sum3.fused_add_mul(v3, v3);
                            sum4.fused_add_mul(v4, v4);
                            sum5.fused_add_mul(v5, v5);
                            sum6.fused_add_mul(v6, v6);
                            sum7.fused_add_mul(v7, v7);
                            i += 64;
                        }
                        while i + packet_size <= size {
                            let v = AvxFmaPacketF32::load(ptr.add(i));
                            sum0.fused_add_mul(v, v);
                            i += packet_size;
                        }
                    }

                    let mut sum_packet =
                        (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);

                    let mut sum = sum_packet.sum();

                    while i < size {
                        unsafe {
                            let v = *ptr.add(i);
                            sum += v * v;
                        }
                        i += 1;
                    }
                    return Some(sum);
                }
            }
        }
        None
    }

    fn scale_vectorized<S>(mat: &mut crate::core::matrix::Matrix<Self, S>, factor: Self) -> bool
    where
        S: crate::core::storage::Storage<Self>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::core::arch::x86::AvxPacketF32;
                use crate::core::arch::Packet;

                let size = mat.size();
                if mat.has_linear_access() {
                    let ptr = mat.storage_mut().data_mut().as_mut_ptr();
                    let packet_size = AvxPacketF32::SIZE;
                    let vec_factor = AvxPacketF32::set1(factor);
                    let mut i = 0;

                    unsafe {
                        while i + 64 <= size {
                            let v0 = AvxPacketF32::load(ptr.add(i));
                            let v1 = AvxPacketF32::load(ptr.add(i + 8));
                            let v2 = AvxPacketF32::load(ptr.add(i + 16));
                            let v3 = AvxPacketF32::load(ptr.add(i + 24));
                            let v4 = AvxPacketF32::load(ptr.add(i + 32));
                            let v5 = AvxPacketF32::load(ptr.add(i + 40));
                            let v6 = AvxPacketF32::load(ptr.add(i + 48));
                            let v7 = AvxPacketF32::load(ptr.add(i + 56));

                            (v0 * vec_factor).store(ptr.add(i));
                            (v1 * vec_factor).store(ptr.add(i + 8));
                            (v2 * vec_factor).store(ptr.add(i + 16));
                            (v3 * vec_factor).store(ptr.add(i + 24));
                            (v4 * vec_factor).store(ptr.add(i + 32));
                            (v5 * vec_factor).store(ptr.add(i + 40));
                            (v6 * vec_factor).store(ptr.add(i + 48));
                            (v7 * vec_factor).store(ptr.add(i + 56));

                            i += 64;
                        }
                        while i + packet_size <= size {
                            let val = AvxPacketF32::load(ptr.add(i));
                            (val * vec_factor).store(ptr.add(i));
                            i += packet_size;
                        }
                        while i < size {
                            *ptr.add(i) *= factor;
                            i += 1;
                        }
                    }
                    return true;
                }
            }
        }
        false
    }
}

impl Scalar for f64 {
    type Real = f64;
    
    fn from_usize(v: usize) -> Self {
        v as f64
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn from_real(v: Self::Real) -> Self {
        v
    }
    fn real(self) -> Self::Real {
        self
    }
    fn imag(self) -> Self::Real {
        0.0
    }
    fn abs(self) -> Self::Real {
        self.abs()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn recip(self) -> Self {
        1.0 / self
    }
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn asin(self) -> Self {
        self.asin()
    }
    fn acos(self) -> Self {
        self.acos()
    }
    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
    }
    fn powf(self, n: Self) -> Self {
        self.powf(n)
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    fn epsilon() -> Self::Real {
        f64::EPSILON
    }
    fn conj(self) -> Self {
        self
    }
    fn norm_sq(self) -> Self::Real {
        self * self
    }
    fn to_f64(self) -> f64 {
        self
    }

    fn assign_vectorized<S, X>(mat: &mut crate::core::matrix::Matrix<Self, S>, xpr: &X) -> bool
    where
        S: crate::core::storage::Storage<Self>,
        X: crate::core::xpr::MatrixXpr<Self>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::core::arch::x86::AvxPacketF64;
                use crate::core::arch::Packet;

                let rows = mat.rows();
                let cols = mat.cols();
                let size = rows * cols;
                let data_ptr = mat.storage_mut().data_mut().as_mut_ptr();

                // 1. Linear Access Optimization
                if mat.has_linear_access() && xpr.has_linear_access() {
                    if xpr.try_eval_to::<AvxPacketF64>(data_ptr, size) {
                        return true;
                    }

                    let mut i = 0;
                    while i + 4 <= size {
                        unsafe {
                            let packet = xpr.packet_eval_linear::<AvxPacketF64>(i);
                            packet.store(data_ptr.add(i));
                        }
                        i += 4;
                    }
                    while i < size {
                        unsafe {
                            *data_ptr.add(i) = xpr.eval_linear(i);
                        }
                        i += 1;
                    }
                    return true;
                }

                // 2. Fallback
                for c in 0..cols {
                    let col_offset = c * rows;
                    let mut r = 0;
                    while r + 4 <= rows {
                        unsafe {
                            let dest_ptr = data_ptr.add(col_offset + r);
                            let packet = xpr.packet_eval::<AvxPacketF64>(r, c);
                            packet.store(dest_ptr);
                        }
                        r += 4;
                    }
                    while r < rows {
                        unsafe {
                            *data_ptr.add(col_offset + r) = xpr.eval(r, c);
                        }
                        r += 1;
                    }
                }
                return true;
            }
        }
        false
    }

    fn dot_vectorized<S1, S2>(
        lhs: &crate::core::matrix::Matrix<Self, S1>,
        rhs: &crate::core::matrix::Matrix<Self, S2>,
    ) -> Option<Self>
    where
        S1: crate::core::storage::Storage<Self>,
        S2: crate::core::storage::Storage<Self>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::core::arch::x86::AvxFmaPacketF64;
                use crate::core::arch::Packet;

                let size = lhs.size();
                if size != rhs.size() {
                    return None;
                }

                if lhs.has_linear_access() && rhs.has_linear_access() {
                    let ptr_l = lhs.storage().data().as_ptr();
                    let ptr_r = rhs.storage().data().as_ptr();
                    let packet_size = AvxFmaPacketF64::SIZE;
                    let mut i = 0;
                    let mut sum0 = AvxFmaPacketF64::set1(0.0);
                    let mut sum1 = AvxFmaPacketF64::set1(0.0);
                    let mut sum2 = AvxFmaPacketF64::set1(0.0);
                    let mut sum3 = AvxFmaPacketF64::set1(0.0);

                    unsafe {
                        while i + 16 <= size {
                            sum0.fused_add_mul(
                                AvxFmaPacketF64::load(ptr_l.add(i)),
                                AvxFmaPacketF64::load(ptr_r.add(i)),
                            );
                            sum1.fused_add_mul(
                                AvxFmaPacketF64::load(ptr_l.add(i + 4)),
                                AvxFmaPacketF64::load(ptr_r.add(i + 4)),
                            );
                            sum2.fused_add_mul(
                                AvxFmaPacketF64::load(ptr_l.add(i + 8)),
                                AvxFmaPacketF64::load(ptr_r.add(i + 8)),
                            );
                            sum3.fused_add_mul(
                                AvxFmaPacketF64::load(ptr_l.add(i + 12)),
                                AvxFmaPacketF64::load(ptr_r.add(i + 12)),
                            );
                            i += 16;
                        }
                        while i + packet_size <= size {
                            let a = AvxFmaPacketF64::load(ptr_l.add(i));
                            let b = AvxFmaPacketF64::load(ptr_r.add(i));
                            sum0.fused_add_mul(a, b);
                            i += packet_size;
                        }
                    }

                    let mut sum = (sum0 + sum1 + sum2 + sum3).sum();

                    while i < size {
                        unsafe {
                            sum += *ptr_l.add(i) * *ptr_r.add(i);
                        }
                        i += 1;
                    }
                    return Some(sum);
                }
            }
        }
        None
    }

    fn squared_norm_vectorized<S>(mat: &crate::core::matrix::Matrix<Self, S>) -> Option<Self::Real>
    where
        S: crate::core::storage::Storage<Self>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::core::arch::x86::AvxFmaPacketF64;
                use crate::core::arch::Packet;

                let size = mat.size();

                if mat.has_linear_access() {
                    let ptr = mat.storage().data().as_ptr();
                    let packet_size = AvxFmaPacketF64::SIZE;
                    let mut i = 0;
                    let mut sum0 = AvxFmaPacketF64::set1(0.0);
                    let mut sum1 = AvxFmaPacketF64::set1(0.0);
                    let mut sum2 = AvxFmaPacketF64::set1(0.0);
                    let mut sum3 = AvxFmaPacketF64::set1(0.0);

                    unsafe {
                        while i + 16 <= size {
                            let v0 = AvxFmaPacketF64::load(ptr.add(i));
                            let v1 = AvxFmaPacketF64::load(ptr.add(i + 4));
                            let v2 = AvxFmaPacketF64::load(ptr.add(i + 8));
                            let v3 = AvxFmaPacketF64::load(ptr.add(i + 12));

                            sum0.fused_add_mul(v0, v0);
                            sum1.fused_add_mul(v1, v1);
                            sum2.fused_add_mul(v2, v2);
                            sum3.fused_add_mul(v3, v3);
                            i += 16;
                        }
                        while i + packet_size <= size {
                            let v = AvxFmaPacketF64::load(ptr.add(i));
                            sum0.fused_add_mul(v, v);
                            i += packet_size;
                        }
                    }

                    let mut sum = (sum0 + sum1 + sum2 + sum3).sum();

                    while i < size {
                        unsafe {
                            let v = *ptr.add(i);
                            sum += v * v;
                        }
                        i += 1;
                    }
                    return Some(sum);
                }
            }
        }
        None
    }

    fn scale_vectorized<S>(mat: &mut crate::core::matrix::Matrix<Self, S>, factor: Self) -> bool
    where
        S: crate::core::storage::Storage<Self>,
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                use crate::core::arch::x86::AvxPacketF64;
                use crate::core::arch::Packet;

                let size = mat.size();
                if mat.has_linear_access() {
                    let ptr = mat.storage_mut().data_mut().as_mut_ptr();
                    let packet_size = AvxPacketF64::SIZE;
                    let vec_factor = AvxPacketF64::set1(factor);
                    let mut i = 0;

                    unsafe {
                        while i + packet_size <= size {
                            let val = AvxPacketF64::load(ptr.add(i));
                            (val * vec_factor).store(ptr.add(i));
                            i += packet_size;
                        }
                        while i < size {
                            *ptr.add(i) *= factor;
                            i += 1;
                        }
                    }
                    return true;
                }
            }
        }
        false
    }
}
// Future: impl Scalar for Complex<f32>, etc.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn dot_vectorized_avx2_f32<S1, S2>(
    lhs: &crate::core::matrix::Matrix<f32, S1>,
    rhs: &crate::core::matrix::Matrix<f32, S2>,
) -> Option<f32>
where
    S1: crate::core::storage::Storage<f32>,
    S2: crate::core::storage::Storage<f32>,
{
    use crate::core::arch::x86::AvxFmaPacketF32;
    use crate::core::arch::Packet;

    // eprintln!("DEBUG: dot_vectorized_avx2_f32 called");

    let size = lhs.size();
    if size != rhs.size() {
        return None;
    }

    if lhs.has_linear_access() && rhs.has_linear_access() {
        let ptr_l = lhs.storage().data().as_ptr();
        let ptr_r = rhs.storage().data().as_ptr();
        let packet_size = AvxFmaPacketF32::SIZE; // 8
        let mut i = 0;

        // Accumulators
        let mut sum0 = AvxFmaPacketF32::set1(0.0);
        let mut sum1 = AvxFmaPacketF32::set1(0.0);
        let mut sum2 = AvxFmaPacketF32::set1(0.0);
        let mut sum3 = AvxFmaPacketF32::set1(0.0);
        let mut sum4 = AvxFmaPacketF32::set1(0.0);
        let mut sum5 = AvxFmaPacketF32::set1(0.0);
        let mut sum6 = AvxFmaPacketF32::set1(0.0);
        let mut sum7 = AvxFmaPacketF32::set1(0.0);

        // 64-element unrolling (8 packets)
        while i + 64 <= size {
            sum0.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i)),
                AvxFmaPacketF32::load(ptr_r.add(i)),
            );
            sum1.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i + 8)),
                AvxFmaPacketF32::load(ptr_r.add(i + 8)),
            );
            sum2.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i + 16)),
                AvxFmaPacketF32::load(ptr_r.add(i + 16)),
            );
            sum3.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i + 24)),
                AvxFmaPacketF32::load(ptr_r.add(i + 24)),
            );
            sum4.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i + 32)),
                AvxFmaPacketF32::load(ptr_r.add(i + 32)),
            );
            sum5.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i + 40)),
                AvxFmaPacketF32::load(ptr_r.add(i + 40)),
            );
            sum6.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i + 48)),
                AvxFmaPacketF32::load(ptr_r.add(i + 48)),
            );
            sum7.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i + 56)),
                AvxFmaPacketF32::load(ptr_r.add(i + 56)),
            );
            i += 64;
        }

        // Single packet cleanup
        while i + packet_size <= size {
            sum0.fused_add_mul(
                AvxFmaPacketF32::load(ptr_l.add(i)),
                AvxFmaPacketF32::load(ptr_r.add(i)),
            );
            i += packet_size;
        }

        // Reduce packets
        let mut sum_packet = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
        let mut sum = sum_packet.sum();

        // Scalar cleanup
        while i < size {
            sum += *ptr_l.add(i) * *ptr_r.add(i);
            i += 1;
        }
        return Some(sum);
    }
    None
}

// Implementation for num_complex::Complex
// We implement Scalar for num_complex::Complex<T> where T is a real-valued Scalar (f32/f64).

impl<T: Scalar<Real = T> + Float + num_traits::NumAssign + num_traits::Num + num_traits::NumCast + num_traits::One + num_traits::ToPrimitive + num_traits::Zero> Scalar for num_complex::Complex<T> {
    type Real = T;

    fn from_usize(v: usize) -> Self {
        Self::new(T::from_usize(v), T::zero())
    }
    fn from_f64(v: f64) -> Self {
        Self::new(T::from_f64(v), T::zero())
    }
    fn from_real(v: Self::Real) -> Self {
        Self::new(v, T::zero())
    }
    fn real(self) -> Self::Real {
        self.re
    }
    fn imag(self) -> Self::Real {
        self.im
    }
    fn abs(self) -> Self::Real {
        self.norm()
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn recip(self) -> Self {
        self.inv()
    }
    fn sin(self) -> Self {
         self.sin()
    }
    fn cos(self) -> Self {
         self.cos()
    }
    fn asin(self) -> Self {
         self.asin()
    }
    fn acos(self) -> Self {
         self.acos()
    }
    fn atan2(self, _other: Self) -> Self {
         unimplemented!("atan2 not supported for Complex")
    }
    fn powf(self, n: Self) -> Self {
         self.powc(n)
    }
    fn exp(self) -> Self {
         self.exp()
    }
    fn ln(self) -> Self {
         self.ln()
    }
    fn epsilon() -> Self::Real {
        <T as Float>::epsilon()
    }
    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    fn norm_sq(self) -> Self::Real {
        self.norm_sqr()
    }
    fn to_f64(self) -> f64 {
        self.norm().to_f64()
    }
}
