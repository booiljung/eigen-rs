//! Scalar trait for eigen-rs.
//! Defines requirements for types that can be used as matrix elements.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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
    + PartialOrd
    + std::fmt::Debug
    + std::fmt::Display
    + Send
    + Sync
    + 'static
    + num_traits::Zero
{
    fn from_usize(v: usize) -> Self;
    fn from_f64(v: f64) -> Self;
    fn abs(self) -> Self;
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
    fn epsilon() -> Self;
    fn conj(self) -> Self;
    fn norm_sq(self) -> Self;
    fn to_f64(self) -> f64;
}

impl Scalar for f32 {
    fn from_usize(v: usize) -> Self {
        v as f32
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn abs(self) -> Self {
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
    fn epsilon() -> Self {
        f32::EPSILON
    }
    fn conj(self) -> Self {
        self
    }
    fn norm_sq(self) -> Self {
        self * self
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}
impl Scalar for f64 {
    fn from_usize(v: usize) -> Self {
        v as f64
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn abs(self) -> Self {
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
    fn epsilon() -> Self {
        f64::EPSILON
    }
    fn conj(self) -> Self {
        self
    }
    fn norm_sq(self) -> Self {
        self * self
    }
    fn to_f64(self) -> f64 {
        self
    }
}
// Future: impl Scalar for Complex<f32>, etc.
