//! Complex number implementation for eigen-rs.

use crate::core::scalar::Scalar;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, Default)]
pub struct Complex<T: Scalar> {
    pub re: T,
    pub im: T,
}

impl<T: Scalar> num_traits::Zero for Complex<T> {
    fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }
}

impl<T: Scalar + num_traits::One> num_traits::One for Complex<T> {
    fn one() -> Self {
        Self::new(T::one(), T::zero())
    }
}

impl<T: Scalar> std::fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {:?})", self.re, self.im)
    }
}

impl<T: Scalar> Complex<T> {
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn norm_sq(self) -> T {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(self) -> T {
        self.norm_sq().sqrt()
    }
}

impl<T: Scalar> Add for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Scalar> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl<T: Scalar> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: Scalar> Div for Complex<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl<T: Scalar> Neg for Complex<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: Scalar> AddAssign for Complex<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: Scalar> SubAssign for Complex<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: Scalar> MulAssign for Complex<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: Scalar> DivAssign for Complex<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Scalar> PartialEq for Complex<T> {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re && self.im == other.im
    }
}

// Complex numbers are not naturally ordered, but we implement PartialOrd
// lexicographically to satisfy the Scalar trait requirements.
impl<T: Scalar> PartialOrd for Complex<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.re.partial_cmp(&other.re) {
            Some(std::cmp::Ordering::Equal) => self.im.partial_cmp(&other.im),
            ord => ord,
        }
    }
}

impl<T: Scalar> Scalar for Complex<T> {
    fn from_usize(v: usize) -> Self {
        Self::new(T::from_usize(v), T::default())
    }
    fn from_f64(v: f64) -> Self {
        Self::new(T::from_f64(v), T::default())
    }
    fn abs(self) -> Self {
        Self::new(self.norm(), T::default())
    }
    fn sqrt(self) -> Self {
        let r = self.norm();
        let re = ((r + self.re) / T::from_f64(2.0)).sqrt();
        let im = ((r - self.re) / T::from_f64(2.0)).sqrt();
        if self.im < T::default() {
            Self::new(re, -im)
        } else {
            Self::new(re, im)
        }
    }
    fn recip(self) -> Self {
        Self::from_usize(1) / self
    }

    // Minimal implementations for transcendents, can be expanded
    fn sin(self) -> Self {
        unimplemented!("Trigonometric functions for Complex not yet required")
    }
    fn cos(self) -> Self {
        unimplemented!("Trigonometric functions for Complex not yet required")
    }
    fn asin(self) -> Self {
        unimplemented!("Trigonometric functions for Complex not yet required")
    }
    fn acos(self) -> Self {
        unimplemented!("Trigonometric functions for Complex not yet required")
    }
    fn atan2(self, _other: Self) -> Self {
        unimplemented!("Atan2 for Complex not yet required")
    }
    fn powf(self, _n: Self) -> Self {
        unimplemented!("Powf for Complex not yet required")
    }
    fn exp(self) -> Self {
        let exp_re = self.re.exp();
        Self::new(exp_re * self.im.cos(), exp_re * self.im.sin())
    }
    fn ln(self) -> Self {
        Self::new(self.norm().ln(), self.im.atan2(self.re))
    }
    fn epsilon() -> Self {
        Self::new(T::epsilon(), T::default())
    }
    fn conj(self) -> Self {
        self.conj()
    }
    fn norm_sq(self) -> Self {
        Self::new(self.norm_sq(), T::default())
    }
    fn to_f64(self) -> f64 {
        self.norm().to_f64()
    }
}
