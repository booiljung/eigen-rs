use crate::core::scalar::Scalar;

// Use libm for special functions (Pure Rust port of MUSL libm)
// This removes the need for libc bindings and unsafe blocks.

/// Trait for special functions support.
pub trait SpecialFunctions: Scalar {
    fn erf(self) -> Self;
    fn erfc(self) -> Self;
    fn lgamma(self) -> Self;
    fn bessel_j0(self) -> Self;
    fn bessel_j1(self) -> Self;
}

impl SpecialFunctions for f32 {
    #[inline]
    fn erf(self) -> Self {
        libm::erff(self)
    }

    #[inline]
    fn erfc(self) -> Self {
        libm::erfcf(self)
    }

    #[inline]
    fn lgamma(self) -> Self {
        libm::lgammaf(self)
    }

    #[inline]
    fn bessel_j0(self) -> Self {
        libm::j0f(self)
    }

    #[inline]
    fn bessel_j1(self) -> Self {
        libm::j1f(self)
    }
}

impl SpecialFunctions for f64 {
    #[inline]
    fn erf(self) -> Self {
        libm::erf(self)
    }

    #[inline]
    fn erfc(self) -> Self {
        libm::erfc(self)
    }

    #[inline]
    fn lgamma(self) -> Self {
        libm::lgamma(self)
    }

    #[inline]
    fn bessel_j0(self) -> Self {
        libm::j0(self)
    }

    #[inline]
    fn bessel_j1(self) -> Self {
        libm::j1(self)
    }
}
