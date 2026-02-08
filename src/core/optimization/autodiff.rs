//! Automatic differentiation using Dual numbers.

use crate::core::scalar::Scalar;
use crate::core::matrix::MatrixX;
use crate::core::optimization::Functor;
use crate::core::xpr::MatrixXpr;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};

/// A dual number for forward-mode automatic differentiation.
/// Value = real + epsilon * grad
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Dual<T: Scalar> {
    pub real: T,
    pub grad: T,
}

impl<T: Scalar> num_traits::Zero for Dual<T> {
    fn zero() -> Self {
        Self::constant(T::zero())
    }
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.grad.is_zero()
    }
}

impl<T: Scalar> Dual<T> {
    pub fn new(real: T, grad: T) -> Self {
        Self { real, grad }
    }

    pub fn constant(real: T) -> Self {
        Self { real, grad: T::default() }
    }

    pub fn variable(real: T) -> Self {
        Self { real, grad: T::from_f64(1.0) }
    }
}

impl<T: Scalar> std::fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}e", self.real, self.grad)
    }
}

impl<T: Scalar> Add for Dual<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.real + rhs.real, self.grad + rhs.grad)
    }
}

impl<T: Scalar> Sub for Dual<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.real - rhs.real, self.grad - rhs.grad)
    }
}

impl<T: Scalar> Mul for Dual<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        // (a + be)(c + de) = ac + (ad + bc)e
        Self::new(self.real * rhs.real, self.real * rhs.grad + self.grad * rhs.real)
    }
}

impl<T: Scalar> Div for Dual<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        // (a + be)/(c + de) = a/c + (bc - ad)/c^2 e
        let c2 = rhs.real * rhs.real;
        Self::new(self.real / rhs.real, (self.grad * rhs.real - self.real * rhs.grad) / c2)
    }
}

impl<T: Scalar> Neg for Dual<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.real, -self.grad)
    }
}

impl<T: Scalar> AddAssign for Dual<T> {
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}
impl<T: Scalar> SubAssign for Dual<T> {
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}
impl<T: Scalar> MulAssign for Dual<T> {
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}
impl<T: Scalar> DivAssign for Dual<T> {
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

impl<T: Scalar> Scalar for Dual<T> {
    fn from_usize(v: usize) -> Self { Self::constant(T::from_usize(v)) }
    fn from_f64(v: f64) -> Self { Self::constant(T::from_f64(v)) }
    
    fn abs(self) -> Self {
        if self.real >= T::default() { self } else { -self }
    }
    
    fn sqrt(self) -> Self {
        let s = self.real.sqrt();
        Self::new(s, self.grad / (T::from_f64(2.0) * s))
    }
    
    fn recip(self) -> Self {
        let r = self.real.recip();
        Self::new(r, -self.grad * r * r)
    }
    
    fn sin(self) -> Self {
        Self::new(self.real.sin(), self.grad * self.real.cos())
    }
    
    fn cos(self) -> Self {
        Self::new(self.real.cos(), -self.grad * self.real.sin())
    }
    
    fn asin(self) -> Self {
        let val = T::from_f64(1.0) - self.real * self.real;
        Self::new(self.real.asin(), self.grad / val.sqrt())
    }
    
    fn acos(self) -> Self {
        let val = T::from_f64(1.0) - self.real * self.real;
        Self::new(self.real.acos(), -self.grad / val.sqrt())
    }
    
    fn atan2(self, _other: Self) -> Self {
        // Simplified atan2 gradient if needed, but LM mostly uses simple ops
        unimplemented!("atan2 for Dual not implemented yet")
    }
    
    fn powf(self, n: Self) -> Self {
        // (u^v)' = u^v * (v' ln u + v u' / u)
        // If n is constant (grad=0): v u^(v-1) u'
        if n.grad == T::default() {
            let p = self.real.powf(n.real);
            let p_prev = self.real.powf(n.real - T::from_f64(1.0));
            Self::new(p, n.real * p_prev * self.grad)
        } else {
            unimplemented!("General u^v for Dual not implemented yet")
        }
    }

    fn exp(self) -> Self {
        let e = self.real.exp();
        Self::new(e, e * self.grad)
    }
    
    fn ln(self) -> Self {
        Self::new(self.real.ln(), self.grad / self.real)
    }
    
    fn epsilon() -> Self { Self::constant(T::epsilon()) }
    fn conj(self) -> Self { self }
    fn norm_sq(self) -> Self { self * self }
    fn to_f64(self) -> f64 { self.real.to_f64() }
}

// Trick for atan2 placeholder to avoid compilation error if not used
impl<T: Scalar> Dual<T> {
}

/// A trait for functions that can be evaluated with any Scalar type (including Dual).
pub trait AdResiduals<T: Scalar> {
    fn inputs(&self) -> usize;
    fn values(&self) -> usize;
    fn operator<D: Scalar>(&self, x: &MatrixX<D>, fvec: &mut MatrixX<D>) -> Result<(), String>;
}

/// A wrapper that adds automatic Jacobian computation to an `AdResiduals` implementation.
pub struct AutoDiff<F: AdResiduals<T>, T: Scalar> {
    func: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<F: AdResiduals<T>, T: Scalar> AutoDiff<F, T> {
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F: AdResiduals<T>, T: Scalar + 'static> Functor<T> for AutoDiff<F, T> {
    fn inputs(&self) -> usize { self.func.inputs() }
    fn values(&self) -> usize { self.func.values() }

    fn operator(&self, x: &MatrixX<T>, fvec: &mut MatrixX<T>) -> Result<(), String> {
        self.func.operator(x, fvec)
    }

    fn jacobian(&self, x: &MatrixX<T>, fjac: &mut MatrixX<T>) -> Result<(), String> {
        let n = self.inputs();
        let m = self.values();

        for j in 0..n {
            // Create input vector of Dual numbers
            let mut x_dual = MatrixX::<Dual<T>>::new_dynamic(n, 1)?;
            for i in 0..n {
                let val = *x.get(i, 0).unwrap();
                let grad = if i == j { T::from_f64(1.0) } else { T::default() };
                *x_dual.get_mut(i, 0).unwrap() = Dual::new(val, grad);
            }
            
            let mut fvec_dual = MatrixX::<Dual<T>>::new_dynamic(m, 1)?;
            self.func.operator(&x_dual, &mut fvec_dual)?;
            
            for i in 0..m {
                *fjac.get_mut(i, j).unwrap() = fvec_dual.eval(i, 0).grad;
            }
        }
        
        Ok(())
    }
}
