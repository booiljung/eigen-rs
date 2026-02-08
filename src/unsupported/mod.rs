pub mod kronecker;
pub mod matrix_functions;
pub mod polynomials;
pub mod special_functions;
pub mod splines;

pub use kronecker::kronecker_product;
pub use polynomials::{poly_eval, PolynomialSolver};
pub use splines::Spline;

pub mod fft;
#[cfg(feature = "std")]
pub mod market_io;
pub mod skyline;
