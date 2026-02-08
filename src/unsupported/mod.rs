pub mod matrix_functions;
pub mod kronecker;
pub mod polynomials;
pub mod splines;
pub mod special_functions;

pub use kronecker::kronecker_product;
pub use polynomials::{poly_eval, PolynomialSolver};
pub use splines::Spline;

pub mod fft;
pub mod skyline;
pub mod market_io;
