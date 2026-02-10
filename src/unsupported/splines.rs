use crate::core::matrix::MatrixX;
use crate::core::scalar::Scalar;

use crate::core::xpr::MatrixXpr;
use alloc::vec::Vec;
use std::fmt::Debug;

/// B-Spline curve of degree `k`.
///
/// Defined by a set of knots and control points.
/// S(u) = Sum(P_i * N_i,k(u))
///
/// Control Points are stored in a (Dim x n) matrix, where each column is a point.
#[derive(Debug, Clone)]
pub struct Spline<T: Scalar> {
    /// Knot vector. length = n + k + 1.
    knots: Vec<T>,
    /// Control points matrix. Size: Dim x n.
    ctrl_points: MatrixX<T>,
    /// Degree of the spline.
    degree: usize,
}

impl<T: Scalar + 'static> Spline<T> {
    /// Creates a new B-Spline.
    ///
    /// # Arguments
    /// * `knots` - Knot vector (sorted).
    /// * `ctrl_points` - Control points matrix (Dim x n).
    /// * `degree` - Degree of the spline.
    ///
    /// # Panics
    /// * If knots.len() != ctrl_points.cols() + degree + 1
    pub fn new(knots: Vec<T>, ctrl_points: MatrixX<T>, degree: usize) -> Self {
        let n = ctrl_points.cols();
        assert_eq!(
            knots.len(),
            n + degree + 1,
            "Invalid knot vector length: expected {}, got {}",
            n + degree + 1,
            knots.len()
        );

        Self {
            knots,
            ctrl_points,
            degree,
        }
    }

    /// Returns the dimension of the spline space.
    pub fn dim(&self) -> usize {
        self.ctrl_points.rows()
    }

    /// Evaluates the spline at parameter `u`.
    ///
    /// Uses De Boor's algorithm.
    pub fn eval(&self, u: T) -> MatrixX<T> {
        let k = self.degree;

        let i = self.find_span(u);

        // We need a temporary buffer for points.
        // Initialize with P_{i-k} ... P_i
        let mut d: Vec<MatrixX<T>> = Vec::with_capacity(k + 1);
        for j in 0..=k {
            let idx = i - k + j;
            let block = self.ctrl_points.col(idx);
            // Explicitly convert BlockOp to MatrixX
            let mut m =
                MatrixX::new_dynamic(block.rows(), block.cols()).expect("Allocation failed");
            m.assign(&block).expect("Block assignment failed");
            d.push(m);
        }

        for r in 1..=k {
            for j in (r..=k).rev() {
                // alpha = (u - t_{i-k+j}) / (t_{i+1+j-r} - t_{i-k+j})
                let t_left = self.knots[i - k + j];
                let t_right = self.knots[i + 1 + j - r];

                let alpha = if t_right == t_left {
                    T::from_usize(0)
                } else {
                    (u - t_left) / (t_right - t_left)
                };

                // Clone to satisfy borrow checker / avoid aliasing during update
                let p_prev = d[j - 1].clone();
                let p_curr = d[j].clone();

                // d[j] = (1-alpha)*d[j-1] + alpha*d[j]
                let coeff1 = T::from_usize(1) - alpha;

                // Manual assignment with direct element access
                for r_idx in 0..d[j].rows() {
                    for c_idx in 0..d[j].cols() {
                        if let Some(val_ref) = d[j].get_mut(r_idx, c_idx) {
                            // Direct evaluation of terms without intermediate Expression structs
                            // Logic: d[j] = (1-alpha)*d[j-1] + alpha*d[j]
                            let val1 = *p_prev.get(r_idx, c_idx).unwrap();
                            let val2 = *p_curr.get(r_idx, c_idx).unwrap();
                            *val_ref = val1 * coeff1 + val2 * alpha;
                        }
                    }
                }
            }
        }

        d[k].clone()
    }

    // Binary search for knot span
    fn find_span(&self, u: T) -> usize {
        let k = self.degree;
        let n = self.ctrl_points.cols();
        // Domain [t_k, t_{n+1}]

        // Special case for upper boundary (u == t_{n+1})
        // In this case, we return span n-1.
        if u >= self.knots[n + 1] {
            return n - 1;
        }

        // Find index idx such that knots[idx] > u
        // Equivalent to std::upper_bound or partition_point(|t| t <= u)
        let idx = self.knots.partition_point(|t| *t <= u);

        // The span index i should satisfy: t_i <= u < t_{i+1}
        // idx is the first element > u.
        // So knots[idx-1] <= u.
        // Hence i = idx - 1.
        let mut i = if idx == 0 { 0 } else { idx - 1 };

        // Clamp to valid range [k, n-1] assuming standard B-spline domain
        if i < k {
            i = k;
        }
        if i >= n {
            i = n - 1;
        }

        i
    }

    /// Fits a spline of degree `degree` through the given points.
    ///
    /// Uses chord-length parameterization and solves the global interpolation system.
    /// Returns a new Spline.
    pub fn interpolate(points: &MatrixX<T>, degree: usize) -> Result<Self, String> {
        let n = points.cols();
        let dim = points.rows();

        if n < degree + 1 {
            return Err(format!(
                "Need at least degree + 1 points (got {}, degree {})",
                n, degree
            ));
        }

        // 1. Parameterization (Chord Length)
        let mut u = vec![T::from_usize(0); n];
        u[0] = T::from_usize(0);
        for i in 1..n {
            let mut dist_sq = T::from_usize(0);
            for d in 0..dim {
                let diff = *points.get(d, i).unwrap() - *points.get(d, i - 1).unwrap();
                dist_sq += diff * diff;
            }
            u[i] = u[i - 1] + dist_sq.sqrt();
        }

        // Normalize paremeters to [0, 1]
        let total_len = u[n - 1];
        if total_len > T::from_usize(0) {
            let inv_len = T::from_usize(1) / total_len;
            for u_i in u.iter_mut().take(n) {
                *u_i *= inv_len;
            }
        }

        // 2. Knot Vector Generation (De Boor / Averaging)
        // Knots size: n + k + 1. (Number of control points to find is n).
        // Domain [0, 1].
        let mut knots = vec![T::from_usize(0); n + degree + 1];

        // First k+1 knots are 0
        for k in knots.iter_mut().take(degree + 1) {
            *k = T::from_usize(0);
        }
        // Last k+1 knots are 1
        for k in knots.iter_mut().skip(n).take(degree + 1) {
            *k = T::from_usize(1);
        }

        // Internal knots: knots[j+degree] = mean(u_{j+1} ... u_{j+degree})
        // Number of internal knots = (n+degree+1) - 2(degree+1) = n - degree - 1.
        // If n = degree + 1, no internal knots.
        if n > degree + 1 {
            for j in 1..n - degree {
                let mut sum = T::from_usize(0);
                for u_val in u.iter().skip(j).take(degree) {
                    sum += *u_val;
                }
                knots[j + degree] = sum / T::from_usize(degree);
            }
        }

        // 3. System Assembly: A * C^T = P^T
        // A is n x n. A_{ij} = N_{j,k}(u_i)
        // We use dummy spline trick to eval basis functions.
        let mut a = MatrixX::<T>::new_dynamic(n, n)?;

        // Pre-allocate dummy control points (1 scalar dimension, n points)
        // Column j=1, others 0.
        let mut dummy_ctrl = MatrixX::<T>::new_dynamic(1, n)?;

        for j in 0..n {
            // Set dummy control points for Basis j
            // Clear dummy
            for k_idx in 0..n {
                *dummy_ctrl.get_mut(0, k_idx).unwrap() = T::from_usize(0);
            }
            *dummy_ctrl.get_mut(0, j).unwrap() = T::from_usize(1);

            let dummy_spline = Spline::new(knots.clone(), dummy_ctrl.clone(), degree);

            // Eval at all u_i
            for (i, &u_val) in u.iter().enumerate().take(n) {
                let val_vec = dummy_spline.eval(u_val);
                let val = *val_vec.get(0, 0).unwrap();
                *a.get_mut(i, j).unwrap() = val;
            }
        }

        // 4. Solve
        // B = P^T (n x dim)
        let mut b = MatrixX::<T>::new_dynamic(n, dim)?;
        // Transpose copy
        for i in 0..n {
            for d in 0..dim {
                *b.get_mut(i, d).unwrap() = *points.get(d, i).unwrap();
            }
        }

        // Solve A * X = B
        let qr = a.householder_qr()?;
        let x = qr.solve(&b)?;

        // C = X^T (dim x n)
        let mut ctrl_points = MatrixX::<T>::new_dynamic(dim, n)?;
        for d in 0..dim {
            for i in 0..n {
                *ctrl_points.get_mut(d, i).unwrap() = *x.get(i, d).unwrap();
            }
        }

        Ok(Spline::new(knots, ctrl_points, degree))
    }
}
