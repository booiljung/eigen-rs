extern crate eigen_rs;
use eigen_rs::core::decompositions::RealSchur;
use eigen_rs::core::matrix::MatrixX;
use std::time::Instant;

use eigen_rs::core::xpr::MatrixXpr; // Import trait

fn run_schur(n: usize) {
    println!("Running RealSchur on N={}...", n);
    let mut mat = MatrixX::<f32>::new_dynamic(n, n).unwrap();
    for i in 0..n * n {
        *mat.get_mut(i % n, i / n).unwrap() = (i % 17) as f32;
    }

    let start = Instant::now();
    // RealSchur::new computes the decomposition immediately
    let schur = RealSchur::new(&mat);
    let duration = start.elapsed();

    match schur {
        Ok(s) => {
            println!("N={} : Success in {:?}", n, duration);
            // Verify A * Q = Q * T
            let q = s.matrix_q();
            let t = s.matrix_t();

            // Force evaluation to MatrixX
            let mut lhs = MatrixX::<f32>::new_dynamic(n, n).unwrap();
            lhs.assign(&(&mat * q)).unwrap();

            let mut rhs = MatrixX::<f32>::new_dynamic(n, n).unwrap();
            rhs.assign(&(q * t)).unwrap();

            // Manual error norm calculation
            let mut diff_sq_sum: f64 = 0.0;
            for i in 0..lhs.size() {
                let r = i % n;
                let c = i / n;
                let v_lhs = *lhs.get(r, c).unwrap();
                let v_rhs = *rhs.get(r, c).unwrap();
                let d = (v_lhs - v_rhs) as f64;
                diff_sq_sum += d * d;
            }

            let error = diff_sq_sum.sqrt();
            println!("N={} : Error = {:e}", n, error);

            // Relax tolerance for f32 accumulation
            if error > 1e-1 {
                println!("FAILURE: High error!");
            }
        }
        Err(e) => println!("N={} : Failed: {:?}", n, e),
    }
}

fn main() {
    let sizes = [16, 68, 128, 256];
    // 16 and 68 were fast in reports. 128 and 256 were slow.
    for &n in &sizes {
        run_schur(n);
    }
}
