use eigen_rs::core::matrix::MatrixX;
use std::time::Instant;

fn bench_qr(n: usize, iterations: usize) -> f64 {
    let mut mat = MatrixX::<f64>::new_dynamic(n, n).unwrap();
    // Fill with values
    for i in 0..n {
        for j in 0..n {
            *mat.get_mut(i, j).unwrap() = ((i + j) % 100) as f64;
        }
        *mat.get_mut(i, i).unwrap() += 10.0; // Diagonally dominant / verification
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _qr = mat.householder_qr();
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn bench_llt(n: usize, iterations: usize) -> f64 {
    let mut mat = MatrixX::<f64>::new_dynamic(n, n).unwrap();
    for i in 0..n {
        for j in 0..n {
            *mat.get_mut(i, j).unwrap() = ((i + j) % 100) as f64;
        }
        *mat.get_mut(i, i).unwrap() += 10.0;
    }
    let t = mat.transpose();
    let sym_expr = &mat * &t; // SPD

    // Evaluate to concrete matrix to isolate decomposition time
    let mut sym = MatrixX::<f64>::new_dynamic(n, n).unwrap();
    sym.assign(&sym_expr).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _llt = sym.llt();
    }
    start.elapsed().as_secs_f64() / iterations as f64
}

fn main() {
    println!("| Operation | N | Time (us) |");
    println!("|---|---|---|");

    let sizes = [288, 289, 290, 314, 315, 316];
    let iters = 100;

    for &n in &sizes {
        let t_qr = bench_qr(n, iters) * 1e6;
        println!("| QR | {} | {:.2} |", n, t_qr);
    }

    println!("|---|---|---|");

    for &n in &sizes {
        let t_llt = bench_llt(n, iters) * 1e6;
        println!("| LLT | {} | {:.2} |", n, t_llt);
    }
}
