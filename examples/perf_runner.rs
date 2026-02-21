use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::xpr::MatrixXpr;
use std::hint::black_box;
use std::time::Instant;

fn init_matrix(m: &mut MatrixX<f32>) {
    let rows = m.rows();
    for i in 0..m.size() {
        let r = i % rows;
        let c = i / rows;
        *m.get_mut(r, c).unwrap() = ((r + c * rows) % 17) as f32 / 10.0;
    }
}

fn matches_filter(name: &str, filter: &str) -> bool {
    if filter.is_empty() {
        return true;
    }
    for part in filter.split('|') {
        if !part.is_empty() && name.contains(part) {
            return true;
        }
    }
    false
}

fn init_vector(v: &mut MatrixX<f32>) {
    for i in 0..v.size() {
        *v.get_mut(i, 0).unwrap() = (i % 17) as f32 / 10.0;
    }
}

fn bench_matmul(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    let mut b = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    let mut c = MatrixX::<f32>::new_dynamic(size, size).unwrap();

    // Randomize
    for i in 0..size*size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 100) as f32;
        *b.get_mut(i % size, i / size).unwrap() = ((i + 1) % 100) as f32;
    }

    let iterations = if size < 64 { 100 } else if size < 256 { 10 } else { 1 };

    // Warm up
    for _ in 0..5 {
        c.assign_product(&(&a * &b)).unwrap();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        c.assign_product(&(&a * &b)).unwrap();
    }
    let duration = start.elapsed().as_nanos();

    println!("MatMul,{},{}", size, duration / iterations as u128);
}

fn bench_llt(size: usize) {
    // Make SPD
    let mut temp = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size * size {
        *temp.get_mut(i % size, i / size).unwrap() = (i % 100) as f32;
    }
    // M = A * A.T + I * size
    let mut m = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    m.assign_product(&(&temp * &temp.transpose())).unwrap();
    for i in 0..size {
        *m.get_mut(i, i).unwrap() += size as f32;
    }

    let iterations = if size < 64 { 100 } else if size < 256 { 10 } else { 1 };

    // Warm up
    for _ in 0..5 {
        let _ = m.llt();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = m.llt();
    }
    let duration = start.elapsed().as_nanos();

    println!("LLT,{},{}", size, duration / iterations as u128);
}

fn bench_svd(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size * size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 100) as f32;
    }

    let iterations = if size < 64 { 100 } else if size < 256 { 10 } else { 1 };

    // Warm up
    for _ in 0..2 {
        let _ = a.jacobi_svd();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.jacobi_svd();
    }
    let duration = start.elapsed().as_nanos();

    println!("SVD,{},{}", size, duration / iterations as u128);
}

fn bench_eigenvalues(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size * size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 100) as f32;
    }
    // Make symmetric M = A + A.T
    let mut m = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size {
        for j in 0..size {
            let val = a.get(i, j).unwrap() + a.get(j, i).unwrap();
            *m.get_mut(i, j).unwrap() = val;
        }
    }

    let iterations = if size < 64 { 100 } else if size < 256 { 10 } else { 1 };

    // Warm up
    for _ in 0..2 {
        let _ = m.self_adjoint_eigen_solver(true); // compute vectors too? C++ bench implies yes
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = m.self_adjoint_eigen_solver(true);
    }
    let duration = start.elapsed().as_nanos();

    println!("EigenValues,{},{}", size, duration / iterations as u128);
}

fn bench_vector_ops(size: usize) {
    let mut v1 = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    init_vector(&mut v1);
    let mut v2 = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    init_vector(&mut v2);
    // v2 += 1.0
    for i in 0..size {
        *v2.get_mut(i, 0).unwrap() += 1.0;
    }

    let iterations = if size < 1000 {
        10000
    } else if size < 100000 {
        1000
    } else {
        100
    };

    // Dot
    let start = Instant::now();
    let mut dot_res = 0.0;
    for _ in 0..iterations {
        // C++: res += v1.dot(v2); asm barrier
        // Rust: match this
        let res = v1.dot(&v2);
        dot_res += res;
        unsafe { *v1.get_mut(0, 0).unwrap() += 1e-6; }
    }
    let duration = start.elapsed().as_nanos();
    println!(
        "VecDot,{},{}",
        size,
        duration / iterations as u128
    );

    // Norm
    let start = Instant::now();
    let mut norm_res = 0.0;
    for _ in 0..iterations {
        norm_res += v1.norm();
        unsafe {
            *v1.get_mut(0, 0).unwrap() += 1e-6;
        }
    }
    let duration = start.elapsed().as_nanos();
    println!(
        "VecNorm,{},{}",
        size,
        duration / iterations as u128
    );
}

fn bench_matrix_arithmetic(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    init_matrix(&mut a);
    let mut b = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    init_matrix(&mut b);
    let mut c = MatrixX::<f32>::new_dynamic(size, size).unwrap();

    let iterations = if size < 64 { 1000 } else if size < 512 { 1000 } else { 100 };

    // Add
    let start = Instant::now();
    for _ in 0..iterations {
        c.assign(&(&a + &b)).unwrap();
        // C++ Bench did NOT modify 'a' in final version (lines 128+ in eigen_bench.cpp).
        // C++: c.noalias() = a + b;
        // So no modification needed here.
    }
    let duration = start.elapsed().as_nanos();
    println!(
        "MatAdd,{},{},{}",
        size,
        duration / iterations as u128,
        c.sum()
    );

    // Scale
    init_matrix(&mut a); // Reset
    let start = Instant::now();
    for _ in 0..iterations {
        a.scale(1.01);
    }
    let duration = start.elapsed().as_nanos();
    println!(
        "MatScale,{},{},{}",
        size,
        duration / iterations as u128,
        a.sum()
    );
}

fn bench_geometry(filter: &str) {
    use eigen_rs::core::matrix::{FixedStorage, Matrix, Vector3};
    use eigen_rs::core::xpr::MatrixXpr;
    use eigen_rs::geometry::Quaternion;

    let iterations = 100_000_000;

    // Cross Product (3D)
    let mut v1 = Matrix::<f32, FixedStorage<f32, 3, 1, 3>>::from_array([1.0, 2.0, 3.0]);
    let v2 = Matrix::<f32, FixedStorage<f32, 3, 1, 3>>::from_array([4.0, 5.0, 6.0]);

    let start = Instant::now();
    for _ in 0..iterations {
        let v3 = v1.cross(&v2);
        *v1.get_mut(0, 0).unwrap() += v3.get(0, 0).unwrap() * 0.001;
    }
    let duration = start.elapsed().as_nanos();
    println!("Cross3D,3,{},{}", duration / iterations as u128, v1.sum());

    // Quaternion Mul
    let mut q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    let q2 = Quaternion::new(0.0, 1.0, 0.0, 0.0);

    let start = Instant::now();
    for _ in 0..iterations {
        q1 = q1 * q2;
    }
    let duration = start.elapsed().as_nanos();
    let q_sum = q1.w() + q1.x() + q1.y() + q1.z();
    println!("QuatMul,4,{},{}", duration / iterations as u128, q_sum);

    // QuatRot
    if matches_filter("QuatRot", filter) {
        let mut v_rot = Matrix::<f32, FixedStorage<f32, 3, 1, 3>>::from_array([1.0, 0.0, 0.0]);
        let mut v_res = Matrix::<f32, FixedStorage<f32, 3, 1, 3>>::zeros();
        let start = Instant::now();
        for _ in 0..iterations {
             *v_rot.get_mut(0, 0).unwrap() += 1e-6;
             v_res = &q1 * &v_rot;
        }
        let duration = start.elapsed().as_nanos();
        println!("QuatRot,3,{},{}", duration / iterations as u128, v_res.sum());
    }
}

fn bench_dense_decomp_extra(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    // fill
    for i in 0..size * size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 17) as f32;
    }

    let iterations = if size < 64 { 100 } else if size < 256 { 10 } else { 1 };

    // LU
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.partial_piv_lu();
    }
    let duration = start.elapsed().as_nanos();
    println!("LU,{},{}", size, duration / iterations as u128);

    // QR
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.householder_qr();
    }
    let duration = start.elapsed().as_nanos();
    println!("QR,{},{}", size, duration / iterations as u128);
}

fn bench_decompositions_advanced(size: usize, filter: &str) {
    use eigen_rs::core::decompositions::{
        GeneralizedEigenSolver, HessenbergDecomposition, RealSchur, Tridiagonalization,
    };

    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    // fill
    for i in 0..size * size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 17) as f32;
    }

    // Symmetric for LDLT, Tridiagonal
    let mut sym = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    sym.assign_product(&(&a * &a.transpose())).unwrap();

    // PD for GeneralizedEigen B
    let mut pd = sym.clone();
    for i in 0..size {
        *pd.get_mut(i, i).unwrap() += size as f32;
    }

    let iterations = if size < 64 { 20 } else { 1 };

    // 1. Determinant
    if matches_filter("Determinant", filter) {
        let start = Instant::now();
        let mut det_sum = 0.0;
        for _ in 0..iterations {
            // eigen-rs requires explicit LU for N > 3
            det_sum += a.partial_piv_lu().unwrap().determinant();
        }
        let duration = start.elapsed().as_nanos();
        println!("Determinant,{},{}", size, duration / iterations as u128);
        black_box(det_sum);
    }

    // 2. LDLT
    if matches_filter("LDLT", filter) {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sym.ldlt();
        }
        let duration = start.elapsed().as_nanos();
        println!("LDLT,{},{}", size, duration / iterations as u128);
    }

    // 3. Hessenberg
    if matches_filter("Hessenberg", filter) {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = HessenbergDecomposition::new(&a);
        }
        let duration = start.elapsed().as_nanos();
        println!("Hessenberg,{},{}", size, duration / iterations as u128);
    }

    // 4. Tridiagonalization
    if matches_filter("Tridiagonal", filter) {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Tridiagonalization::new(&sym);
        }
        let duration = start.elapsed().as_nanos();
        println!("Tridiagonal,{},{}", size, duration / iterations as u128);
    }

    // 5. GeneralizedSelfAdjointEigenSolver
    if matches_filter("GeneralizedEigen", filter) {
        use eigen_rs::core::decompositions::GeneralizedSelfAdjointEigenSolver;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = GeneralizedSelfAdjointEigenSolver::new(&sym, &pd, false);
        }
        let duration = start.elapsed().as_nanos();
        println!("GeneralizedEigen,{},{}", size, duration / iterations as u128);
    }

    // 6. RealSchur
    if matches_filter("RealSchur", filter) {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = RealSchur::new(&a);
        }
        let duration = start.elapsed().as_nanos();
        println!("RealSchur,{},{}", size, duration / iterations as u128);
    }

    // 7. BDCSVD
    if matches_filter("BDCSVD", filter) {
        use eigen_rs::core::decompositions::bdc_svd::BDCSVD;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = BDCSVD::new(&a);
        }
        let duration = start.elapsed().as_nanos();
        println!("BDCSVD,{},{}", size, duration / iterations as u128);
    }
}

fn bench_geometry_advanced(filter: &str) {
    use eigen_rs::core::geometry::{AngleAxis, EulerAngles, Scaling, Transform3, Translation};
    use eigen_rs::core::matrix::{Vector3, Matrix3};
    use std::ops::Mul; // For .mul() calls or * syntax
    
    let iterations = 100_000_000;
    
    // Transform
    // C++: Translation * Scaling
    let tr = Translation::new(Vector3::from_array([1.0, 2.0, 3.0]));
    let sc = Scaling::uniform(0.5);
    let t_tr = Transform3::from_translation(&tr);
    let t_sc = Transform3::from_scaling(&sc);
    let t = t_tr.mul(&t_sc);
    
    let mut v = Vector3::<f32>::from_array([0.5, 0.5, 0.5]); 
    
    let start = Instant::now();
    for _ in 0..iterations {
        v = t.transform_point(&v);
    }
    let duration = start.elapsed().as_nanos();
    println!("Transform,3,{}", duration / iterations as u128);
    black_box(v);

    // TransformMul
    if matches_filter("TransformMul", filter) {
        let mut t_res = Transform3::identity();
        let start = Instant::now();
        for _ in 0..iterations {
            t_res = t_tr.mul(&t_sc);
        }
        let duration = start.elapsed().as_nanos();
        println!("TransformMul,4,{}", duration / iterations as u128);
        black_box(t_res.matrix().get(0,0));
    }

    // Translation
    let mut v = Vector3::<f32>::from_array([0.5, 0.5, 0.5]);
    let start = Instant::now();
    for _ in 0..iterations {
        // v = tr * v
        v = (&tr).mul(&v);
    }
    let duration = start.elapsed().as_nanos();
    println!("Translation,3,{}", duration / iterations as u128);

    // Scaling
    let mut v = Vector3::<f32>::from_array([0.5, 0.5, 0.5]);
    let start = Instant::now();
    for _ in 0..iterations {
        v = (&sc).mul(&v);
    }
    let duration = start.elapsed().as_nanos();
    println!("Scaling,3,{}", duration / iterations as u128);

    // AngleAxis -> Matrix
    let aa = AngleAxis::new(0.5, Vector3::from_array([1.0, 0.0, 0.0]));
    let mut rot = Matrix3::<f32>::identity();
    
    let mut v_dummy = 0.0;
    let start = Instant::now();
    for _ in 0..iterations {
        rot = aa.to_rotation_matrix();
        let val = *rot.get(0, 0).unwrap(); 
        v_dummy += val * 1e-6; 
    }
    let duration = start.elapsed().as_nanos();
    println!("AngleAxis,3,{}", duration / iterations as u128);
    black_box(v_dummy);

    // EulerAngles
    let start = Instant::now();
    for _ in 0..iterations {
        let _euler = EulerAngles::from_rotation_matrix(&rot);
    }
    let duration = start.elapsed().as_nanos();
    println!("EulerAngles,3,{}", duration / iterations as u128);
}

fn bench_sparse_advanced(size: usize) {
    use eigen_rs::core::sparse::solvers::{SparseLU, SparseQR};
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::storage::Storage;
    
    // Generate sparse matrix
    let mut sp = SparseMatrix::<f32>::new(size, size, StorageOrder::ColMajor);
    let density = 0.05;
    let nnz = (size as f32 * size as f32 * density) as usize;
    let mut triplets = Vec::with_capacity(nnz + size);
    
    for i in 0..nnz {
        let r = (i * 17) % size;
        let c = (i * 23) % size;
        triplets.push(Triplet::new(r, c, 1.0));
    }
    for i in 0..size {
        triplets.push(Triplet::new(i, i, 2.0));
    }
    sp.set_from_triplets(triplets);
    
    // new_random not available. Use set random.
    let mut b = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    for i in 0..size {
        *b.get_mut(i, 0).unwrap() = (i % 100) as f32 / 10.0;
    }
    
    let iterations = if size < 64 { 20 } else { 1 };
    
    // SparseLU
    {
        let mut lu = SparseLU::new();
        let start = Instant::now();
        for _ in 0..iterations {
            if let Ok(_) = lu.compute(&sp) {
                let _ = lu.solve(&b);
            }
        }
        let duration = start.elapsed().as_nanos();
        println!("SparseLU,{},{}", size, duration / iterations as u128);
    }
    
    // SparseQR
    {
        let mut qr = SparseQR::new();
        let start = Instant::now();
        for _ in 0..iterations {
            if let Ok(_) = qr.compute(&sp) {
                let _ = qr.solve(&b);
            }
        }
        let duration = start.elapsed().as_nanos();
        println!("SparseQR,{},{}", size, duration / iterations as u128);
    }

    // SparseView (Dense -> Sparse Conversion)
    {
        use eigen_rs::core::matrix::MatrixX;
        let mut dense = MatrixX::<f32>::new_dynamic(size, size).unwrap();
        // Make it sparse-ish (same logic as C++)
        for i in 0..size*size {
            // "if (rand() % 100 > 5) dense(i%size, i/size) = 0.0f;"
            // This means 5% density (95% zeros).
            let r = i % size;
            let c = i / size;
             if (i % 100) > 5 {
                *dense.get_mut(r, c).unwrap() = 0.0;
             } else {
                *dense.get_mut(r, c).unwrap() = (i % 10) as f32;
             }
        }

        let start = Instant::now();
        // C++ runs iterations * 10
        let loop_iters = iterations * 10;
        for _ in 0..loop_iters {
            let mut values = Vec::with_capacity(size * size / 20);
            let mut inner_indices = Vec::with_capacity(size * size / 20);
            let mut outer_starts = Vec::with_capacity(size + 1);

            let mut nnz = 0;
            for c in 0..size {
                outer_starts.push(nnz);
                for r in 0..size {
                     let val = unsafe { *dense.get_unchecked(r, c) };
                     if val != 0.0 {
                         values.push(val);
                         inner_indices.push(r);
                         nnz += 1;
                     }
                }
            }
            outer_starts.push(nnz);

            let s = SparseMatrix::<f32>::from_raw(
                size, size, values, inner_indices, outer_starts, StorageOrder::ColMajor
            );
            black_box(s.non_zeros());
        }
        let duration = start.elapsed().as_nanos();
        println!("SparseView,{},{}", size, duration / loop_iters as u128);
    }
}

fn bench_optimization() {
    // Placeholder
}
fn bench_sparse(size: usize) {
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::xpr::MatrixXpr; // Import for sum()

    let mut sp = SparseMatrix::new(size, size, StorageOrder::RowMajor);
    let density = 0.05;
    let nnz = (size as f32 * size as f32 * density) as usize;

    let mut triplets = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let r = (k * 17) % size;
        let c = (k * 29) % size;
        let v = (k % 100) as f32 / 10.0;
        triplets.push(Triplet::new(r, c, v));
    }
    sp.set_from_triplets(triplets);

    let mut v = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    init_vector(&mut v);
    let mut res = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    res.set_zero();

    // SpMV iterations match C++: 1000 for small
    let iterations = if size < 256 { 1000 } else { 100 };

    // SpMV
    let mut start = Instant::now();
    for _ in 0..iterations {
        sp.mul_dense_into(&v, &mut res).unwrap();
    }
    let duration = start.elapsed().as_nanos();
    println!(
        "SpMV,{},{},{}",
        size,
        duration / iterations as u128,
        0.0 // no res_sum to avoid overhead
    );

    // SpMM
    let mut m = MatrixX::<f32>::new_dynamic(size, 32).unwrap();
    for i in 0..size {
        for j in 0..32 {
            *m.get_mut(i, j).unwrap() = ((i + j) % 100) as f32 / 10.0;
        }
    }
    let mut res_m = MatrixX::<f32>::new_dynamic(size, 32).unwrap();
    res_m.set_zero();

    start = Instant::now();
    for _ in 0..iterations {
        sp.mul_dense_into(&m, &mut res_m).unwrap();
    }
    let duration = start.elapsed().as_nanos();
    println!(
        "SpMM_Dense,{},{},{}",
        size,
        duration / iterations as u128,
        0.0
    );
}

fn bench_inverse(size: usize) {
    let mut a = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    for i in 0..size * size {
        *a.get_mut(i % size, i / size).unwrap() = (i % 17) as f32;
    }
    for i in 0..size {
        *a.get_mut(i, i).unwrap() += size as f32;
    }

    let lu = a.partial_piv_lu().unwrap();

    let mut identity = MatrixX::<f32>::new_dynamic(size, size).unwrap();
    // Set identity
    for i in 0..size {
        *identity.get_mut(i, i).unwrap() = 1.0;
    }

    let iterations = if size < 64 { 10 } else { 1 };

    // Warm up
    for _ in 0..2 {
        let _ = lu.solve(&identity);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = lu.solve(&identity);
    }
    let duration = start.elapsed().as_nanos();
    println!("Inverse,{},{}", size, duration / iterations as u128);
}

fn bench_sparse_iterative(size: usize) {
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::iterative_solvers::{ConjugateGradient, BiCGSTAB, IterativeSolver};
    use eigen_rs::core::matrix::MatrixX; // Explicit import if needed, though usually available
    
    // perf_runner uses local helper init_vector
    
    let density = 0.05;
    let nnz = (size as f32 * size as f32 * density) as usize;
    
    // A = B * B^T + 10I
    let mut triplets = Vec::with_capacity(nnz + size);
    for k in 0..nnz {
        let r = (k * 17) % size;
        let c = (k * 29) % size;
        if r != c {
            let v = 0.1;
            triplets.push(Triplet::new(r, c, v));
            triplets.push(Triplet::new(c, r, v)); // Symmetric
        }
    }
    for i in 0..size {
        triplets.push(Triplet::new(i, i, 10.0)); // Diagonal
    }
    
    let mut sp = SparseMatrix::<f32>::new(size, size, StorageOrder::RowMajor);
    sp.set_from_triplets(triplets);
    
    let mut b = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    // init_vector(&mut b); // Helper function in file
    for i in 0..b.size() {
        *b.get_mut(i, 0).unwrap() = (i % 17) as f32 / 10.0;
    }
    
    let iterations = if size < 64 { 20 } else { 5 };
    
    // Conjugate Gradient
    {
        let mut cg = ConjugateGradient::<f32, SparseMatrix<f32>>::new()
            .with_max_iterations(100)
            .with_tolerance(1e-6);
            
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            cg.compute(&sp);
            let _ = cg.solve(&b);
        }
        let duration = start.elapsed().as_nanos();
        println!("SparseCG,{},{}", size, duration / iterations as u128);
    }
    
    // BiCGSTAB
    {
        let mut bicg = BiCGSTAB::<f32, SparseMatrix<f32>>::new()
            .with_max_iterations(100)
            .with_tolerance(1e-6);
            
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            bicg.compute(&sp);
            let _ = bicg.solve(&b);
        }
        let duration = start.elapsed().as_nanos();
        println!("SparseBiCGSTAB,{},{}", size, duration / iterations as u128);
    }
}

fn bench_sparse_cholesky(size: usize) {
    use eigen_rs::core::sparse::{SparseMatrix, StorageOrder, Triplet};
    use eigen_rs::core::sparse::solvers::{SimplicialLLT, SimplicialLDLT};
    use eigen_rs::core::matrix::MatrixX;
    
    let density = 0.05;
    let nnz = (size as f32 * size as f32 * density) as usize;
    
    // A = B * B^T + 2I
    let mut triplets = Vec::with_capacity(nnz);
    for k in 0..nnz {
        let r = (k * 17) % size;
        let c = (k * 29) % size;
        let v = 0.5; // Avoid huge numbers
        triplets.push(Triplet::new(r, c, v));
    }
    
    let mut b_mat = SparseMatrix::<f32>::new(size, size, StorageOrder::ColMajor);
    b_mat.set_from_triplets(triplets);
    
    // A = B * B^T (Sparse * SparseTranspose)
    // SparseMatrix::mul requires refs.
    // Transpose returns a new matrix or expression? 
    // SparseMatrix::transpose() returns SparseMatrix with flipped order.
    // We need same order for multiplication.
    let bt = b_mat.transpose_reordered();
    let mut a: SparseMatrix<f32> = (&b_mat * &bt).unwrap();
    
    // Add Identity to diagonal to ensure PD
    let mut i_triplets = Vec::with_capacity(size);
    for i in 0..size {
        i_triplets.push(Triplet::new(i, i, 2.0));
    }
    let mut ident = SparseMatrix::<f32>::new(size, size, StorageOrder::ColMajor);
    ident.set_from_triplets(i_triplets);
    
    let sp: SparseMatrix<f32> = (&a + &ident).unwrap();
    
    let mut b = MatrixX::<f32>::new_dynamic(size, 1).unwrap();
    for i in 0..size {
        *b.get_mut(i, 0).unwrap() = (i % 17) as f32 / 10.0;
    }
    
    let iterations = if size < 64 { 20 } else { 5 };
    
    // SimplicialLLT
    {
        let mut llt = SimplicialLLT::<f32>::new();
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            llt.compute(&sp).unwrap();
            let _ = llt.solve(&b);
        }
        let duration = start.elapsed().as_nanos();
        println!("SimplicialLLT,{},{}", size, duration / iterations as u128);
    }
    
    // SimplicialLDLT
    {
        let mut ldlt = SimplicialLDLT::<f32>::new();
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            ldlt.compute(&sp).unwrap();
            let _ = ldlt.solve(&b);
        }
        let duration = start.elapsed().as_nanos();
        println!("SimplicialLDLT,{},{}", size, duration / iterations as u128);
    }
}

fn main() {
    println!("Running Performance Test (Sweep 16..384 + Large Vecs)");
    let args: Vec<String> = std::env::args().collect();

    // Defaults
    let mut sizes: Vec<usize> = Vec::new();
    let mut small_sizes: Vec<usize> = Vec::new();

    let mut filter = String::new();

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--sizes" && i + 1 < args.len() {
            let parts: Vec<&str> = args[i + 1].split(',').collect();
            for p in parts {
                if let Ok(val) = p.trim().parse::<usize>() {
                    sizes.push(val);
                }
            }
            i += 2;
        } else if args[i] == "--small-sizes" && i + 1 < args.len() {
            let parts: Vec<&str> = args[i + 1].split(',').collect();
            for p in parts {
                if let Ok(val) = p.trim().parse::<usize>() {
                    small_sizes.push(val);
                }
            }
            i += 2;
        } else if args[i] == "--filter" && i + 1 < args.len() {
             filter = args[i+1].clone();
             i += 2;
        } else {
            i += 1;
        }
    }

    if sizes.is_empty() {
        // 1. Standard Sweep: 4 to 384, stride 13
        sizes = (4..=384).step_by(13).collect();
        sizes.extend_from_slice(&[8, 12, 16, 64, 128, 256]);
    }

    if small_sizes.is_empty() {
        // 2. Small Sweep: 4 to 64, stride 7
        small_sizes = (4..=64).step_by(7).collect();
        small_sizes.extend_from_slice(&[8, 12, 16, 32, 64]);
    }

    sizes.sort();
    sizes.dedup();

    small_sizes.sort();
    small_sizes.dedup();

    for &size in &sizes {
        if matches_filter("MatMul", &filter) { bench_matmul(size); }
        if matches_filter("LLT", &filter) { bench_llt(size); }
        if matches_filter("QR", &filter) || matches_filter("LU", &filter) { bench_dense_decomp_extra(size); }
        
        // Jacobi SVD (Slow for large N, but useful for comparison)
        if matches_filter("SVD", &filter) { bench_svd(size); }
        
        // Pass filter to advanced decompositions (RealSchur, Hessenberg, etc.)
        bench_decompositions_advanced(size, &filter);
        
        if matches_filter("MatAdd", &filter) || matches_filter("MatScale", &filter) { bench_matrix_arithmetic(size); }
        if matches_filter("VecDot", &filter) || matches_filter("VecNorm", &filter) { bench_vector_ops(size); }
        if matches_filter("SpMV", &filter) || matches_filter("SpMM", &filter) { bench_sparse(size); }
        if matches_filter("SparseAdvanced", &filter) 
            || matches_filter("SparseLU", &filter) 
            || matches_filter("SparseQR", &filter) 
            || matches_filter("SparseView", &filter) 
        { 
            bench_sparse_advanced(size); 
        }
        if matches_filter("SparseCG", &filter) || matches_filter("SparseBiCGSTAB", &filter) { bench_sparse_iterative(size); }
        if matches_filter("SimplicialLLT", &filter) || matches_filter("SimplicialLDLT", &filter) { bench_sparse_cholesky(size); }
        if matches_filter("Inverse", &filter) { bench_inverse(size); }
    }

    // 3. Large Vector Benchmarks (Fixed for now, or could be added to args)
    let large_vec_sizes = [4096, 16384, 65536]; // Removed 1M to match args
    for &size in &large_vec_sizes {
        if matches_filter("VecDot", &filter) || matches_filter("VecNorm", &filter) {
            bench_vector_ops(size);
        }
    }

    for &size in &small_sizes {
        if matches_filter("SVD", &filter) { bench_svd(size); }
        if matches_filter("Eigenvalues", &filter) { bench_eigenvalues(size); }
    }

    if matches_filter("Geometry", &filter) 
        || matches_filter("AngleAxis", &filter) 
        || matches_filter("EulerAngles", &filter) 
        || matches_filter("Cross3D", &filter) 
        || matches_filter("QuatMul", &filter) 
        || matches_filter("QuatRot", &filter) 
        || matches_filter("Transform", &filter) 
        || matches_filter("TransformMul", &filter) 
        || matches_filter("Translation", &filter) 
        || matches_filter("Scaling", &filter) 
    {
        bench_geometry(&filter);
        bench_geometry_advanced(&filter);
    }
    bench_optimization();
}
