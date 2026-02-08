use criterion::{black_box, criterion_group, criterion_main, Criterion};
use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::sparse::solvers::{SimplicialLLT, SparseLU};
use eigen_rs::core::sparse::sparse_matrix::{SparseMatrix, StorageOrder, Triplet};
#[cfg(feature = "cuda")]
use eigen_rs::core::sparse::{cuda_ops, CudaSparseStorage};
#[cfg(feature = "cuda")]
use eigen_rs::core::storage::CudaStorage;
use eigen_rs::core::storage::Storage;

fn bench_sparse_spmv(c: &mut Criterion) {
    let size = 1000;
    let nnz_per_row = 10;

    let mut a = SparseMatrix::<f64>::new(size, size, StorageOrder::ColMajor);
    let mut triplets = Vec::new();
    for i in 0..size {
        for k in 0..nnz_per_row {
            let j = (i + k) % size;
            triplets.push(Triplet::new(i, j, (i + j) as f64));
        }
    }
    a.set_from_triplets(triplets);

    let x = MatrixX::<f64>::new_dynamic(size, 1).unwrap();
    let mut y = MatrixX::<f64>::new_dynamic(size, 1).unwrap();

    let mut group = c.benchmark_group("Sparse SpMV");
    group.bench_function("f64_1000x1000_nnz10_cpu", |b| {
        b.iter(|| (black_box(&a) * black_box(&x)).unwrap());
    });

    #[cfg(feature = "cuda")]
    {
        let mut cuda_a = CudaSparseStorage::<f64>::new(size, size, a.non_zeros()).unwrap();
        // Extract raw indices for transfer. We need i32 for cuSPARSE.
        let values = a.values().to_vec();
        let col_indices: Vec<i32> = a.inner_indices().iter().map(|&x| x as i32).collect();
        let row_offsets: Vec<i32> = a.outer_starts().iter().map(|&x| x as i32).collect();
        cuda_a
            .copy_from_host(&values, &col_indices, &row_offsets)
            .unwrap();

        let mut cuda_x = CudaStorage::<f64>::new(size, 1).unwrap();
        cuda_x.copy_from_host(x.storage().data()).unwrap();

        let mut cuda_y = CudaStorage::<f64>::new(size, 1).unwrap();
        cuda_y.copy_from_host(&vec![0.0; size]).unwrap();

        let handle = cuda_ops::CusparseHandle::new().unwrap();

        group.bench_function("f64_1000x1000_nnz10_gpu", |b| {
            b.iter(|| {
                cuda_ops::spmv_cuda(
                    &handle,
                    black_box(&cuda_a),
                    black_box(&cuda_x),
                    black_box(&mut cuda_y),
                    1.0,
                    0.0,
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

fn bench_sparse_lu(c: &mut Criterion) {
    let size = 500;
    let mut a = SparseMatrix::<f64>::new(size, size, StorageOrder::ColMajor);
    let mut triplets = Vec::new();
    for i in 0..size {
        triplets.push(Triplet::new(i, i, size as f64));
        if i > 0 {
            triplets.push(Triplet::new(i, i - 1, -1.0));
        }
        if i < size - 1 {
            triplets.push(Triplet::new(i, i + 1, -1.0));
        }
    }
    a.set_from_triplets(triplets);

    let mut group = c.benchmark_group("Sparse LU");
    group.bench_function("f64_500x500", |b| {
        b.iter(|| {
            let mut lu = SparseLU::new();
            lu.compute(black_box(&a)).unwrap();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_sparse_spmv, bench_sparse_lu);
criterion_main!(benches);
