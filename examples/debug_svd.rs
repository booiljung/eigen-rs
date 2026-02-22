use eigen_rs::core::decompositions::cuda_bridge::CudaDecompositionExt;
use eigen_rs::core::storage::DynamicStorage;
use eigen_rs::Matrix;

fn main() {
    println!("Starting SVD debug...");
    let mut mat = Matrix::<f32, DynamicStorage<f32>>::new_dynamic(128, 128).unwrap();
    for i in 0..128 {
        for j in 0..128 {
            *mat.get_mut(i, j).unwrap() = (i as f32) + (j as f32 * 0.1);
        }
    }

    println!("Calling try_svd_cuda...");
    let result = mat.try_svd_cuda();
    match result {
        Ok(Some((u, s, v))) => {
            println!(
                "Success! U: {}x{}, S: {}x{}, V: {}x{}",
                u.rows(),
                u.cols(),
                s.rows(),
                s.cols(),
                v.rows(),
                v.cols()
            );
        }
        Ok(None) => println!("Returned Ok(None) - skipped by heuristics"),
        Err(e) => println!("Returned Err: {}", e),
    }
    println!("Done!");
}
