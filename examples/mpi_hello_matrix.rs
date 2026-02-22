use mpi::traits::*;
use eigen_rs::core::matrix::MatrixX;
use eigen_rs::core::distributed::DistributedMatrix;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    let global_rows = 8;
    let global_cols = 4;
    
    // Each node acts on a chunk of rows
    let chunk_rows = global_rows / (size as usize);
    let my_rows = if rank == size - 1 {
        // give the remainder to the last node
        global_rows - (chunk_rows * (size as usize - 1))
    } else {
        chunk_rows
    };
    
    // Create local chunk
    let mut local_mat = MatrixX::<f64>::new_dynamic(my_rows, global_cols).unwrap();
    let val = rank as f64 + 1.0;
    local_mat.set_constant(val);
    
    println!("Rank {}/{} created local chunk: {}x{} populated with {}", rank, size, my_rows, global_cols, val);
    
    // Abstract local memory into DistributedMatrix
    let dist_mat = DistributedMatrix::new(local_mat, global_rows, global_cols, &world);
    
    // Gather to root
    if let Ok(result) = dist_mat.gather_to_root() {
        if let Some(mut global_mat) = result {
             if rank == 0 {
                 println!("Root gathered the distributed matrix {}x{}", global_mat.rows(), global_mat.cols());
                 
                 // Display element samples from gathered matrix
                 for i in 0..global_rows {
                     println!("Global Row {}: {}", i, global_mat.get(i, 0).unwrap());
                 }
                 
                 println!("Distributed Computing successful.");
             }
        }
    } else {
         println!("Rank {} failed to gather.", rank);
    }
}
