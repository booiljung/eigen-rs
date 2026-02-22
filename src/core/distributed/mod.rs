use crate::core::matrix::Matrix;
use crate::core::scalar::Scalar;
use crate::core::storage::Storage;

#[cfg(feature = "mpi_support")]
use mpi::traits::*;
#[cfg(feature = "mpi_support")]
use mpi::datatype::PartitionMut;

/// Represents a matrix distributed across multiple MPI nodes.
///
/// Current chunking strategy: rows are divided evenly (or near-evenly)
/// across the MPI ranks in the provided communicator.
#[cfg(feature = "mpi_support")]
pub struct DistributedMatrix<'comm, T: Scalar, S: Storage<T>, C: mpi::topology::Communicator> {
    /// The global number of rows in the virtual matrix.
    pub global_rows: usize,
    /// The global number of columns in the virtual matrix.
    pub global_cols: usize,
    /// The local partition (chunk) of the matrix held by this MPI rank.
    pub local_chunk: Matrix<T, S>,
    /// The MPI rank of this process.
    pub rank: i32,
    /// The total number of MPI ranks.
    pub size: i32,
    /// Reference to the MPI Communicator (usually Universe::world())
    pub comm: &'comm C,
}

#[cfg(feature = "mpi_support")]
impl<'comm, T, S, C> DistributedMatrix<'comm, T, S, C>
where
    T: Scalar + mpi::traits::Equivalence,
    S: Storage<T>,
    C: mpi::topology::Communicator,
{
    /// Creates a new DistributedMatrix from a local chunk.
    pub fn new(
        local_chunk: Matrix<T, S>,
        global_rows: usize,
        global_cols: usize,
        comm: &'comm C,
    ) -> Self {
        Self {
            local_chunk,
            global_rows,
            global_cols,
            rank: comm.rank(),
            size: comm.size(),
            comm,
        }
    }
    
    /// Returns the global dimensions of the matrix.
    pub fn global_shape(&self) -> (usize, usize) {
        (self.global_rows, self.global_cols)
    }

    /// Returns a reference to the local chunk.
    pub fn local_chunk(&self) -> &Matrix<T, S> {
        &self.local_chunk
    }

    /// Gathers all local chunks to the root node (Rank 0) and returns the assembled Matrix.
    /// Only the root node returns `Some(Matrix)`, other nodes return `None`.
    pub fn gather_to_root(&self) -> Result<Option<Matrix<T, crate::core::storage::DynamicStorage<T>>>, String> {
        let root_rank = 0;
        let mut local_vec: Vec<T> = Vec::with_capacity(self.local_chunk.size());
        
        // Flatten column-major matrix to a buffer 
        for c in 0..self.local_chunk.cols() {
            for r in 0..self.local_chunk.rows() {
                local_vec.push(*self.local_chunk.get(r, c).unwrap());
            }
        }
        
        if self.rank == root_rank {
            let total_size = self.global_rows * self.global_cols;
            let mut global_vec: Vec<T> = vec![T::default(); total_size];
            
            // Assume roughly equal chunks for simplicity in this PoC
            let chunk_size = self.local_chunk.size() as i32;
            let mut counts = vec![chunk_size; self.size as usize];
            let mut displs = vec![0_i32; self.size as usize];
            
            for i in 1..(self.size as usize) {
                displs[i] = displs[i-1] + counts[i-1];
            }
            
            // Fix remainder count for the last node if uneven
            let remainder = total_size as i32 - displs[self.size as usize - 1];
            counts[self.size as usize - 1] = remainder;

            let mut partition = PartitionMut::new(&mut global_vec, counts.clone(), &displs[..]);
            self.comm.process_at_rank(root_rank).gather_varcount_into_root(&local_vec[..], &mut partition);
            
            let mut global_mat = Matrix::<T, crate::core::storage::DynamicStorage<T>>::new_dynamic(self.global_rows, self.global_cols)?;
            let mut offset = 0;
            let mut current_row_offset = 0;
            
            // Unpack gathered chunks into the global column-major matrix
            for rank_i in 0..(self.size as usize) {
                let rank_num_elements = counts[rank_i] as usize;
                let rank_rows = rank_num_elements / self.global_cols;
                
                let mut local_idx = 0;
                for c in 0..self.global_cols {
                    for r in 0..rank_rows {
                        if let Some(val) = global_mat.get_mut(current_row_offset + r, c) {
                            *val = global_vec[offset + local_idx];
                        }
                        local_idx += 1;
                    }
                }
                
                offset += rank_num_elements;
                current_row_offset += rank_rows;
            }
            
            Ok(Some(global_mat))
            
        } else {
            self.comm.process_at_rank(root_rank).gather_varcount_into(&local_vec[..]);
            Ok(None)
        }
    }
}
