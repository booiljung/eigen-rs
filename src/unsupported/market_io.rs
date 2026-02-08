//! Matrix Market I/O
//!
//! Support for reading and writing Matrix Market (.mtx) files.
//! Currently supports:
//! - Format: Coordinate
//! - DataType: Real
//! - Structure: General / Symmetric (Partial)

use alloc::vec::Vec;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

use crate::core::scalar::Scalar;
use crate::core::sparse::SparseMatrix;
// use crate::core::matrix::Matrix; // For Dense support later

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum MtxFormat {
    Coordinate,
    Array,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum MtxDataType {
    Real,
    Complex,
    Integer,
    Pattern,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum MtxSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

/// load a SparseMatrix from a Matrix Market file.
///
/// Supported: `%%MatrixMarket matrix coordinate real general` (and symmetric)
pub fn load_matrix_market<P: AsRef<Path>, T: Scalar + FromStr>(
    path: P,
) -> Result<SparseMatrix<T>, String>
where
    <T as FromStr>::Err: std::fmt::Display,
{
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // 1. Parse Header
    let first_line = lines
        .next()
        .ok_or("Empty file")?
        .map_err(|e| e.to_string())?;
    if !first_line.starts_with("%%MatrixMarket matrix") {
        return Err("Invalid MatrixMarket header".to_string());
    }

    let parts: Vec<&str> = first_line.split_whitespace().collect();
    // %%MatrixMarket matrix coordinate real general
    if parts.len() < 5 {
        return Err("Incomplete MatrixMarket header".to_string());
    }

    let _format = match parts[2] {
        "coordinate" => MtxFormat::Coordinate,
        "array" => return Err("Array format not supported for SparseMatrix load".to_string()),
        _ => return Err(format!("Unknown format: {}", parts[2])),
    };

    let _data_type = match parts[3] {
        "real" | "integer" | "complex" => MtxDataType::Real, // Treat as generic scalar T
        "pattern" => MtxDataType::Pattern,                   // T should be f64=1.0?
        _ => return Err(format!("Unknown data type: {}", parts[3])),
    };

    let symmetry = match parts[4] {
        "general" => MtxSymmetry::General,
        "symmetric" => MtxSymmetry::Symmetric,
        _ => MtxSymmetry::General, // Treat others as General for now or error?
    };

    // 2. Skip Comments and Parse Size
    let (rows, cols, nnz) = loop {
        let line_res = match lines.next() {
            Some(res) => res,
            None => return Err("Dimensions line not found or unexpected EOF".to_string()),
        };
        let line = line_res.map_err(|e| e.to_string())?;
        let trimmed = line.trim();

        if !trimmed.is_empty() && !trimmed.starts_with('%') {
            // This is the dimensions line
            let dims: Vec<&str> = trimmed.split_whitespace().collect();
            if dims.len() < 3 {
                return Err(format!("Invalid dimensions line: {}", trimmed));
            }
            let r = dims[0].parse::<usize>().map_err(|_| "Invalid rows")?;
            let c = dims[1].parse::<usize>().map_err(|_| "Invalid cols")?;
            let n = if dims.len() > 2 {
                dims[2].parse::<usize>().map_err(|_| "Invalid nnz")?
            } else {
                0
            };
            break (r, c, n);
        }
    };

    // 3. Parse Data (Coordinate format)
    let mut triplets = Vec::new();
    let mut count = 0;

    for line_res in lines {
        if count >= nnz {
            break;
        }

        let line = line_res.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('%') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }

        let r = parts[0].parse::<usize>().map_err(|_| "Invalid row index")?;
        let c = parts[1].parse::<usize>().map_err(|_| "Invalid col index")?;

        // Convert 1-based to 0-based
        let row_idx = r - 1;
        let col_idx = c - 1;

        let val = T::from_str(parts[2]).map_err(|_| "Invalid scalar value")?;

        triplets.push(crate::core::sparse::Triplet::new(row_idx, col_idx, val));

        if symmetry == MtxSymmetry::Symmetric && row_idx != col_idx {
            triplets.push(crate::core::sparse::Triplet::new(col_idx, row_idx, val));
        }

        count += 1;
    }

    let mut mat = SparseMatrix::<T>::new(rows, cols, crate::core::sparse::StorageOrder::ColMajor);
    mat.set_from_triplets(triplets);

    Ok(mat)
}

/// Save a SparseMatrix to a Matrix Market file.
pub fn save_matrix_market<P: AsRef<Path>, T: Scalar + std::fmt::Display>(
    path: P,
    mat: &SparseMatrix<T>,
) -> Result<(), String> {
    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(file);

    // Header
    // Assume Coordinate Real General
    // If we can detect pattern vs real, cool, but generic T usually means Real/Complex.
    writeln!(writer, "%%MatrixMarket matrix coordinate real general").map_err(|e| e.to_string())?;

    // Dimensions and NNZ count
    let mut triplet_count = 0;
    // Count exact non-zeros (SparseMatrix might calculate this differently if duplicates existed, but here we iterate)
    for k in 0..mat.outer_size() {
        let mut it = crate::core::sparse::iterators::InnerIterator::new(mat, k);
        while it.is_valid() {
            triplet_count += 1;
            it.next();
        }
    }

    writeln!(writer, "{} {} {}", mat.rows(), mat.cols(), triplet_count)
        .map_err(|e| e.to_string())?;

    // Data
    for k in 0..mat.outer_size() {
        let mut it = crate::core::sparse::iterators::InnerIterator::new(mat, k);
        while it.is_valid() {
            let (r, c) = if mat.order() == crate::core::sparse::StorageOrder::ColMajor {
                (it.row(), k)
            } else {
                (k, it.col())
            };
            let val = it.value();
            writeln!(writer, "{} {} {}", r + 1, c + 1, val).map_err(|e| e.to_string())?;
            it.next();
        }
    }

    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}
