//! Advanced Neural Network Primitives.
//!
//! Provides Conv2D, MaxPool, and BatchNorm operations for 4D Tensors (NCHW Format).

use crate::core::scalar::Scalar;
use crate::core::tensor::{Tensor, device::Device};

/// Performs a 2D Convolution on a 4D tensor.
/// Assumes NCHW format:
/// - input: `[batch, in_channels, in_height, in_width]`
/// - weight: `[out_channels, in_channels, kernel_height, kernel_width]`
/// - bias: `[out_channels]`
pub fn conv2d<T: Scalar, D: Device + Clone>(
    input: &Tensor<T, 4, D>,
    weight: &Tensor<T, 4, D>,
    bias: Option<&Tensor<T, 1, D>>,
    stride: usize,
    padding: usize,
) -> Result<Tensor<T, 4, D>, String> {
    let [batch, in_c, in_h, in_w] = input.dims();
    let [out_c, w_in_c, k_h, k_w] = weight.dims();

    if in_c != w_in_c {
        return Err("Input channels do not match weight in_channels".to_string());
    }

    let out_h = (in_h + 2 * padding - k_h) / stride + 1;
    let out_w = (in_w + 2 * padding - k_w) / stride + 1;

    let device = input.device().clone();
    let mut out = Tensor::<T, 4, D>::new_with_device([batch, out_c, out_h, out_w], device)?;

    // Fallback: Naive CPU implementation
    // A robust version would use im2col + gemm, potentially offloading to GPU `matmul`.
    if let (Some(in_data), Some(w_data), Some(out_data)) = (input.data(), weight.data(), out.data_mut()) {
        let b_data = bias.and_then(|b| b.data());

        for b in 0..batch {
            for oc in 0..out_c {
                let bias_val = b_data.map_or(T::default(), |b_slice| b_slice[oc]);
                
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = bias_val;
                        
                        for ic in 0..in_c {
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih_pos = (oh * stride + kh) as isize - padding as isize;
                                    let iw_pos = (ow * stride + kw) as isize - padding as isize;
                                    
                                    if ih_pos >= 0 && ih_pos < in_h as isize && iw_pos >= 0 && iw_pos < in_w as isize {
                                        let in_idx = b * (in_c * in_h * in_w) + ic * (in_h * in_w) + (ih_pos as usize * in_w) + iw_pos as usize;
                                        let w_idx = oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + kh * k_w + kw;
                                        
                                        sum += in_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }
                        
                        let out_idx = b * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
                        out_data[out_idx] = sum;
                    }
                }
            }
        }
        return Ok(out);
    }
    
    Err("GPU Conv2D not yet natively supported without cuDNN or manual im2col+cublas".to_string())
}

/// Computes 2D Max Pooling on a 4D Tensor (NCHW).
pub fn max_pool2d<T: Scalar + PartialOrd, D: Device + Clone>(
    input: &Tensor<T, 4, D>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<Tensor<T, 4, D>, String> {
    let [batch, c, in_h, in_w] = input.dims();
    
    let out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    let device = input.device().clone();
    let mut out = Tensor::<T, 4, D>::new_with_device([batch, c, out_h, out_w], device)?;

    if let (Some(in_data), Some(out_data)) = (input.data(), out.data_mut()) {
        for b in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = None;
                        
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih_pos = (oh * stride + kh) as isize - padding as isize;
                                let iw_pos = (ow * stride + kw) as isize - padding as isize;
                                
                                if ih_pos >= 0 && ih_pos < in_h as isize && iw_pos >= 0 && iw_pos < in_w as isize {
                                    let in_idx = b * (c * in_h * in_w) + ch * (in_h * in_w) + (ih_pos as usize * in_w) + iw_pos as usize;
                                    let val = in_data[in_idx];
                                    
                                    if let Some(mv) = max_val {
                                        if val > mv {
                                            max_val = Some(val);
                                        }
                                    } else {
                                        max_val = Some(val);
                                    }
                                }
                            }
                        }
                        
                        let out_idx = b * (c * out_h * out_w) + ch * (out_h * out_w) + oh * out_w + ow;
                        out_data[out_idx] = max_val.unwrap_or_else(|| T::default());
                    }
                }
            }
        }
        return Ok(out);
    }

    Err("GPU MaxPool2D not yet implemented".to_string())
}

/// Applies Batch Normalization across the batch dimension.
pub fn batch_norm<T: Scalar, D: Device + Clone>(
    input: &Tensor<T, 4, D>,
    running_mean: &Tensor<T, 1, D>,
    running_var: &Tensor<T, 1, D>,
    weight: Option<&Tensor<T, 1, D>>,
    bias: Option<&Tensor<T, 1, D>>,
    epsilon: f64,
) -> Result<Tensor<T, 4, D>, String> {
    let [batch, c, h, w] = input.dims();

    let device = input.device().clone();
    let mut out = Tensor::<T, 4, D>::new_with_device([batch, c, h, w], device)?;
    
    let eps_t = T::from_f64(epsilon);

    if let (Some(in_data), Some(mean_data), Some(var_data), Some(out_data)) = 
        (input.data(), running_mean.data(), running_var.data(), out.data_mut()) 
    {
        let w_data = weight.and_then(|w| w.data());
        let b_data = bias.and_then(|b| b.data());

        for b_idx in 0..batch {
            for ch in 0..c {
                let mean = mean_data[ch];
                let var = var_data[ch];
                let inv_std = T::from_f64(1.0) / (var + eps_t).sqrt();
                
                let gamma = w_data.map_or(T::from_f64(1.0), |ws| ws[ch]);
                let beta = b_data.map_or(T::default(), |bs| bs[ch]);

                for oh in 0..h {
                    for ow in 0..w {
                        let idx = b_idx * (c * h * w) + ch * (h * w) + oh * w + ow;
                        let val = in_data[idx];
                        
                        let normalized = (val - mean) * inv_std;
                        out_data[idx] = normalized * gamma + beta;
                    }
                }
            }
        }
        return Ok(out);
    }

    Err("GPU BatchNorm not yet implemented".to_string())
}
