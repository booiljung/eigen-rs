extern "C" __global__ void add_kernel_f32(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void add_kernel_f64(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void sub_kernel_f32(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

extern "C" __global__ void sub_kernel_f64(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

extern "C" __global__ void mul_kernel_f32(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__ void mul_kernel_f64(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__ void scalar_mul_kernel_f32(const float* a, float s, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * s;
    }
}

extern "C" __global__ void scalar_mul_kernel_f64(const double* a, double s, double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * s;
    }
}

extern "C" __global__ void assign_kernel_f32(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

extern "C" __global__ void assign_kernel_f64(double* out, const double* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

// Naive GEMM for verification: C = A * B
// A: M x K, B: K x N, C: M x N
// ColMajor assumptions:
// A(r, k) = A[k*M + r]
// B(k, c) = B[c*K + k]
// C(r, c) = C[c*M + r]
extern "C" __global__ void matmul_kernel_f32(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[col * K + k];
        }
        C[col * M + row] = sum;
    }
}

extern "C" __global__ void matmul_kernel_f64(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[col * K + k];
        }
        C[col * M + row] = sum;
    }
}

extern "C" __global__ void permute_kernel_f32(
    float* out, 
    const float* inp, 
    int size, 
    int rank, 
    const int* out_dims, 
    const int* in_strides, 
    const int* perm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Decompose Output Linear Index 'idx' into Output Multi-Index
    // And simultaneously compute Input Linear Index
    
    int temp = idx; // Remaining index to decompose
    int in_idx = 0;
    
    // Column-Major decompostion
    // To match Rust's:
    // for d in 0..RANK: indices[d] = temp % dims[d]; temp /= dims[d];
    
    for (int d = 0; d < rank; ++d) {
        int dim_val = out_dims[d];
        int dim_idx = temp % dim_val;
        temp /= dim_val;
        
        // This dimension 'd' in Output corresponds to Input dimension 'perm[d]'
        // Contribution to Input Linear Index = index * stride
        in_idx += dim_idx * in_strides[perm[d]];
    }
    
    out[idx] = inp[in_idx];
}

extern "C" __global__ void permute_kernel_f64(
    double* out, 
    const double* inp, 
    int size, 
    int rank, 
    const int* out_dims, 
    const int* in_strides, 
    const int* perm
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int temp = idx;
    int in_idx = 0;
    
    for (int d = 0; d < rank; ++d) {
        int dim_val = out_dims[d];
        int dim_idx = temp % dim_val;
        temp /= dim_val;
        
        in_idx += dim_idx * in_strides[perm[d]];
    }
    
    out[idx] = inp[in_idx];
}
