use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/core/cuda/kernels.cu");

    // Detect CUDA installation
    let cuda_path = std::env::var("CUDA_PATH").ok().or_else(|| {
        if Path::new("/usr/local/cuda").exists() {
            Some("/usr/local/cuda".to_string())
        } else {
            // Try to find nvcc and guess path
            Command::new("which")
                .arg("nvcc")
                .output()
                .ok()
                .and_then(|out| {
                    if out.status.success() {
                        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                        Path::new(&s)
                            .parent()
                            .and_then(|p| p.parent())
                            .map(|p| p.to_string_lossy().to_string())
                    } else {
                        None
                    }
                })
        }
    });



    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = Path::new(&out_dir).join("kernels.ptx");

    // Only attempt to detect CUDA if the feature is requested
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        if let Some(path) = cuda_path {
            println!("cargo:rustc-cfg=feature=\"cuda_enabled\"");
            println!("cargo:rustc-link-search=native={}/lib64", path);
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu"); // Standard Ubuntu path
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-link-lib=cudart");
            println!("cargo:rustc-link-lib=cusparse");
    
            // Compile .cu kernels if they exist
            let kernel_src = "src/core/cuda/kernels.cu";
            if Path::new(kernel_src).exists() {
                let status = Command::new("nvcc")
                    .args(["-ptx", "-O3", kernel_src, "-o"])
                    .arg(&ptx_path)
                    .status();
    
                if let Ok(s) = status {
                    if s.success() {
                        println!("cargo:rustc-cfg=feature=\"cuda_kernels_compiled\"");
                    } else {
                        // Fallback to empty PTX if compilation fails
                        std::fs::write(&ptx_path, "// Empty PTX (compilation failed)\n").unwrap();
                    }
                } else {
                    std::fs::write(&ptx_path, "// Empty PTX (nvcc not found)\n").unwrap();
                }
            } else {
                std::fs::write(&ptx_path, "// Empty PTX (source not found)\n").unwrap();
            }
        } else {
            println!("cargo:warning=CUDA Toolkit not found, but 'cuda' feature was requested. CUDA features will be disabled.");
            std::fs::write(&ptx_path, "// Empty PTX (CUDA not found)\n").unwrap();
        }
    } else {
        // Feature not requested, just write empty PTX so include_str! works
        std::fs::write(&ptx_path, "// Empty PTX (CUDA feature disabled)\n").unwrap();
    }
}
