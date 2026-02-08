use std::fs;
use std::path::Path;
use std::process::Command;

/// Compiles and runs a C++ harness, returning the parsed output as (row, col, value).
#[allow(dead_code)]
pub fn run_cpp_harness(cpp_file: &str) -> Result<Vec<(usize, usize, f32)>, String> {
    let cpp_path = Path::new(cpp_file);
    let bin_path = cpp_path.with_extension("bin");

    // 1. Compile C++ code with Eigen include path
    let status = Command::new("g++")
        .arg("-I")
        .arg("eigen-src") // Relative to project root
        .arg("-O3")
        .arg("-march=native") // Enable host-specific optimizations (AVX, FMA, etc.) to match Rust
        .arg(cpp_file)
        .arg("-o")
        .arg(&bin_path)
        .output()
        .map_err(|e| format!("Failed to run g++: {}", e))?;

    if !status.status.success() {
        let stderr = String::from_utf8_lossy(&status.stderr);
        return Err(format!("C++ compilation failed:\n{}", stderr));
    }

    // 2. Run the binary
    let output = Command::new(&bin_path)
        .output()
        .map_err(|e| format!("Failed to run C++ binary: {}", e))?;

    if !output.status.success() {
        return Err("C++ binary execution failed".to_string());
    }

    // Clean up binary
    let _ = fs::remove_file(&bin_path);

    // 3. Parse output: row,col,value
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut results = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 3 {
            let r: usize = parts[0]
                .parse()
                .map_err(|e| format!("Parse error: {}", e))?;
            let c: usize = parts[1]
                .parse()
                .map_err(|e| format!("Parse error: {}", e))?;
            let v: f32 = parts[2]
                .parse()
                .map_err(|e| format!("Parse error: {}", e))?;
            results.push((r, c, v));
        }
    }

    Ok(results)
}

/// Runs a C++ harness and returns the raw stdout.
#[allow(dead_code)]
pub fn run_cpp_harness_stdout(cpp_file: &str) -> Result<String, String> {
    let cpp_path = Path::new(cpp_file);
    // Use a unique name to avoid "Text file busy" in parallel tests
    let thread_id = std::thread::current().id();
    let bin_name = format!(
        "{}_{:?}.bin",
        cpp_path.file_stem().unwrap().to_str().unwrap(),
        thread_id
    );
    let bin_path = cpp_path.parent().unwrap().join(bin_name);

    let status = Command::new("g++")
        .arg("-I")
        .arg("eigen-src")
        .arg("-O3")
        .arg("-march=native")
        .arg(cpp_file)
        .arg("-o")
        .arg(&bin_path)
        .output()
        .map_err(|e| format!("Failed to run g++: {}", e))?;

    if !status.status.success() {
        let stderr = String::from_utf8_lossy(&status.stderr);
        return Err(format!("C++ compilation failed:\n{}", stderr));
    }

    let output = Command::new(&bin_path)
        .output()
        .map_err(|e| format!("Failed to run C++ binary: {}", e))?;

    let _ = fs::remove_file(&bin_path);

    if !output.status.success() {
        return Err("C++ binary execution failed".to_string());
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}
