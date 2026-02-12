
import subprocess
import os
import sys
import datetime

CPP_BENCH_SRC = "benches/cpp_ref/eigen_bench.cpp"
CPP_BENCH_BIN = "benches/cpp_ref/eigen_bench.bin"
RUST_BENCH_CMD = ["cargo", "run", "--release", "--example", "perf_runner"]
import platform

REPORT_DIR = "docs/reports"
LATEST_REPORT_LINK = "docs/reports/LATEST_PERFORMANCE.md"

# Threshold: Rust/C++ ratio. 1.0 is parity. 1.5 is acceptable. 2.0 is warning.
MAX_RATIO = 2.0 

def get_system_info():
    info = {}
    try:
        # CPU Info (Linux specific)
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["CPU"] = line.split(":")[1].strip()
                    break
    except:
        info["CPU"] = platform.processor()
    
    info["OS"] = f"{platform.system()} {platform.release()}"
    
    try:
        res = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
        info["Rust"] = res.stdout.strip()
    except:
        info["Rust"] = "Unknown"

    try:
        res = subprocess.run(["g++", "--version"], capture_output=True, text=True)
        info["C++"] = res.stdout.splitlines()[0] if res.stdout else "Unknown"
    except:
        info["C++"] = "Unknown"
        
    return info

def compile_cpp():
    print(f"Compiling C++ Benchmark: {CPP_BENCH_SRC}...")
    cmd = ["g++", "-O3", "-march=native", "-I", "eigen-src", CPP_BENCH_SRC, "-o", CPP_BENCH_BIN]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("❌ C++ Compilation Failed:")
        print(res.stderr)
        return False
    return True

def run_cpp():
    print("Running C++ Benchmark...")
    res = subprocess.run([f"./{CPP_BENCH_BIN}"], capture_output=True, text=True)
    if res.returncode != 0:
        print("❌ C++ Benchmark Failed:")
        print(res.stderr)
        return None
    return parse_output(res.stdout)

def run_rust():
    print("Running Rust Benchmark (Release Mode)...")
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"
    res = subprocess.run(RUST_BENCH_CMD, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print("❌ Rust Benchmark Failed:")
        print(res.stderr)
        return None
    # Cargo might print compilation artifacts to stderr/stdout
    # We look for lines starting with Operation name
    return parse_output(res.stdout)

def parse_output(output):
    """
    Parses "Operation,Size,TimeNs"
    Returns dict: {(Op, Size): TimeNs}
    """
    data = {}
    for line in output.splitlines():
        parts = line.split(',')
        if len(parts) == 3:
            try:
                op = parts[0].strip()
                size = int(parts[1].strip())
                time = int(parts[2].strip())
                data[(op, size)] = time
            except:
                pass
    return data

def generate_report(cpp_data, rust_data):
    sys_info = get_system_info()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    report_filename = f"performance_{timestamp}.md"
    
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        
    report_path = os.path.join(REPORT_DIR, report_filename)
    
    lines = []
    lines.append(f"# Performance Verification Report: {timestamp}")
    lines.append(f"> **Generated**: {datetime.datetime.now().isoformat()}")
    lines.append(f"> **Threshold**: < {MAX_RATIO}x of C++ Eigen\n")
    
    lines.append("## System Context")
    lines.append(f"- **CPU**: {sys_info.get('CPU', 'Unknown')}")
    lines.append(f"- **OS**: {sys_info.get('OS', 'Unknown')}")
    lines.append(f"- **Rust**: {sys_info.get('Rust', 'Unknown')}")
    lines.append(f"- **C++**: {sys_info.get('C++', 'Unknown')}\n")
    
    lines.append("## Benchmark Results")
    lines.append("| Operation | Size | C++ (ns) | Rust (ns) | Ratio | Status |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    all_passed = True
    
    # Iterate over sorted keys from C++ data (as source of truth for scope)
    keys = sorted(cpp_data.keys())
    
    for key in keys:
        op, size = key
        cpp_time = cpp_data[key]
        if key in rust_data:
            rust_time = rust_data[key]
            ratio = rust_time / cpp_time if cpp_time > 0 else 0.0
            
            status = "✅ PASS"
            if ratio > MAX_RATIO:
                status = "❌ FAIL (Slow)"
                all_passed = False
            elif ratio < 1.0:
                status = "🚀 FASTER"
                
            lines.append(f"| {op} | {size} | {cpp_time} | {rust_time} | **{ratio:.2f}x** | {status} |")
        else:
            lines.append(f"| {op} | {size} | {cpp_time} | - | - | ⚠️ MISSING |")
            
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    
    # Also update LATEST link/copy
    with open(LATEST_REPORT_LINK, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Report generated: {report_path}")
    print(f"Latest report updated: {LATEST_REPORT_LINK}")
    return all_passed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-benchmark", action="store_true", help="Append results to BENCHMARK.md")
    args = parser.parse_args()

    if not compile_cpp():
        sys.exit(1)
        
    cpp_data = run_cpp()
    if cpp_data is None: sys.exit(1)
        
    rust_data = run_rust()
    if rust_data is None: sys.exit(1)
    
    success = generate_report(cpp_data, rust_data)
    
    if args.update_benchmark:
        update_benchmark_log(cpp_data, rust_data)

    if success:
        print("✅ Performance Verification PASSED")
        sys.exit(0)
    else:
        print("❌ Performance Verification FAILED (Some deviations too high)")
        sys.exit(1)

def update_benchmark_log(cpp_data, rust_data):
    log_file = "BENCHMARK_LOG.md"
    if not os.path.exists(log_file):
        print(f"⚠️ {log_file} not found. Skipping persistent log update.")
        return

    import datetime
    today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Select key metrics for brevity
    key_metrics = [
        ("MatMul", 256),
        ("LLT", 256),
    ]
    
    lines = []
    lines.append(f"\n### {today}: Automated Run")
    lines.append("| Operation | Size | Ratio | Status |")
    lines.append("| :--- | :--- | :--- | :--- |")
    
    for op, size in key_metrics:
        key = (op, size)
        if key in cpp_data and key in rust_data:
            c = cpp_data[key]
            r = rust_data[key]
            ratio = r / c if c > 0 else 0.0
            status = "✅" if ratio < MAX_RATIO else "❌"
            lines.append(f"| {op} | {size} | **{ratio:.2f}x** | {status} |")
            
    with open(log_file, "a") as f:
        f.write("\n".join(lines))
    print(f"📝 Appended results to {log_file}")

if __name__ == "__main__":
    main()
