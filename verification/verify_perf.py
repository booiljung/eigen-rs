
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
    if "DEBUG" in res.stderr:
        print("--- STDERR DEBUG ---")
        print(res.stderr)
        print("--------------------")
    
    if res.returncode != 0:
        print("❌ Rust Benchmark Failed:")
        print(res.stderr)
        return None
    # Cargo might print compilation artifacts to stderr/stdout
    # We look for lines starting with Operation name
    return parse_output(res.stdout)

def parse_output(output):
    """
    Parses "Operation,Size,TimeNs,Checksum"
    Returns dict: {(Op, Size): (TimeNs, Checksum)}
    """
    data = {}
    for line in output.splitlines():
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                op = parts[0].strip()
                size = int(parts[1].strip())
                time = int(parts[2].strip())
                checksum = float(parts[3].strip()) if len(parts) > 3 else None
                data[(op, size)] = (time, checksum)
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
    
    # 1. Analyze Data & Statistics
    stats = {} # Op -> {ratios: [], fail_count: 0}
    
    keys = sorted(cpp_data.keys())
    table_lines = []
    table_lines.append("| Operation | Size | C++ (ns) | Rust (ns) | Ratio | Status |")
    table_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    all_passed = True
    
    for key in keys:
        op, size = key
        cpp_val = cpp_data[key]
        cpp_time = cpp_val[0]
        cpp_check = cpp_val[1]
        
        if key in rust_data:
            rust_val = rust_data[key]
            rust_time = rust_val[0]
            rust_check = rust_val[1]
            
            ratio = rust_time / cpp_time if cpp_time > 0 else 0.0
            
            # Record Stat
            if op not in stats: stats[op] = {'ratios': [], 'fail_count': 0}
            stats[op]['ratios'].append(ratio)
            
            # Verify Checksum
            check_pass = True
            epsilon = 0.1 
            # If checksum is very large, absolute epsilon is too strict. Use relative.
            if cpp_check is not None and rust_check is not None:
                 diff = abs(cpp_check - rust_check)
                 # 1% relative error tolerance for very large numbers, or 0.1 absolute
                 if diff > epsilon and diff > abs(cpp_check)*0.01:
                     check_pass = False
            
            status = ""
            if not check_pass:
                status = f"❌ CHECKSUM ({cpp_check:.2f} vs {rust_check:.2f})"
                all_passed = False
                stats[op]['fail_count'] += 1
            elif ratio > MAX_RATIO:
                status = "❌ FAIL (Slow)"
                all_passed = False
                stats[op]['fail_count'] += 1
            elif ratio < 1.0:
                status = "🚀 FASTER"
            else:
                status = "✅ PASS"
                
            table_lines.append(f"| {op} | {size} | {cpp_time} | {rust_time} | **{ratio:.2f}x** | {status} |")
        else:
            table_lines.append(f"| {op} | {size} | {cpp_time} | - | - | ⚠️ MISSING |")

    # 2. Print Summary
    lines.append("## Summary Statistics")
    lines.append("| Operation | Count | Min Ratio | Max Ratio | Avg Ratio | Fails |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for op in sorted(stats.keys()):
        s = stats[op]
        ratios = s['ratios']
        if ratios:
            min_r = min(ratios)
            max_r = max(ratios)
            avg_r = sum(ratios) / len(ratios)
            count = len(ratios)
            fail = s['fail_count']
            fail_mark = "❌" if fail > 0 else "✅"
            lines.append(f"| {op} | {count} | {min_r:.2f}x | **{max_r:.2f}x** | {avg_r:.2f}x | {fail_mark} {fail} |")
        else:
            lines.append(f"| {op} | 0 | - | - | - | - |")
    lines.append("\n")

    lines.append("## Detailed Results")
    lines.extend(table_lines)
            
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    
    with open(LATEST_REPORT_LINK, 'w') as f:
        f.write("\n".join(lines))
        
    print(f"Report generated: {report_path}")
    print(f"Latest report updated: {LATEST_REPORT_LINK}")
    return all_passed

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--update-benchmark", action="store_true", help="Append results to BENCHMARK.md")
    args = parser.parse_args()

    if not compile_cpp():
        sys.exit(1)
        
    cpp_data = run_cpp()
    if cpp_data is None: sys.exit(1)
        
    rust_data = run_rust()
    if rust_data is None: sys.exit(1)
    
    success = generate_report(cpp_data, rust_data)
    
    if success:
        print("✅ Performance Verification PASSED")
        sys.exit(0)
    else:
        print("❌ Performance Verification FAILED (Some deviations too high)")
        sys.exit(1)

if __name__ == "__main__":
    main()
