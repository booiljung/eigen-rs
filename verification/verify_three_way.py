
import json
import os
import re
import sys

# Import shared logic/data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from verify_parity import TEST_MAPPING
from verify_coverage import RUST_EXTENSIONS

PARITY_SUITE_FILES = {t[0] + ".rs" for t in TEST_MAPPING}
RUST_API_FILE = 'tmp/verification/rust_api_list.json'
CPP_API_FILE = 'tmp/verification/cpp_api_list.json'
TEST_DIR = 'tests'

def get_test_usage(method_name):
    """
    Returns the set of test files that use the given method.
    (Simplified grep logic similar to verify_coverage.py)
    """
    cmd = f"grep -l '\\.{method_name}(' {TEST_DIR}/*.rs"
    try:
        output = os.popen(cmd).read().strip()
        if not output: return set()
        return {os.path.basename(p) for p in output.split('\n')}
    except:
        return set()

def verify_consistency():
    print("running Three-Way Consistency Check...")
    print(f"Parity Suite contains {len(PARITY_SUITE_FILES)} test files.")
    
    with open(RUST_API_FILE, 'r') as f:
        rust_api = json.load(f)
        
    with open(CPP_API_FILE, 'r') as f:
        cpp_api = json.load(f)

    # Flatten C++ API
    cpp_methods = set()
    for methods in cpp_api.values():
        cpp_methods.update(methods)

    total_apis = 0
    consistent_count = 0
    ghost_coverage_count = 0
    
    report_lines = []
    report_lines.append("# Three-Way Consistency Report")
    report_lines.append("Checking: **C++ Spec** <-> **Rust Impl** <-> **Parity Execution**\n")
    report_lines.append("| API | Used In (Static) | Included in Parity Suite? | Status |")
    report_lines.append("| :--- | :--- | :--- | :--- |")

    for struct_name, methods in rust_api.items():
        if struct_name == 'Global': continue
        
        for method in methods:
            total_apis += 1
            
            # 1. Existence Check (Rust vs C++)
            # For this check, we assume verify_coverage handles the naming mapping.
            # We focus on the "Rust Extracted" vs "Parity Verified" link.
            
            # 2. Usage Check (Static Analysis)
            used_files = get_test_usage(method)
            
            # 3. Intersection Check (Static vs Dynamic)
            # EXEMPTION: If it is a Rust Extension, it doesn't need Parity (No C++ Oracle)
            if method in RUST_EXTENSIONS:
                continue # Skip reporting, inherently consistent for our definition (or track separately)

            # Does any of the used_files exist in PARITY_SUITE_FILES?
            parity_covered = used_files.intersection(PARITY_SUITE_FILES)
            
            usage_str = ", ".join(list(used_files)[:1]) if used_files else "None"
            
            if parity_covered:
                status = "✅ CONSISTENT"
                consistent_count += 1
            elif used_files:
                status = "⚠️ GHOST (Tested but not in Parity Suite)"
                ghost_coverage_count += 1
                report_lines.append(f"| `{struct_name}::{method}` | `{usage_str}` | ❌ NO | {status} |")
            else:
                status = "❌ UNUSED"
                # report_lines.append(f"| `{struct_name}::{method}` | None | ❌ NO | {status} |")

    print(f"Total APIs Checked: {total_apis}")
    print(f"Consistent (Verified & In Suite): {consistent_count}")
    print(f"Ghost (Verified but NOT in Suite): {ghost_coverage_count}")
    
    report_path = 'tmp/verification/CONSISTENCY_REPORT.md'
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
        f.write(f"\n\n**Consistency Score**: {consistent_count}/{total_apis} APIs are fully parity-verified.")
    
    print(f"Report generated: {report_path}")

if __name__ == '__main__':
    verify_consistency()
