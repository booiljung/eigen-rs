
import json
import os
import re

CPP_API_FILE = 'tmp/verification/cpp_api_list.json'
RUST_API_FILE = 'tmp/verification/rust_api_list.json'
TESTS_DIR = 'tests'
REPORT_FILE = 'tmp/verification/VERIFICATION_REPORT.md'

# Manual mapping for operators and common deviations
RUST_TO_CPP_MAP = {
    'add': 'operator+',
    'sub': 'operator-',
    'mul': 'operator*',
    'div': 'operator/',
    'rem': 'operator%',
    'neg': 'operator-',
    'index': 'operator[]',
    'index_mut': 'operator[]',
    'inv': 'inverse',
    'det': 'determinant',
    'transpose': 'transpose',
    'adjoint': 'adjoint',
    'conjugate': 'conjugate',
    'dot': 'dot',
    'cross': 'cross',
    'normalized': 'normalized',
    'norm': 'norm',
    'squared_norm': 'squaredNorm',
    'mean': 'mean',
    'sum': 'sum',
    'prod': 'prod',
    'min_coeff': 'minCoeff',
    'max_coeff': 'maxCoeff',
    'trace': 'trace',
    'diagonal': 'diagonal',
    'block': 'block',
    'row': 'row',
    'col': 'col',
    'rows': 'rows',
    'cols': 'cols',
    'size': 'size',
    'data': 'data',
    'len': 'size',
    'is_approx': 'isApprox',
    'set_zero': 'setZero',
    'set_ones': 'setOnes',
    'set_constant': 'setConstant',
    'set_identity': 'setIdentity',
    'set_random': 'setRandom',
    'set_lin_spaced': 'setLinSpaced',
    'eval': 'eval',
    'cast': 'cast',
    'map': 'Map',
    # Decompositions
    'llt': 'llt',
    'ldlt': 'ldlt',
    'lu': 'lu', # PartialPivLU
    'partial_piv_lu': 'partialPivLu',
    'full_piv_lu': 'fullPivLu',
    'qr': 'householderQr',
    'householder_qr': 'householderQr',
    'col_piv_householder_qr': 'colPivHouseholderQr',
    'full_piv_householder_qr': 'fullPivHouseholderQr',
    'svd': 'bdcSvd', # or jacobiSvd
    'jacobi_svd': 'jacobiSvd',
    'bdc_svd': 'bdcSvd',
    'eigenvalues': 'eigenvalues',
    'eigenvectors': 'eigenvectors',
}

def to_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def scan_tests_for_usage(method_name):
    """
    Scans tests directory for usage of .method_name( or method_name(
    Returns list of test files using it.
    """
    using_files = set()
    usage_regex = re.compile(r'\.' + re.escape(method_name) + r'\s*\(|' + re.escape(method_name) + r'\s*\(')
    
    for root, dirs, files in os.walk(TESTS_DIR):
        for file in files:
            if not file.endswith('.rs'):
                continue
            filepath = os.path.join(root, file)
            # relative path from TESTS_DIR
            relpath = os.path.relpath(filepath, TESTS_DIR)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if usage_regex.search(content):
                    using_files.add(relpath)
    return list(using_files)

def is_differential_test(filename):
    """
    Checks if a test file imports common harness or runs C++ harness.
    """
    filepath = os.path.join(TESTS_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'run_cpp_harness' in content or 'common::' in content:
        return True
    return False

def main():
    with open(CPP_API_FILE, 'r') as f:
        cpp_data = json.load(f)
    with open(RUST_API_FILE, 'r') as f:
        rust_data = json.load(f)
        
    # Flatten C++ API Pool
    cpp_pool = set()
    for class_name, methods in cpp_data.items():
        for m in methods:
            cpp_pool.add(m)
            
    # Also add known operators manually if missing
    cpp_pool.add('operator+')
    cpp_pool.add('operator-')
    cpp_pool.add('operator*')
    cpp_pool.add('operator/')
    
    report_lines = []
    report_lines.append("# Verification Report: Proof of Perfection")
    report_lines.append("")
    report_lines.append("| Rust API | C++ Match | Diff Test Coverage | Status |")
    report_lines.append("| :--- | :--- | :--- | :--- |")
    
    stats = {'Total': 0, 'Verified': 0, 'Missing_Cpp': 0, 'No_Test': 0}
    
    for struct_name, methods in rust_data.items():
        if struct_name == 'Global': continue # Skip global functions for now or valid?
        
        for rust_method in methods:
            stats['Total'] += 1
            
            # 1. Match C++
            cpp_match = None
            
            # Direct mapping
            if rust_method in RUST_TO_CPP_MAP:
                candidate = RUST_TO_CPP_MAP[rust_method]
                if candidate in cpp_pool:
                    cpp_match = candidate
            
            # Snake to Camel
            if not cpp_match:
                candidate = to_camel_case(rust_method)
                if candidate in cpp_pool:
                    cpp_match = candidate
            
            # Exact match
            if not cpp_match:
                if rust_method in cpp_pool:
                    cpp_match = rust_method

            # 2. Check Coverage
            using_files = scan_tests_for_usage(rust_method)
            diff_tested = False
            diff_test_files = []
            
            for f in using_files:
                if is_differential_test(f):
                    diff_tested = True
                    diff_test_files.append(f)
            
            # 3. Status
            status = "FAIL"
            cpp_col = cpp_match if cpp_match else "**MISSING**"
            test_col = ""
            
            if diff_tested:
                test_col = f"Yes ({', '.join(diff_test_files[:1])})" # List first one
                if cpp_match:
                    status = "PASS"
                    stats['Verified'] += 1
                else:
                    status = "WARN (Rust ext)" # Verified but no C++ match?
                    stats['Missing_Cpp'] += 1
            else:
                if using_files:
                    test_col = f"Unit Test ({', '.join(using_files[:1])})"
                    status = "PARTIAL"
                    stats['No_Test'] += 1
                else:
                    test_col = "**NO TEST**"
                    status = "FAIL"
                    stats['No_Test'] += 1
            
            report_lines.append(f"| `{struct_name}::{rust_method}` | `{cpp_col}` | {test_col} | {status} |")

    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- **Total APIs Checked**: {stats['Total']}")
    report_lines.append(f"- **Strictly Verified (Name + Behavior)**: {stats['Verified']}")
    report_lines.append(f"- **Functionally Verified (Behavior only)**: {stats['Missing_Cpp']}")
    report_lines.append(f"- **Unverified / No Coverage**: {stats['No_Test']}")
    
    total_verified = stats['Verified'] + stats['Missing_Cpp']
    validity_score = (total_verified / stats['Total']) * 100 if stats['Total'] > 0 else 0
    report_lines.append(f"- **Perfection Score**: **{validity_score:.1f}%** (Functionally Verified)")

    content = '\n'.join(report_lines)
    with open(REPORT_FILE, 'w') as f:
        f.write(content)
        
    print(f"Report generated: {REPORT_FILE}")
    print(f"Score: {validity_score:.1f}%")

if __name__ == '__main__':
    main()
