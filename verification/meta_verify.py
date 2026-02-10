
import os
import sys
import shutil
import json
from extract_cpp_api import extract_api as extract_cpp
from extract_rust_api import extract_api as extract_rust

TEST_DIR = 'tmp/verification/test_data'

def setup_test_env():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

    # 1. Create Dummy C++ Header
    cpp_content = """
    namespace Eigen {
    class DummyMatrix {
      public:
        EIGEN_DEVICE_FUNC void testMethod();
        EIGEN_DEVICE_FUNC int operator+(int);
        
        // Should be ignored
        void privateMethod();
      protected:
        EIGEN_DEVICE_FUNC void protectedMethod();
    };
    
    // Test Multi-line
    class ComplexMatrix {
      public:
        EIGEN_DEVICE_FUNC
        void complexMethod(int a,
                           int b);
    };
    }
    """
    with open(os.path.join(TEST_DIR, 'Dummy.h'), 'w') as f:
        f.write(cpp_content)

    # 2. Create Dummy Rust Source
    rust_content = """
    pub struct DummyMatrix;
    
    impl DummyMatrix {
        pub fn test_method(&self) {}
        
        // Operator mapping target
        pub fn add(&self, other: i32) {}
        
        fn private_method(&self) {}
    }
    
    pub struct ComplexMatrix;
    
    impl ComplexMatrix {
        pub fn complex_method(&self) {}
    }
    """
    with open(os.path.join(TEST_DIR, 'dummy.rs'), 'w') as f:
        f.write(rust_content)

def run_meta_test():
    setup_test_env()
    
    print("Running C++ Extractor on Dummy.h...")
    cpp_api = extract_cpp(TEST_DIR)
    
    print("Running Rust Extractor on dummy.rs...")
    rust_api = extract_rust(TEST_DIR)
    
    # Assertions
    errors = []
    
    # Check C++ Extraction
    if 'DummyMatrix' not in cpp_api:
        errors.append("C++: DummyMatrix class not found")
    else:
        methods = cpp_api['DummyMatrix']
        if 'testMethod' not in methods: errors.append("C++: testMethod not extracted")
        if 'operator+' not in methods: errors.append("C++: operator+ not extracted")
        if 'privateMethod' in methods: errors.append("C++: privateMethod wrongly extracted")
        
    if 'ComplexMatrix' not in cpp_api:
        errors.append("C++: ComplexMatrix class not found")
    elif 'complexMethod' not in cpp_api['ComplexMatrix']:
        methods = cpp_api['ComplexMatrix']
        errors.append(f"C++: complexMethod (multi-line) not extracted. Found: {methods}")

    # Check Rust Extraction
    if 'DummyMatrix' not in rust_api:
        errors.append("Rust: DummyMatrix struct not found")
    else:
        methods = rust_api['DummyMatrix']
        if 'test_method' not in methods: errors.append("Rust: test_method not extracted")
        if 'add' not in methods: errors.append("Rust: add not extracted")
        if 'private_method' in methods: errors.append("Rust: private_method wrongly extracted")

    if 'ComplexMatrix' not in rust_api:
        errors.append("Rust: ComplexMatrix struct not found")

    if errors:
        print("❌ Meta-Verification FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("✅ Meta-Verification PASSED: Tools are accurately extracting APIs.")
        
if __name__ == '__main__':
    run_meta_test()
