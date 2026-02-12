
import os
import re
import json

EIGEN_SRC_DIR = 'eigen-src/Eigen/src'
OUTPUT_FILE = 'tmp/verification/cpp_api_list.json'

# Regex to detect class definition: template<...> class ClassName
CLASS_REGEX = re.compile(r'class\s+(\w+)\s*(?:;|{|:)')

# Regex to detect public/protected/private
ACCESS_REGEX = re.compile(r'(public|protected|private):')

# Regex to detect methods marked with EIGEN_DEVICE_FUNC
# Matches: EIGEN_DEVICE_FUNC [modifiers] ReturnType MethodName (Args...
# \w+ for normal names, or operator\s*[^(\s]+ for operators
METHOD_REGEX = re.compile(r'EIGEN_DEVICE_FUNC\s+(?:(?:EIGEN_CONSTEXPR|EIGEN_STRONG_INLINE|inline|virtual|static|const)\s+)*[\w:<>*&]+\s+(operator\s*[^(\s]+|\w+)\s*\(')

# Simple regex for methods that might be defined differently or macros
# This is a heuristic.
SIMPLE_METHOD_REGEX = re.compile(r'^\s*(\w+)\s*\(')

def extract_api(root_dir):
    api_map = {}
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.h'):
                continue
            
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Naive parsing: split by lines is risky for multi-line decls, 
            # but regex on full content is hard to track class scopes without a proper parser.
            # Let's try to track current class and access modifier line by line.
            
            current_class = None
            current_access = 'private' # Default for class
            brace_depth = 0
            
            # Pre-filter lines to look for class and methods
            lines = content.split('\n')
            
            line_buffer = ""
            
            for line in lines:
                original_line = line
                # Remove comments
                if '//' in line:
                    line = line.split('//')[0]
                line = line.strip()
                if not line:
                    continue

                # Check for Class
                class_match = CLASS_REGEX.search(line)
                if class_match:
                    current_class = class_match.group(1)
                    if current_class not in api_map:
                        api_map[current_class] = set()
                    current_access = 'private' # Reset access
                    line_buffer = "" # Reset buffer on class change
                    
                # Check for access modifiers
                access_match = ACCESS_REGEX.search(line)
                if access_match:
                    current_access = access_match.group(1)
                    line_buffer = "" # Reset buffer on section change
                
                # Buffer accumulation
                # We append the cleaned line to buffer with a space
                if line_buffer:
                    line_buffer += " " + line
                else:
                    line_buffer = line

                # Check if buffer contains a potential method end (parenthesis)
                if '(' in line_buffer:
                     # We only care about public or DenseBase specific headers often use EIGEN_DEVICE_FUNC
                    if current_access == 'public' or file.endswith('DenseBase.h') or file.endswith('Matrix.h'):
                        # Check for method
                        method_match = METHOD_REGEX.search(line_buffer)
                        if method_match:
                            method_name = method_match.group(1)
                            if current_class:
                                api_map[current_class].add(method_name)
                            else:
                                clean_filename = file.replace('.h', '')
                                if clean_filename not in api_map:
                                    api_map[clean_filename] = set()
                                api_map[clean_filename].add(method_name)
                            # Reset buffer after run
                            line_buffer = ""
                        elif ';' in line or '{' in line:
                             # If we reached semantic end of statement but no match, clear buffer
                             line_buffer = ""
                elif ';' in line or '{' in line or '}' in line or ':' in line:
                    # Clear buffer if we hit end of statement without '('
                     line_buffer = ""

    # Convert sets to sorted lists
    final_map = {}
    for k, v in api_map.items():
        final_map[k] = sorted(list(v))
        
    return final_map

if __name__ == '__main__':
    print(f"Scanning {EIGEN_SRC_DIR}...")
    api_data = extract_api(EIGEN_SRC_DIR)
    
    # Post-processing: Merge DenseBase into generic categories if needed, 
    # but for now keep raw classes.
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(api_data, f, indent=2)
        
    print(f"Extracted API to {OUTPUT_FILE}")
    
    # Print stats
    total_methods = sum(len(v) for v in api_data.values())
    print(f"Total Classes Found: {len(api_data)}")
    print(f"Total Methods Found: {total_methods}")
