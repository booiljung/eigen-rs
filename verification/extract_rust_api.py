
import os
import re
import json

RUST_SRC_DIR = 'src/core'
OUTPUT_FILE = 'tmp/verification/rust_api_list.json'

# Regex to detect public functions
# pub fn name<...>(...
PUB_FN_REGEX = re.compile(r'pub\s+fn\s+(\w+)\s*(?:<[^>]+>)?\s*\(')

def parse_impl_line(line):
    """
    Extracts the Struct name from a line starting with 'impl'.
    Handles nested generics <...> by counting brackets.
    Returns struct_name or None.
    """
    # Remove 'impl'
    if not line.startswith('impl'):
        return None
    rest = line[4:].strip()
    
    # Skip generics if present
    if rest.startswith('<'):
        depth = 0
        idx = 0
        for i, char in enumerate(rest):
            if char == '<':
                depth += 1
            elif char == '>':
                depth -= 1
                if depth == 0:
                    idx = i + 1
                    break
        rest = rest[idx:].strip()
        
    # Now rest should be "StructName..." or "Trait for StructName..."
    # or "StructName<...>"
    
    # Check for "for"
    if ' for ' in rest:
        parts = rest.split(' for ')
        struct_part = parts[1].strip()
    else:
        struct_part = rest
        
    # Extract the first identifier
    # StructName might be followed by <, {, where, or space
    m = re.match(r'([\w:]+)', struct_part)
    if m:
        name = m.group(1)
        # If it has ::, take the last part
        if '::' in name:
            name = name.split('::')[-1]
        return name
    return None

def extract_api(root_dir):
    api_map = {}
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.rs'):
                continue
                
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            current_struct = 'Global' # Default
            
            for line in lines:
                line = line.strip()
                if line.startswith('//'):
                    continue
                    
                # Check for impl
                if line.startswith('impl'):
                    struct_name = parse_impl_line(line)
                    if struct_name:
                        current_struct = struct_name
                        if current_struct not in api_map:
                            api_map[current_struct] = set()
                        
                # Check for pub fn
                fn_match = PUB_FN_REGEX.search(line)
                if fn_match:
                    method_name = fn_match.group(1)
                    if current_struct not in api_map:
                         api_map[current_struct] = set()
                    api_map[current_struct].add(method_name)

    # Convert sets to sorted lists
    final_map = {}
    for k, v in api_map.items():
        final_map[k] = sorted(list(v))
        
    return final_map

if __name__ == '__main__':
    print(f"Scanning {RUST_SRC_DIR}...")
    api_data = extract_api(RUST_SRC_DIR)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(api_data, f, indent=2)
        
    print(f"Extracted API to {OUTPUT_FILE}")
    total = sum(len(v) for v in api_data.values())
    print(f"Total Structs: {len(api_data)}")
    print(f"Total Methods: {total}")
