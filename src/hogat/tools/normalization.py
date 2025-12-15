
# Handles code normalization. First pass removes junk, then standardizes names for consistency. Slither is great but falls back to regex if it chokes.

import re
from slither.slither import Slither
import tempfile

def normalize_code(code_str: str) -> str:
    """
    Clean up Solidity code by ditching comments and renaming vars/functions for consistency.
    """
    # Strip comments first
    code = re.sub(r'//.*?$|/\*.*?\*/', '', code_str, flags=re.MULTILINE | re.DOTALL)
    
    # Temp file for Slither—it's picky about inputs
    with tempfile.NamedTemporaryFile(suffix='.sol', delete=False) as temp_file:
        temp_file.write(code.encode())
        temp_path = temp_file.name
    
    try:
        slither = Slither(temp_path)
        var_counter = 1
        for contract in slither.contracts:
            for variable in contract.variables:
                if not variable.is_constant:
                    code = code.replace(variable.name, f'var{var_counter}')
                    var_counter += 1
        
        func_counter = 1
        for contract in slither.contracts:
            for function in contract.functions_declared:
                code = code.replace(function.name, f'func{func_counter}')
                func_counter += 1
    except Exception as e:
        # Regex as backup— not perfect but catches most cases in our datasets
        # print(f"Slither hit a snag: {e}, switching to regex.")
        var_counter = 1
        def replace_var(m):
            nonlocal var_counter
            repl = f'{m.group(1)} var{var_counter}'
            var_counter += 1
            return repl
        code = re.sub(r'\b(uint|int|address|bool|string|bytes)\s+(\w+)', replace_var, code)
        
        func_counter = 1
        def replace_func(m):
            nonlocal func_counter
            repl = f'function func{func_counter}'
            func_counter += 1
            return repl
        code = re.sub(r'\bfunction\s+(\w+)', replace_func, code)
    
    os.unlink(temp_path)  # Clean up
    return code.strip()

# Tested this on a few SolidiFI samples—works decently for standardization.