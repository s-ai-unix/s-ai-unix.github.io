import re
import os

def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1. Check for unclosed $$ blocks
    # This is tricky because $$ is used for both start and end.
    # An odd number of $$ overall is a strong indicator.
    # But we should also check if they are "balanced" in the sense that
    # they are not crossing other structural boundaries (like code blocks).
    
    # Exclude code blocks
    content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    
    display_math_count = len(re.findall(r'\$\$', content_no_code))
    if display_math_count % 2 != 0:
        # Check if it's Perl-like code (not in code block?)
        # 2014 posts are likely Perl
        if "2014" not in filepath:
            print(f"SUSPECT: Odd $$ count ({display_math_count}) in {filepath}")
        elif "$$" in content_no_code and "$$ref" not in content_no_code:
            print(f"SUSPECT: Odd $$ count ({display_math_count}) in {filepath} (may be Perl)")

    # 2. Check for \( ... \) mismatches
    # These are usually on the same line, but can be multi-line.
    # We look for \( that doesn't have a following \) before the next \(
    # Or \) that doesn't have a preceding \(
    
    lines = content_no_code.splitlines()
    for i, line in enumerate(lines):
        # Simplistic line-based check for common errors
        if '\\(' in line and '\\)' not in line:
            # Check if closed on next line
            if i + 1 < len(lines) and '\\)' not in lines[i+1]:
                print(f"SUSPECT: Unclosed \\( on line {i+1} of {filepath}")
        if '\\)' in line and '\\(' not in line:
            if i > 0 and '\\(' not in lines[i-1]:
                print(f"SUSPECT: Unopened \\) on line {i+1} of {filepath}")

    # 3. Check for \[ ... \] mismatches
    for i, line in enumerate(lines):
        if '\\[' in line and '\\]' not in line:
            if i + 1 < len(lines) and '\\]' not in lines[i+1]:
                 print(f"SUSPECT: Unclosed \\[ on line {i+1} of {filepath}")
        if '\\]' in line and '\\[' not in line:
             if i > 0 and '\\[' not in lines[i-1]:
                 print(f"SUSPECT: Unopened \\] on line {i+1} of {filepath}")

for root, dirs, files in os.walk("content"):
    for file in files:
        if file.endswith(".md"):
            check_file(os.path.join(root, file))
