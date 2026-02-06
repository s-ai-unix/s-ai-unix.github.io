import re
import os

def check_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1. Check for multiline inline math $...$
    # We look for a line with an odd number of $ that is followed by a line with an odd number of $
    lines = content.splitlines()
    for i in range(len(lines) - 1):
        c1 = len(re.findall(r'(?<!\\)\$', lines[i]))
        if c1 % 2 != 0:
            # Possible start of multiline inline math
            # But wait, it could be a $$ block starting/ending
            if '$$' not in lines[i]:
                print(f"MULTILINE_INLINE: {filepath}:{i+1} -> {lines[i]}")

    # 2. Check for Unicode characters inside $$ blocks
    # This is a bit complex as we need to find the blocks
    blocks = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)
    for block in blocks:
        # Look for non-ascii characters that are not common in comments
        # (Though we should probably just look for specific ones like ∯, ✓)
        if '∯' in block:
            print(f"UNICODE_IN_MATH (∯): {filepath}")
        if '✓' in block:
            print(f"UNICODE_IN_MATH (✓): {filepath}")

    # 3. Check for mismatched \( and \) on same line
    for i, line in enumerate(lines):
        if line.count('\\(') != line.count('\\)'):
            print(f"MISMATCH_PAREN: {filepath}:{i+1} -> {line}")
        if line.count('\\[') != line.count('\\]'):
            print(f"MISMATCH_BRACKET: {filepath}:{i+1} -> {line}")

for root, dirs, files in os.walk("content"):
    for file in files:
        if file.endswith(".md"):
            check_file(os.path.join(root, file))
