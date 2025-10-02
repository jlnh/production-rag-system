#!/usr/bin/env python3
"""Auto-fix common mypy type errors."""

import re
from pathlib import Path

def fix_file(filepath: Path) -> None:
    """Fix common type errors in a file."""
    content = filepath.read_text()
    original = content

    # Fix: def function(...) -> add -> None for functions with no return
    # Pattern: def name(...):  followed by """..."""  with no return statement
    content = re.sub(
        r'(async )?def ([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\):(\s*""")',
        lambda m: f'{m.group(1) or ""}def {m.group(2)}({add_any_to_kwargs(m.group(3))}):{m.group(4)}',
        content
    )

    if content != original:
        print(f"Fixed: {filepath}")
        filepath.write_text(content)

def add_any_to_kwargs(params: str) -> str:
    """Add Any type hint to **kwargs parameters."""
    if '**' in params and ': Any' not in params:
        params = re.sub(r'\*\*([a-zA-Z_][a-zA-Z0-9_]*)', r'**\1: Any', params)
    return params

def main():
    src_dir = Path("src/rag_system")

    # Process all Python files
    for pyfile in src_dir.rglob("*.py"):
        if pyfile.name != "__pycache__":
            fix_file(pyfile)

if __name__ == "__main__":
    main()
