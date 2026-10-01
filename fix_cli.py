
import sys

with open('src/isaac/cli.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    stripped = line.strip()
    # If the line is a top-level command decorator or function, it should have 0 indentation.
    # If it's part of a function body, it should have 4 spaces (or more).
    if stripped.startswith('@app.command') or stripped.startswith('@goal_app.command') or \
       stripped.startswith('@kanban_app.command') or stripped.startswith('def ') or \
       stripped.startswith('app.add_typer') or stripped.startswith('mcp_app =') or \
       stripped.startswith('mcp_app.add_typer'):
        
        # Ensure no leading whitespace
        new_lines.append(stripped + '\n')
    elif line.startswith('    '):
        # If it's a body line, we keep the indentation but shift it back if we are in the shifted zone.
        # For now, let's just try to fix the obvious shifted top-levels.
        new_lines.append(line)
    else:
        new_lines.append(line)

with open('src/isaac/cli.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
