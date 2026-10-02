"""Assemble a delimiter-separated cell source into a Jupyter notebook.

Usage:
    python scripts/build_notebook.py <source.src> <output.ipynb>

Cell sources live in a plain text file rather than inline in Python, because
notebook code contains its own quotes and docstrings, and nesting triple-quoted
strings inside a generator script is a reliable way to corrupt it.

Source format: blocks separated by a line containing exactly `===MD===`
(markdown cell) or `===CODE===` (code cell).
"""

import sys

import nbformat as nbf

if len(sys.argv) != 3:
    sys.exit(__doc__)

SRC, OUT = sys.argv[1], sys.argv[2]

raw = open(SRC).read()

cells = []
current_type = None
buffer = []

for line in raw.split("\n"):
    stripped = line.strip()
    if stripped in ("===MD===", "===CODE==="):
        if current_type is not None:
            cells.append((current_type, "\n".join(buffer).strip("\n")))
        current_type = "markdown" if stripped == "===MD===" else "code"
        buffer = []
    else:
        buffer.append(line)

if current_type is not None:
    cells.append((current_type, "\n".join(buffer).strip("\n")))

cells = [(t, s) for t, s in cells if s.strip()]
if not cells:
    sys.exit("no cells parsed from " + SRC)

nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell(s) if t == "markdown" else nbf.v4.new_code_cell(s)
    for t, s in cells
]
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
    "authors": [{"name": "Sumit Jha"}],
}

nbf.write(nb, OUT)

n_md = sum(1 for t, _ in cells if t == "markdown")
n_code = len(cells) - n_md
print(f"wrote {OUT}: {len(cells)} cells ({n_md} markdown, {n_code} code)")