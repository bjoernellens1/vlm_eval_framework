import json

notebook_path = "notebooks/embed_slam_comparison.ipynb"
output_path = "verify_notebook.py"

with open(notebook_path, "r") as f:
    nb = json.load(f)

code_cells = []
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        code_cells.append(source)

with open(output_path, "w") as f:
    f.write("\n\n".join(code_cells))

print(f"Extracted code to {output_path}")
