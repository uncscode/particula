# %%
"""
For vector store upload, we need to convert all notebooks and python files in
a directory to Markdown files.
"""
import os
import shutil
from pathlib import Path
import nbformat
from nbconvert import MarkdownExporter
from handsdown.utils.path_finder import PathFinder
from handsdown.generators.material import MaterialGenerator
from handsdown.processors.pep257 import PEP257DocstringProcessor


def convert_notebooks_to_markdown(notebook_paths, output_dir):
    """
    Convert a list of notebooks (.ipynb) and markdown files (.md) to Markdown
    files, writing the results into the specified output directory.
    """
    exporter = MarkdownExporter()
    exporter.template_name = "classic"

    for nb_file in notebook_paths:
        nb_file = Path(nb_file)  # Ensure we have a Path object
        rel_path = nb_file.relative_to(repo_path)
        new_output_dir = Path(output_dir) / rel_path.parent
        new_output_dir.mkdir(parents=True, exist_ok=True)

        if nb_file.suffix == ".md":
            # Copy .md files directly to the output folder
            shutil.copy(nb_file, new_output_dir / nb_file.name)
            continue

        # Derive the output filename from the notebook name
        notebook_name = nb_file.stem
        out_file = Path(output_dir) / f"{notebook_name}.md"

        # Load the notebook and convert to Markdown
        out_file = new_output_dir / f"{notebook_name}.md"

        with open(nb_file, "r", encoding="utf-8") as f:
            nb_content = nbformat.read(f, as_version=4)
        body, _ = exporter.from_notebook_node(nb_content)

        # Write the Markdown content to a file
        with open(out_file, "w", encoding="utf-8") as f_out:
            f_out.write(body)


# Set repository path and initialize the path finder (with exclusions)
repo_path = Path.cwd()
path_finder = PathFinder(repo_path)
path_finder.exclude(
    "tests/*",
    "build/*, docs/*, .venv/**, private_dev/**, .git/*, .vscode/*, .github/*",
)

# Find all .ipynb and .md files
source_paths_ipynb = list(path_finder.glob("**/*.ipynb"))
source_paths_md = list(path_finder.glob("**/*.md"))
source_paths = source_paths_ipynb + source_paths_md
source_paths_list = list(source_paths)

# -------------------------
# Process Examples Notebooks
# -------------------------
filtered_examples = [
    p for p in source_paths_list if "particula/docs/Examples/" in p.as_posix()
]
output_examples = repo_path / "docs/.assets/temp/"
os.makedirs(output_examples, exist_ok=True)
convert_notebooks_to_markdown(filtered_examples, output_examples)

# -------------------------
# Process Theory Notebooks
# -------------------------
filtered_theory = [
    p for p in source_paths_list if "particula/docs/Theory/" in p.as_posix()
]
output_theory = repo_path / "docs/.assets/temp/"
os.makedirs(output_theory, exist_ok=True)
convert_notebooks_to_markdown(filtered_theory, output_theory)

# ----------------------------------------
# Process API python files using Handsdown
# ----------------------------------------

source_paths = path_finder.glob("**/*.py")
source_paths_list = list(source_paths)
filtered_paths = [
    p for p in source_paths_list if "particula/particula" in p.as_posix()
]

# Define an output folder for markdown files
output_folder = repo_path / "docs/.assets/temp/"
os.makedirs(output_folder, exist_ok=True)

# Initialize Handsdown generator with the flat output folder as output_path
handsdown = MaterialGenerator(
    input_path=repo_path,
    output_path=output_folder,
    source_paths=filtered_paths,
    source_code_url="https://github.com/uncscode/particula/blob/main/",
    docstring_processor=PEP257DocstringProcessor(),
)

# 1. Generate the multi-file docs.
handsdown.generate_docs()

# 2. Generate index.md.
handsdown.generate_index()
