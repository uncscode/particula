"""
Generate the API reference documentation for the Particula package.
Outputs line-based JSON where each line = {"function_name": "doc text..."}.
Headings are matched against Markdown lines that start with '## '.
"""

import os
import glob
import json
import shutil  # Uncomment if you'd like to remove .md files afterwards
from pathlib import Path

from handsdown.generators.material import MaterialGenerator
from handsdown.processors.pep257 import PEP257DocstringProcessor
from handsdown.utils.path_finder import PathFinder


# -----------------------------------------------------------------------------
# Read the raw Markdown text
# -----------------------------------------------------------------------------
def read_md_content(md_file_path: Path) -> str:
    """Read entire Markdown file as text."""
    with open(md_file_path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------------------------------------------------------
# Parse out '## FunctionName' headings and their content
# -----------------------------------------------------------------------------
def parse_markdown_headings(md_text: str):
    """
    Given the text of a Markdown file, find all headings that start with "## ".
    For each heading:
      - The heading text (after "## ") = function name
      - The content extends until the next "## " or end of file.
    Returns a list of (func_name, doc_content).
    """
    lines = md_text.split("\n")
    results = []
    current_func_name = None
    current_content_lines = []

    for line in lines:
        if line.startswith("## "):
            # If we were already capturing a previous function, store it
            if current_func_name is not None:
                results.append(
                    (current_func_name, "\n".join(current_content_lines))
                )
                current_content_lines = []

            # Start a new function heading
            current_func_name = line[3:].strip()  # everything after '## '
        else:
            # If we are in a heading's content, add this line
            if current_func_name is not None:
                current_content_lines.append(line)

    # End of file: store the last one if present
    if current_func_name is not None:
        results.append((current_func_name, "\n".join(current_content_lines)))

    return results


# -----------------------------------------------------------------------------
# Path setup and config
# -----------------------------------------------------------------------------
repo_path = Path.cwd()

# Exclude certain directories and gather .py source paths
path_finder = PathFinder(repo_path)
path_finder.exclude(
    "tests/*",
    "build/*, docs/*, .venv/**, private_dev/**, .git/*, .vscode/*, .github/*",
)
source_paths = path_finder.glob("**/*.py")
source_paths_list = list(source_paths)

# Filter only your package .py files
filtered_paths = [
    p for p in source_paths_list if "particula/particula" in p.as_posix()
]

# Ensure output directories exist
os.makedirs(repo_path / "docs/.assets/api_reference", exist_ok=True)
os.makedirs(
    repo_path / "site/development/single_page_reference", exist_ok=True
)

# -----------------------------------------------------------------------------
# Handsdown generator
# -----------------------------------------------------------------------------
handsdown = MaterialGenerator(
    input_path=repo_path,
    output_path=repo_path / "docs/.assets/api_reference",
    source_paths=filtered_paths,
    source_code_url="https://github.com/uncscode/particula/blob/main/",
    docstring_processor=PEP257DocstringProcessor(),
)


# -----------------------------------------------------------------------------
# Merge subdirectory .md files into line-based JSON
# -----------------------------------------------------------------------------
def process_subfolder(md_files, subfolder_name):
    """
    For each .md file (including deeper subfolders under <subfolder_name>):
      1) Parse out headings (## SomeFunction).
      2) Write a single JSON line for each heading to Particula_API_reference_<subfolder_name>.txt
         with the key as the function name, value as the doc string block.
    """
    output_dir = repo_path / "site/development/single_page_reference"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_file_path = (
        output_dir / f"Particula_API_reference_{subfolder_name}.json"
    )

    with open(out_file_path, "w", encoding="utf-8") as out_txt:
        for md_file_path in md_files:
            md_content = read_md_content(Path(md_file_path))
            headings = parse_markdown_headings(md_content)

            for func_name, doc_block in headings:
                # Each line is a JSON object: { "<func_name>": "<doc_block>" }
                line_data = {func_name: doc_block}
                out_txt.write(json.dumps(line_data, ensure_ascii=False))
                out_txt.write("\n")


# -----------------------------------------------------------------------------
# 1. Generate Handsdown docs
# -----------------------------------------------------------------------------
handsdown.generate_docs()

# -----------------------------------------------------------------------------
# 2. Identify first-level subdirectories under docs/.assets/api_reference/particula
# -----------------------------------------------------------------------------
api_ref_dir = repo_path / "docs/.assets/api_reference/particula"
subdirectories = [d for d in api_ref_dir.iterdir() if d.is_dir()]

# -----------------------------------------------------------------------------
# 3. For each subdirectory, gather .md files (recursively), skip index.md,
#    parse headings, and write them line-by-line as JSON
# -----------------------------------------------------------------------------
for subdir in subdirectories:
    all_md_files = sorted(glob.glob(str(subdir / "**/*.md"), recursive=True))
    # Optionally skip index.md
    all_md_files = [f for f in all_md_files if not f.endswith("index.md")]

    if all_md_files:
        process_subfolder(
            md_files=all_md_files,
            subfolder_name=subdir.name,  # e.g. 'xxx', 'yyy'
        )

# -----------------------------------------------------------------------------
# 4. (Optional) Remove original .md files after generating
# -----------------------------------------------------------------------------
# shutil.rmtree(api_ref_dir)
