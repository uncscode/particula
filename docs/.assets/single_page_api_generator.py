"""
Generate the API reference documentation for the Particula package.
To be used by GPT Assistants to generate the API reference documentation.
"""

import os
import shutil
from pathlib import Path
from handsdown.generators.material import MaterialGenerator
from handsdown.processors.pep257 import PEP257DocstringProcessor
from handsdown.utils.path_finder import PathFinder
import glob

# move to the package folder
repo_path = Path.cwd()

# Initialize path finder and exclude directories
path_finder = PathFinder(repo_path)
path_finder.exclude(
    "tests/*",
    "build/*, docs/*, .venv/**, private_dev/**, .git/*, .vscode/*, .github/*",
)
source_paths = path_finder.glob("**/*.py")


source_paths_list = list(source_paths)
filtered_paths = [
    p for p in source_paths_list if "particula/particula" in p.as_posix()
]

# Generate folder structure, if needed
os.makedirs(repo_path / "docs/.assets/api_reference", exist_ok=True)
os.makedirs(
    repo_path / "docs/development/single_page_reference", exist_ok=True
)

# Initialize Handsdown generator
handsdown = MaterialGenerator(
    input_path=repo_path,
    output_path=repo_path / "docs/.assets/api_reference",
    source_paths=filtered_paths,
    source_code_url="https://github.com/uncscode/particula/blob/main/",
    docstring_processor=PEP257DocstringProcessor(),
)

# 1. Generate the multi-file docs.
handsdown.generate_docs()

# 2. Generate index.md.
handsdown.generate_index()

# 3. Merge all .md files into one file.
#    Note: you can customize which files are included/excluded.
all_md_files = glob.glob(
    str(repo_path / "docs/.assets/api_reference/**/*.md"), recursive=True
)

output_file = (
    repo_path
    / "docs/development/single_page_reference/Particula_API_reference.md"
)

with open(output_file, "w", encoding="utf-8") as merged:
    for md_file_path in sorted(all_md_files):
        # Optionally skip the main index.md or other file patterns
        if md_file_path.endswith("index.md"):
            continue

        # Use the filename as a heading or separator
        md_file_name = os.path.basename(md_file_path)
        merged.write(f"\n\n---\n# {md_file_name}\n\n")

        with open(md_file_path, "r", encoding="utf-8") as f:
            merged.write(f.read())

# Remove the entire directory containing the merged files
# (rather than deleting individual files)
shutil.rmtree(repo_path / "docs/.assets/api_reference")
