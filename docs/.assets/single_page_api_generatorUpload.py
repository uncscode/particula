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

# Set the repository path
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

# Define a flat output folder for markdown files
output_folder = repo_path / "site/development/api_reference"
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

# Flatten any nested markdown files into the flat output folder
for md_file in list(output_folder.rglob("*.md")):
    if md_file.parent != output_folder:
        # Create a new filename by prefixing with the parent directory name
        new_name = f"{md_file.parent.name}_{md_file.name}"
        dest_file = output_folder / new_name
        shutil.move(str(md_file), str(dest_file))

# Optionally, remove any empty subdirectories within the output folder
for subdir in output_folder.iterdir():
    if subdir.is_dir():
        try:
            subdir.rmdir()  # Removes the directory if it is empty
        except OSError:
            pass  # Skip if the directory is not empty
