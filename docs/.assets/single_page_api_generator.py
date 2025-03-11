"""
Generate the API reference documentation for the Particula package.
To be used by GPT Assistants to generate the API reference documentation.
"""

import os
# import shutil
from pathlib import Path
from handsdown.generators.material import MaterialGenerator
from handsdown.processors.pep257 import PEP257DocstringProcessor
from handsdown.utils.path_finder import PathFinder
# import glob

from single_page_ref_generator import merge_markdown_files

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
    repo_path / "site/development/single_page_reference", exist_ok=True
)

# Initialize Handsdown generator
handsdown = MaterialGenerator(
    input_path=repo_path,
    output_path=repo_path / "docs/.assets/api_reference",
    source_paths=filtered_paths,
    source_code_url="https://github.com/uncscode/particula/blob/main/",
    docstring_processor=PEP257DocstringProcessor(),
)


# def _chunk_write_content(md_file_path, file_object, max_size, current_size):
#     """Write file content to file_object chunk-wise without exceeding
#     max_size.
#     """
#     with open(md_file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             line_bytes = line.encode("utf-8")
#             if current_size + len(line_bytes) > max_size:
#                 # We’ve run out of space in the current file;
#                 # break, so the caller can start a new file if needed
#                 break
#             file_object.write(line)
#             current_size += len(line_bytes)
#     return current_size


# # 1. Generate docs
# handsdown.generate_docs()

# # 3. Merge and chunk files
# max_chunk_size = 250 * 1024  # ~250 KB
# all_md_files = glob.glob(
#     str(repo_path / "docs/.assets/api_reference/**/*.md"), recursive=True
# )

# output_dir = repo_path / "site/development/single_page_reference"
# output_dir.mkdir(parents=True, exist_ok=True)

# Merge all Markdown files into one
merge_markdown_files(
    input_glob=str(repo_path / "docs/.assets/api_reference/**/*.md"),
    output_file=repo_path
    / "site/development/single_page_reference/Particula_API_reference.md",
    remove_dir=repo_path / "docs/.assets/api_reference",
    skip_filenames=["index.md"],
)


# def get_output_file_path(index):
#     return output_dir / f"Particula_API_reference_part{index}.md"


# part_index = 1
# current_size = 0

# # Open the first chunk
# with open(get_output_file_path(part_index), "w", encoding="utf-8") as merged:
#     for md_file_path in sorted(all_md_files):
#         # Optionally skip the main index.md
#         if md_file_path.endswith("index.md"):
#             continue

#         md_file_name = os.path.basename(md_file_path)
#         heading = f"\n\n---\n# {md_file_name}\n\n"
#         heading_bytes = heading.encode("utf-8")
#         heading_size = len(heading_bytes)

#         # If adding heading exceeds the limit, start a new part
#         if current_size + heading_size > max_chunk_size:
#             # Close the current 'merged' file by exiting its context
#             # and open a new one in a new context
#             part_index += 1
#             with open(
#                 get_output_file_path(part_index), "w", encoding="utf-8"
#             ) as new_merged:
#                 new_merged.write(heading)
#                 current_size = heading_size
#                 _chunk_write_content(
#                     md_file_path, new_merged, max_chunk_size, current_size
#                 )
#             # Re-open the chunk file for subsequent iteration
#             with open(
#                 get_output_file_path(part_index), "a", encoding="utf-8"
#             ) as merged_append:
#                 current_size = os.path.getsize(
#                     get_output_file_path(part_index)
#                 )
#             # Move on to the next .md file
#             continue
#         else:
#             merged.write(heading)
#             current_size += heading_size
#             # Write file content chunk-wise
#             current_size = _chunk_write_content(
#                 md_file_path,
#                 merged,
#                 max_chunk_size,
#                 current_size,
#             )

# # Clean up: remove the directory with original .md files
# shutil.rmtree(repo_path / "docs/.assets/api_reference")
