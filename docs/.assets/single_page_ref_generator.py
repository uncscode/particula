# %%
"""
Convert all notebooks in `How-To-Guides` directory to Markdown files.

- Looks for *.ipynb and .md in `How-To-Guides`.
- Uses nbconvert to produce Markdown outputs with Python code blocks.
- Outputs *.md files into a corresponding `How-To-Guides_markdown` directory.
"""
import os
import glob
import shutil
from pathlib import Path
import nbformat
from nbconvert import MarkdownExporter
from handsdown.utils.path_finder import PathFinder


def convert_notebooks_to_markdown(
    notebook_paths: list[str],
    output_dir: str,
) -> None:
    """
    Convert a list of .ipynb files to Markdown files,
    placing results in the `output_dir`.

    Arguments:
        - notebook_paths : list[str]
            List of full file paths to Jupyter notebooks to convert.
        - output_dir : str
            Directory to write resulting Markdown files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Configure the Markdown exporter
    exporter = MarkdownExporter()
    exporter.template_name = "classic"

    for nb_file in notebook_paths:
        # copy the md file to the output directory
        if nb_file.suffix == ".md":
            shutil.copy(nb_file, output_dir)
            continue
        # Derive the output filename from the notebook name
        notebook_name = Path(nb_file).stem
        out_file = Path(output_dir) / f"{notebook_name}.md"

        # Load the notebook
        with open(nb_file, "r", encoding="utf-8") as f:
            nb_content = nbformat.read(f, as_version=4)

        # Convert to Markdown
        body, _ = exporter.from_notebook_node(nb_content)

        # Write the Markdown content to a file
        with open(out_file, "w", encoding="utf-8") as f_out:
            f_out.write(body)


def merge_markdown_files(
    input_glob: str,
    output_file: Path,
    remove_dir: Path,
    skip_filenames: list[str] = None,
) -> None:
    """
    Merge all Markdown files matching `input_glob` into one output file,
    and then remove the entire directory specified by `remove_dir`.

    Arguments:
       - input_glob : str
            A glob pattern (including directory path) matching the .md files
            to merge.
            Example: 'docs/.assets/api_reference/**/*.md'
       - output_file : Path
            The output path for the merged Markdown file.
       - remove_dir : Path
            The directory you want to completely remove after merging.
       - skip_filenames : list[str], optional
            Filenames to skip (e.g., ['index.md']). Defaults to None.
    """
    if skip_filenames is None:
        skip_filenames = []

    all_md_files = glob.glob(input_glob, recursive=True)

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as merged:
        for md_file_path in sorted(all_md_files):
            filename = os.path.basename(md_file_path)

            # Skip certain filenames if needed
            if filename in skip_filenames:
                continue

            # Write a separator + the filename as a heading
            merged.write(f"\n\n---\n# {filename}\n\n")

            # Append the contents
            with open(md_file_path, "r", encoding="utf-8") as f:
                merged.write(f.read())

    # Remove the entire directory containing the merged files
    # (rather than deleting individual files)
    shutil.rmtree(remove_dir)


# move to the package folder
repo_path = Path.cwd()

# Initialize path finder and exclude directories
path_finder = PathFinder(repo_path)
path_finder.exclude(
    "tests/*",
    "build/*, docs/*, .venv/**, private_dev/**, .git/*, .vscode/*, .github/*",
)

source_paths_ipynb = list(path_finder.glob("**/*.ipynb"))
source_paths_md = list(path_finder.glob("**/*.md"))
source_paths = source_paths_ipynb + source_paths_md

# Tutorial Reference
source_paths_list = list(source_paths)
filtered_paths = [
    p for p in source_paths_list if "particula/docs/Tutorials/" in p.as_posix()
]

# Generate folder structure, if needed
os.makedirs(repo_path / "docs/.assets/tutorial_reference", exist_ok=True)

# Convert Notebooks to Markdown
convert_notebooks_to_markdown(
    notebook_paths=filtered_paths,
    output_dir=repo_path / "docs/.assets/tutorial_reference",
)

# Merge all Markdown files into one
merge_markdown_files(
    input_glob=str(repo_path / "docs/.assets/tutorial_reference/**/*.md"),
    output_file=repo_path
    / "docs/.assets/single_page_reference/Particula_Tutorial_Reference.md",
    remove_dir=repo_path / "docs/.assets/tutorial_reference",
    skip_filenames=["index.md"],
)

# How-To-Guide Reference
filtered_paths = [
    p
    for p in source_paths_list
    if "particula/docs/How-To-Guides/" in p.as_posix()
]

# Generate folder structure, if needed
os.makedirs(repo_path / "docs/.assets/how_to_guide_reference", exist_ok=True)

# Convert Notebooks to Markdown
convert_notebooks_to_markdown(
    notebook_paths=filtered_paths,
    output_dir=repo_path / "docs/.assets/how_to_guide_reference",
)

# Merge all Markdown files into one
merge_markdown_files(
    input_glob=str(repo_path / "docs/.assets/how_to_guide_reference/**/*.md"),
    output_file=repo_path
    / "docs/.assets/single_page_reference/Particula_How-To-Guide_Reference.md",
    remove_dir=repo_path / "docs/.assets/how_to_guide_reference",
    skip_filenames=["index.md"],
)

# Discussions
filtered_paths = [
    p
    for p in source_paths_list
    if "particula/docs/Discussions/" in p.as_posix()
]

# Generate folder structure, if needed
os.makedirs(repo_path / "docs/.assets/discussion_reference", exist_ok=True)

# Convert Notebooks to Markdown
convert_notebooks_to_markdown(
    notebook_paths=filtered_paths,
    output_dir=repo_path / "docs/.assets/discussion_reference",
)

# Merge all Markdown files into one
merge_markdown_files(
    input_glob=str(repo_path / "docs/.assets/discussion_reference/**/*.md"),
    output_file=repo_path
    / "docs/.assets/single_page_reference/Particula_Discussion_Reference.md",
    remove_dir=repo_path / "docs/.assets/discussion_reference",
    skip_filenames=["index.md"],
)
