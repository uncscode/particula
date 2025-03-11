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


def line_stream(all_md_files, skip_filenames):
    """
    Yield lines from each Markdown file (plus a heading), skipping certain
    filenames.
    """
    for md_file_path in all_md_files:
        filename = os.path.basename(md_file_path)
        if filename in skip_filenames:
            continue

        # Yield a heading so we know which file's content follows
        heading = f"\n\n---\n# {filename}\n\n"
        yield heading

        # Then yield each line of the file
        with open(md_file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line


def chunked_line_stream(line_iter, max_chunk_size):
    """
    Group lines from `line_iter` into chunks that stay below `max_chunk_size`
    bytes.
    Yields lists of lines, each forming one chunk.
    """
    current_chunk = []
    current_size = 0

    for line in line_iter:
        line_size = len(line.encode("utf-8"))

        # If adding this line would exceed the chunk size *and* we already
        # have lines,
        # yield the current chunk first. Then start a new chunk with this line.
        if current_size + line_size > max_chunk_size and current_chunk:
            yield current_chunk
            current_chunk = [line]
            current_size = line_size
        else:
            # Either the chunk is empty or the line fits
            current_chunk.append(line)
            current_size += line_size

    # Yield the final chunk if it has any lines
    if current_chunk:
        yield current_chunk


def merge_markdown_files(
    input_glob: str,
    output_file: Path,
    remove_dir: Path,
    skip_filenames: list[str] = None,
    max_chunk_size_kb: int = 250,
) -> None:
    """
    Merge all Markdown files matching `input_glob` into multiple part files
    (each up to ~max_chunk_size_kb KB), then remove the directory `remove_dir`.

    Parameters
    ----------
    input_glob : str
        A glob pattern (including directory path) matching the .md
        files to merge.
        Example: 'docs/.assets/api_reference/**/*.md'
    output_file : Path
        The *base* output path for the merged Markdown files.
        Chunked output files will be named like 'BASE_part1.md', 
        BASE_part2.md', etc.
    remove_dir : Path
        The directory you want to remove entirely after merging.
    skip_filenames : list[str], optional
        Filenames to skip (e.g., ['index.md']). Defaults to None.
    max_chunk_size_kb : int, optional
        Approximate size limit (in kilobytes) for each chunked file,
        defaults to 250 KB.
    """
    if skip_filenames is None:
        skip_filenames = []

    # Get all matching Markdown files
    all_md_files = glob.glob(input_glob, recursive=True)
    all_md_files.sort()

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    max_chunk_size = max_chunk_size_kb * 1024  # convert to bytes

    # 1) Create a generator that yields lines (including headings) from each
    # file.
    lines = line_stream(all_md_files, skip_filenames)

    # 2) Chunk those lines so we don't exceed the size limit.
    chunked_lines = chunked_line_stream(lines, max_chunk_size)

    # 3) Write each chunk into its own file using a 'with' context.
    part_index = 1
    for chunk in chunked_lines:
        chunk_file = (
            output_file.parent
            / f"{output_file.stem}_part{part_index}{output_file.suffix}"
        )
        with open(chunk_file, "w", encoding="utf-8") as merged:
            merged.write("".join(chunk))
        part_index += 1

    # Finally, remove the directory containing the original Markdown files
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
os.makedirs(
    repo_path / "site/development/single_page_reference", exist_ok=True
)

# Convert Notebooks to Markdown
convert_notebooks_to_markdown(
    notebook_paths=filtered_paths,
    output_dir=repo_path / "docs/.assets/tutorial_reference",
)

# Merge all Markdown files into one
merge_markdown_files(
    input_glob=str(repo_path / "docs/.assets/tutorial_reference/**/*.md"),
    output_file=repo_path
    / "site/development/single_page_reference/Particula_Tutorial_Reference.md",
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
    / "site/development/single_page_reference/Particula_How-To-Guide_Reference.md",
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
    / "site/development/single_page_reference/Particula_Discussion_Reference.md",
    remove_dir=repo_path / "docs/.assets/discussion_reference",
    skip_filenames=["index.md"],
)
