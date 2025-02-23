# %%
"""
Convert all notebooks in `How-To-Guides` directory to Markdown files.

- Looks for *.ipynb in `How-To-Guides`.
- Uses nbconvert to produce Markdown outputs with Python code blocks.
- Outputs *.md files into a corresponding `How-To-Guides_markdown` directory.
"""
import os
import glob
from pathlib import Path
import nbformat
from nbconvert import MarkdownExporter

# %%
def convert_notebooks_to_markdown(
    input_dir: str = "How-To-Guides",
    output_dir: str = "How-To-Guides_markdown",
) -> None:
    """
    Convert all .ipynb files in the `input_dir` to Markdown files,
    placing results in the `output_dir`.

    Parameters
    ----------
    input_dir : str
        Directory containing Jupyter notebooks to convert.
    output_dir : str
        Directory to write resulting Markdown files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all .ipynb notebooks in the input directory
    notebook_files = glob.glob(os.path.join(input_dir, "*.ipynb"))

    # Configure the Markdown exporter
    # You can use different templates or exporters if you wish
    exporter = MarkdownExporter()
    exporter.template_name = "classic"  # Basic, widely available template

    for nb_file in notebook_files:
        notebook_name = os.path.splitext(os.path.basename(nb_file))[0]
        out_file = os.path.join(output_dir, f"{notebook_name}.md")

        # Load the notebook
        with open(nb_file, "r", encoding="utf-8") as f:
            nb_content = nbformat.read(f, as_version=4)

        # Convert to Markdown
        body, _ = exporter.from_notebook_node(nb_content)

        # Write the Markdown content to a file
        with open(out_file, "w", encoding="utf-8") as f_out:
            f_out.write(body)

        print(f"Converted: {nb_file} -> {out_file}")
