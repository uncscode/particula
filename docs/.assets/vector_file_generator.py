# %%
"""Convert notebooks and Python files in a directory to Markdown."""

import os
import shutil
from pathlib import Path

import nbformat
from nbconvert import MarkdownExporter

try:
    from griffe import load
except ImportError:
    load = None
    print("Warning: griffe not installed, API docs will not be generated")


def convert_notebooks_to_markdown(notebook_paths, output_dir):
    """Convert notebooks and Markdown files to Markdown."""
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


# Set repository path
repo_path = Path.cwd()

# Find all .ipynb and .md files in docs directory
source_paths_ipynb = list(repo_path.glob("docs/**/*.ipynb"))
source_paths_md = list(repo_path.glob("docs/**/*.md"))
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
# Process API python files using griffe
# ----------------------------------------


def generate_module_markdown(obj, depth=0):
    """Generate markdown documentation for a module/class/function."""
    lines = []

    # Add heading
    if obj.kind.value == "module":
        lines.append(f"{'#' * (depth + 1)} Module: {obj.name}\n")
    elif obj.kind.value == "class":
        lines.append(f"{'#' * (depth + 2)} Class: {obj.name}\n")
    elif obj.kind.value == "function":
        lines.append(f"{'#' * (depth + 3)} Function: {obj.name}\n")
    elif obj.kind.value == "attribute":
        lines.append(f"{'#' * (depth + 3)} Attribute: {obj.name}\n")

    # Add signature for functions/methods
    if hasattr(obj, "parameters") and obj.parameters:
        params = ", ".join([p.name for p in obj.parameters])
        lines.append(f"\n**Signature:** `{obj.name}({params})`\n")

    # Add docstring
    if obj.docstring and obj.docstring.value:
        lines.append(f"\n{obj.docstring.value}\n")

    # Add source file path
    if hasattr(obj, "filepath"):
        lines.append(f"\n*Source: {obj.filepath}*\n")

    lines.append("\n---\n")
    return "\n".join(lines)


def generate_api_markdown(package_name, output_folder):
    """Generate markdown files for all modules in a package."""
    if load is None:
        print("Skipping API documentation: griffe not installed")
        return

    try:
        # Load the package
        package = load(package_name, docstring_parser="google")

        # Create output directory
        api_output = Path(output_folder) / "api"
        api_output.mkdir(parents=True, exist_ok=True)

        # Generate markdown for each top-level module
        for module_name, module_obj in package.modules.items():
            if module_obj.is_alias:
                continue

            md_content = []
            md_content.append(f"# API Reference: {module_name}\n\n")

            # Document the module
            if module_obj.docstring and module_obj.docstring.value:
                md_content.append(f"{module_obj.docstring.value}\n\n")

            # Document classes
            for obj in module_obj.classes.values():
                md_content.append(generate_module_markdown(obj, depth=1))

            # Document functions
            for obj in module_obj.functions.values():
                md_content.append(generate_module_markdown(obj, depth=1))

            # Write to file
            safe_name = module_name.replace(".", "_")
            output_file = api_output / f"{safe_name}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.writelines(md_content)

            print(f"Generated API docs: {output_file}")

    except Exception as e:
        print(f"Error generating API markdown: {e}")


# Define an output folder for markdown files
output_folder = repo_path / "docs/.assets/temp/"
os.makedirs(output_folder, exist_ok=True)

# Generate API documentation as markdown
generate_api_markdown("particula", output_folder)
