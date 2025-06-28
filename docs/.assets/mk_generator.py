"""Generate documentation using Handsdown."""

import os
from pathlib import Path

from handsdown.generators.material import MaterialGenerator
from handsdown.processors.pep257 import PEP257DocstringProcessor
from handsdown.utils.path_finder import PathFinder

repo_path = Path.cwd()

# this little tool works like `pathlib.Path.glob` with some extra magic
# but in this case `repo_path.glob("**/*.py")` would do as well
path_finder = PathFinder(repo_path)

# no docs for tests and build
path_finder.exclude("tests/*", "build/*")

source_paths = path_finder.glob("**/*.py")

source_paths_list = list(source_paths)
filtered_paths = [
    p for p in source_paths_list if "particula/particula" in p.as_posix()
]

# generate folder structure, if needed
os.makedirs(repo_path / "docs/API/", exist_ok=True)

# initialize generator
handsdown = MaterialGenerator(
    input_path=repo_path,
    output_path=repo_path / "docs/API",
    source_paths=filtered_paths,
    source_code_url="https://github.com/uncscode/particula/blob/main/",
    docstring_processor=PEP257DocstringProcessor(),
)

# generate all docs at once
handsdown.generate_docs()

# generate index.md file
handsdown.generate_index()
