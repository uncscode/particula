"""Build complete study downloads for the website and installed docs."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def write_study_zip(archive, root, config, source):
    """Write a reproducible archive with the script and its declared inputs."""
    flag = " --cuda" if config.get("device") == "cuda" else ""
    readme = (
        "Extract this entire archive into one directory. Install the "
        "Particula version documented on the study page and its "
        "optional dependencies. From the extracted directory run:\n\n"
        f"python example.py{flag}\n\n"
        "Helper/data paths are preserved below that directory. CUDA "
        "studies require an available NVIDIA GPU and Warp.\n"
    )
    entries = {"example.py": source, "README.txt": readme}
    for name in config["inputs"]:
        destination = Path(config.get("input_destinations", {}).get(name, name))
        if destination.is_absolute() or ".." in destination.parts:
            raise ValueError(
                "Study bundle input must be a relative contained path"
            )
        if destination.as_posix() in entries:
            raise ValueError("Study input collides with a bundle entry")
        entries[destination.as_posix()] = (root / name).read_bytes()
    with ZipFile(archive, "w") as bundle:
        for filename, content in entries.items():
            # Use the same member timestamps across wheel and website builds.
            bundle.writestr(
                ZipInfo(filename), content, compress_type=ZIP_DEFLATED
            )
