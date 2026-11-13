"""Check documentation ownership, links, source pairs, and published redirects.

Requires the MkDocs documentation environment (Markdown and PyYAML).
Run from any directory; --site-dir requires a completed full publication build.
"""

import argparse
import ast
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
import yaml

SECTIONS = ("Guide", "Examples", "Theory", "API", "Contribute")
OWNERS = {"guide", "examples", "theory", "contribute", "API"}


class Links(HTMLParser):
    """Collect HTML links, images, and anchor destinations."""

    def __init__(self, text):
        """Parse a rendered page or Markdown fragment."""
        super().__init__()
        self.urls = []
        self.ids = set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        """Collect only actual attributes, not code examples."""
        values = dict(attrs)
        if "id" in values:
            self.ids.add(values["id"])
        if tag == "a" and "name" in values:
            self.ids.add(values["name"])
        attr = "src" if tag == "img" else "href" if tag == "a" else None
        if attr and attr in values:
            self.urls.append(values[attr])


def source_text(path):
    """Read prose without confusing notebook code with Markdown links."""
    if path.suffix == ".ipynb":
        return "\n\n".join(
            "".join(cell["source"])
            for cell in json.loads(path.read_text())["cells"]
            if cell["cell_type"] == "markdown"
        )
    if path.suffix == ".py":
        prose = []
        in_markdown = False
        for line in path.read_text().splitlines():
            if re.match(r"# %%(?:\s|$)", line):
                in_markdown = "[markdown]" in line
                prose.append("")
            elif in_markdown and line.startswith("#"):
                prose.append(line.removeprefix("#").removeprefix(" "))
        return "\n".join(prose)
    return re.sub(r"\A---\n.*?\n---\n", "", path.read_text(), flags=re.S)


def nav_targets(items):
    """Yield the leaves of MkDocs' nested navigation."""
    for item in items:
        for value in item.values():
            if isinstance(value, list):
                yield from nav_targets(value)
            else:
                yield value


def local_target(url, source, root):
    """Resolve local links and current upstream/fork source URLs."""
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc:
        match = re.fullmatch(
            r"/(?:uncscode|Gorkowski)/particula/(?:blob|tree)/main/(.+)",
            parsed.path,
        )
        if parsed.netloc == "github.com" and match:
            return (root / unquote(match[1])).resolve(), ""
        return None
    if not parsed.path:
        return source, unquote(parsed.fragment)
    if parsed.path.startswith("/"):
        return (
            root / "docs" / unquote(parsed.path.lstrip("/"))
        ).resolve(), unquote(parsed.fragment)
    target = (source.parent / unquote(parsed.path)).resolve()
    if target.is_dir():
        target /= "index.md"
    return target, unquote(parsed.fragment)


def published_path(source):
    """Map a repository docs path to its MkDocs directory URL."""
    path = Path(source).relative_to("docs")
    if path.name == "index.md":
        return path.parent / "index.html"
    return path.with_suffix("") / "index.html"


def check(root, site=None):
    """Return actionable errors for the complete authored documentation tree."""
    root = root.resolve()
    docs = root / "docs"
    # BaseLoader constructs strings/lists/maps, never Python objects.
    config = yaml.load(
        (root / "mkdocs.yml").read_text(),
        Loader=yaml.BaseLoader,  # noqa: S506 -- non-object loader
    )
    errors = []
    nav = config["nav"]
    if tuple(next(iter(item)) for item in nav) != SECTIONS:
        errors.append(
            "navigation must have Guide, Examples, Theory, API, Contribute"
        )
    manifest = json.loads((docs / ".assets/relocations.json").read_text())
    pages = {
        p.resolve()
        for p in docs.rglob("*")
        if p.suffix in {".md", ".ipynb"}
        and not any(x.startswith(".") for x in p.relative_to(docs).parts)
    }
    errors.extend(
        f"unclassified page: {p.relative_to(root)}"
        for p in pages
        if p != docs / "index.md" and p.relative_to(docs).parts[0] not in OWNERS
    )
    parsed = {}
    sources = {
        p.resolve()
        for p in docs.rglob("*.py")
        if not any(x.startswith(".") for x in p.relative_to(docs).parts)
    }
    for path in sorted(pages | sources):
        parsed[path] = Links(
            markdown.markdown(
                source_text(path),
                extensions=["extra", "toc", "pymdownx.superfences"],
            )
        )
    graph = {p: set() for p in pages}
    reachable = {docs / "index.md"}
    navigation_pages, navigation_errors = check_navigation(nav, docs, pages)
    reachable.update(navigation_pages)
    errors.extend(navigation_errors)
    graph, linked_sources, link_errors = check_links(
        root, pages, parsed, manifest
    )
    errors.extend(link_errors)
    pending = list(reachable)
    while pending:
        for target in graph.get(pending.pop(), set()) - reachable:
            reachable.add(target)
            pending.append(target)
    for path in sorted(pages - reachable):
        errors.append(f"unreachable authored page: {path.relative_to(root)}")
    errors.extend(check_pairs(root, pages, linked_sources))
    if site is not None:
        errors.extend(check_published(root, site.resolve(), pages, manifest))
        errors.extend(check_rendered_links(root, site.resolve(), pages))
    return errors


def check_navigation(nav, docs, pages):
    """Require valid navigation with each section landing first."""
    errors = []
    reachable = set()
    for item in nav:
        title = next(iter(item))
        first = next(nav_targets([item]), None)
        if title != "API" and first != title.lower() + "/index.md":
            errors.append(f"section landing must be first: {title}")
    for target in nav_targets(nav):
        if urlsplit(target).scheme:
            continue
        path = (docs / target).resolve()
        if path not in pages:
            errors.append(f"missing navigation destination: {target}")
        else:
            reachable.add(path)
    return reachable, errors


def check_pairs(root, pages, linked_sources):
    """Require source/notebook parity and discoverable standalone examples."""
    errors = []
    docs = root / "docs"
    for path in sorted(pages):
        if path.suffix != ".ipynb":
            continue
        script = path.with_suffix(".py")
        if not script.exists():
            if path.is_relative_to(docs / "examples"):
                errors.append(
                    f"missing notebook pair: {script.relative_to(root)}"
                )
            continue  # scientific Theory notebooks may be notebook-only
        linked_sources.add(script)
        notebook = json.loads(path.read_text())
        code = "\n".join(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code"
        )
        if ast.dump(ast.parse(code)) != ast.dump(ast.parse(script.read_text())):
            errors.append(
                f"notebook/source code differs: {path.relative_to(root)}"
            )
    for script in sorted((docs / "examples").rglob("*.py")):
        if script not in linked_sources:
            errors.append(
                f"unlinked standalone source: {script.relative_to(root)}"
            )
    return errors


def check_links(root, pages, parsed, manifest):
    """Validate prose targets and build the page reachability graph."""
    errors = []
    graph = {p: set() for p in pages}
    linked_sources = set()
    for path, content in parsed.items():
        for url in content.urls:
            result = local_target(url, path, root)
            if result is None:
                continue
            target, anchor = result
            if not target.is_relative_to(root):
                errors.append(
                    f"{path.relative_to(root)}: link escapes repository: {url}"
                )
                continue
            relative = target.relative_to(root).as_posix()
            if relative in manifest["moves"] or relative in manifest.get(
                "retired", {}
            ):
                errors.append(
                    f"{path.relative_to(root)}: stale moved link: {url}"
                )
            if not target.exists():
                errors.append(
                    f"{path.relative_to(root)}: missing local link: {url}"
                )
            elif target in pages:
                graph.setdefault(path, set()).add(target)
                if anchor and anchor not in parsed[target].ids:
                    errors.append(
                        f"{path.relative_to(root)}: missing anchor: {url}"
                    )
            elif target.suffix == ".py":
                linked_sources.add(target)
    return graph, linked_sources, errors


def check_published(root, site, pages, manifest):
    """Verify rendered pages, API, and all retained path/fragment redirects."""
    errors = []
    for source in [
        *(p.relative_to(root).as_posix() for p in pages),
        "docs/API/index.md",
    ]:
        if not (site / published_path(source)).is_file():
            errors.append(f"missing published page: {source}")
    for old, new in manifest["moves"].items():
        if not old.endswith((".md", ".ipynb")) or not new.startswith("docs/"):
            continue
        redirect = site / published_path(old)
        target = site / published_path(new)
        if not redirect.is_file() or not target.is_file():
            errors.append(
                f"missing published redirect/destination: {old} -> {new}"
            )
        elif 'name="robots" content="noindex"' not in redirect.read_text():
            errors.append(f"redirect is searchable: {old}")
    for old, destination in manifest["fragments"].items():
        source, anchor = destination.split("#", 1)
        target = site / published_path(source)
        if not target.is_file() or anchor not in Links(target.read_text()).ids:
            errors.append(f"missing redirect fragment: {old} -> {destination}")
    errors.extend(check_search(site, manifest))
    return errors


def check_search(site, manifest):
    """Keep retired routes and trash out of the public search index."""
    errors = []
    search = json.loads((site / "search/search_index.json").read_text())
    retired_routes = {
        str(published_path(p).parent) + "/"
        for p in manifest.get("retired", {})
        if p.endswith((".md", ".ipynb"))
    }
    for row in search["docs"]:
        if (
            row["location"].split("#", 1)[0] in retired_routes
            or ".trash" in row["location"]
        ):
            errors.append(f"retired source in search: {row['location']}")
    return errors


def rendered_target(site, page, path, mount):
    """Resolve relative and project-site absolute URLs to built files."""
    if not path:
        return page
    if path.startswith("/"):
        path = path.removeprefix(mount + "/").lstrip("/")
        target = (site / unquote(path)).resolve()
    else:
        target = (page.parent / unquote(path)).resolve()
    return target / "index.html" if target.is_dir() else target


def check_rendered_links(root, site, pages):
    """Catch plugin-produced broken URLs, including notebook Markdown links."""
    errors = []
    cache = {}
    config = yaml.load(
        (root / "mkdocs.yml").read_text(),
        Loader=yaml.BaseLoader,  # noqa: S506 -- non-object loader
    )
    mount = urlsplit(config.get("site_url", "")).path.rstrip("/")
    for source in sorted(pages):
        page = site / published_path(source.relative_to(root).as_posix())
        if not page.exists():
            continue  # missing pages are reported by check_published
        content = Links(page.read_text())
        for link in content.urls:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = rendered_target(site, page, url.path, mount)
            if not target.is_relative_to(site) or not target.is_file():
                errors.append(
                    f"{source.relative_to(root)}: broken rendered link: {link}"
                )
            elif url.fragment and target.suffix == ".html":
                if target not in cache:
                    cache[target] = Links(target.read_text()).ids
                if unquote(url.fragment) not in cache[target]:
                    errors.append(
                        f"{source.relative_to(root)}: "
                        f"broken rendered anchor: {link}"
                    )
    return errors


def main():
    """Run the structural check with an optional rendered publication tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path)
    args = parser.parse_args()
    errors = check(Path(__file__).resolve().parents[1], args.site_dir)
    for error in errors:
        print(error)
    print(f"Documentation structure: {len(errors)} error(s)")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
