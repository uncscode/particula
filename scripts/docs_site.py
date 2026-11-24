"""Prepare and verify the Starlight documentation without running examples."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import posixpath
import re
import shutil
import subprocess
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import docs_studies
import docs_topics
import yaml
from docs_examples import (
    ROOT,
    bundle,
    registry,
    result_state,
    sections,
)
from study_downloads import write_study_zip

SITE = ROOT / "website"
CONTENT = SITE / "src/content/docs"
BASE = "/particula/"
PUBLISHED = "https://uncscode.github.io/particula/"
PAGES = sorted(
    str(p.relative_to(ROOT / "docs"))
    for section in ("guide", "studies", "theory", "contribute")
    for p in (ROOT / "docs" / section).rglob("*.md")
)


def route(path):
    """Map a source path to a stable directory route."""
    return docs_studies.route(path)


def local_routes():
    """Return routes present in the complete documentation site."""
    return (
        {route(p) for p in PAGES}
        | {route(c["source"]) for c in registry().values()}
        | {"", "API", "api"}
        | {
            docs_topics.topic_route(t["id"])
            for t in docs_topics.load()["topics"]
        }
    )


def rewrite_links(text, source):
    """Translate authored relative links outside code into local/public URLs."""

    def replace(match):
        image, label, href = match.groups()
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or href.startswith("#"):
            return match.group(0)
        target = posixpath.normpath(
            posixpath.join(str(Path(source).parent), unquote(parts.path))
        )
        if href.startswith("/"):
            target = "docs/" + parts.path.lstrip("/")
        fragment = "#" + parts.fragment if parts.fragment else ""
        if target.startswith("docs/images/"):
            destination = BASE + target.removeprefix("docs/")
        elif target.startswith(
            ("website/public/study-data/", "website/results/")
        ):
            asset = target.removeprefix("website/").removeprefix("public/")
            destination = BASE + asset
        elif Path(target).suffix == ".py":
            ids = {c["source"]: key for key, c in registry().items()}
            destination = (
                BASE + route(target) + "/"
                if target in ids
                else "https://github.com/uncscode/particula/blob/main/" + target
            )
        elif target.startswith("docs/"):
            page = route(target)
            if page in local_routes() or page.startswith("API/particula"):
                destination = BASE + (page + "/" if page else "")
            else:
                destination = BASE + page + "/"
        else:
            destination = (
                "https://github.com/uncscode/particula/blob/main/" + target
            )
        return f"{image}[{label}]({destination}{fragment})"

    parts = re.split(r"(^```[^\n]*\n.*?^```[^\n]*$)", text, flags=re.M | re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(!?)\[([^\]\n]+)\]\(([^)\s]+)\)", replace, parts[i])
    return "".join(parts)


def write_page(path, title, body, source=None, extra=None):
    """Write a generated page with valid metadata and source attribution."""
    destination = CONTENT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"title": title}
    if str(path) != "index.mdx":
        metadata["slug"] = route(path)
    if source:
        metadata["editUrl"] = (
            "https://github.com/uncscode/particula/edit/main/" + source
        )
    if extra:
        metadata.update(extra)
    if source and str(path) != "index.mdx":
        body += docs_topics.page_links(source, docs_topics.load(), BASE)
    destination.write_text(
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False)
        + "---\n\n"
        + body
        + "\n"
    )


def output_markdown(outputs, url, title):
    """Render captured scientific output, keeping plots outside code folds."""
    result = []
    progress = []
    progress_bar = re.compile(r"\d+(?:\.\d+)?%\|.*?\|\s*[\d,.]+/[\d,.]+")
    for output in outputs:
        if "image" in output:
            result.append(
                '<figure class="scientific-figure">'
                f'<img src="{url}/{output["image"]}" '
                f'alt="{html.escape(title)}: saved scientific figure" '
                'loading="lazy" />'
                f"<figcaption>{html.escape(title)} · "
                "saved execution output</figcaption></figure>"
            )
        elif "html" in output:
            result.append(
                '<div class="result-table">\n\n' + output["html"] + "\n\n</div>"
            )
        elif output.get("text", "").strip():
            # Keep scientific text visible; collect terminal progress updates.
            visible = []
            for line in output["text"].splitlines():
                clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
                if progress_bar.search(clean):
                    progress.append(clean.strip())
                elif clean.strip():
                    visible.append(clean)
            if visible:
                result.append(
                    '<pre class="result-output" data-pagefind-ignore><code>'
                    + html.escape("\n".join(visible))
                    + "</code></pre>"
                )
    if progress:
        result.append(
            '<details class="execution-progress" data-pagefind-ignore>'
            "<summary>Execution progress</summary>"
            '<pre class="result-output"><code>'
            + html.escape("\n".join(progress))
            + "</code></pre></details>"
        )
    return "\n\n".join(result)


def example_downloads(key, config, source):
    """Publish source and a runnable bundle when helpers are needed."""
    download = SITE / "public/downloads" / f"{key}.py"
    download.parent.mkdir(parents=True, exist_ok=True)
    download.write_text(source)
    if config.get("inputs"):
        archive = download.with_suffix(".zip")
        write_study_zip(archive, ROOT, config, source)
    return docs_studies.download_links(key, bool(config.get("inputs")), BASE)


def result_notice(state, manifest):
    """Describe the evidence behind the displayed figures."""
    if state != "current":
        return "Results from an earlier run—refresh needed", "caution"
    if manifest.get("historical_run"):
        return "Historical simulation, updated figures", "caution"
    return "Matching saved results", "note"


def render_credit(manifest):
    """Describe the source of a historical run or saved-data redraw."""
    credit = ""
    if "plot_source_revision" in manifest:
        credit = (
            "This native execution used the archived scientific package "
            "and the example and plotting code from "
            f"revision `{manifest['plot_source_revision'][:9]}`.\n\n"
        )
    if "rendered_at" in manifest:
        credit += (
            f"Plots rendered from saved data on {manifest['rendered_at'][:10]} "
            f"UTC (revision `{manifest['render_revision'][:9]}`); "
            "simulation not rerun.\n\n"
        )
    return credit


def render_example(key, config, study_links="", breadcrumb=""):
    """Render current source and saved output without execution."""
    directory, manifest = bundle(key)
    state = result_state(config, manifest)
    source = (ROOT / config["source"]).read_text()
    chunks = [
        breadcrumb
        + "[View code](#python-source) · "
        + example_downloads(key, config, source)
    ]
    url = BASE + f"results/{key}"
    if directory:
        destination = SITE / "public/results" / key
        shutil.copytree(directory, destination, dirs_exist_ok=True)
        status, kind = result_notice(state, manifest)
        executed_source = (
            "executed_source.py" if "rendered_at" in manifest else "source.py"
        )
        rendering = render_credit(manifest)
        data_link = (
            f"[Numerical plot data]({url}/data.json).\n\n"
            if "data.json" in manifest.get("files", {})
            else ""
        )
        runtime = (
            f" · Script runtime: **{manifest['elapsed_seconds']:,.2f} s**"
            if "elapsed_seconds" in manifest
            else ""
        )
        chunks.append(
            f":::{kind}[{status}]\nExecution: **{manifest['device']}** · "
            f"{manifest['executed_at'][:10]} UTC{runtime}. "
            f"Particula revision `{manifest['revision'][:9]}`. "
            f"[Executed source]({url}/{executed_source}) · "
            f"[Run metadata]({url}/manifest.json).\n\n"
            + rendering
            + data_link
            + "A matching script fingerprint is not a claim of validation "
            "against every later library revision.\n:::"
        )
    else:
        chunks.append(
            ":::note[Awaiting saved results]\nNo saved execution is available "
            f"for this example. Its **{config['policy']}** execution is "
            "explicit; building this page does not run it.\n:::"
        )
    code_index = 0
    code_anchor = '<span id="python-source"></span>\n\n'
    invocation_index = len(chunks)
    for cell in sections(source):
        if cell.cell_type == "markdown":
            prose = re.sub(r"^# (.+)$", r"## \1", cell.source, flags=re.M)
            chunks.append(rewrite_links(prose, config["source"]))
        elif cell.cell_type == "code" and cell.source.strip():
            chunks.append(
                "<details>\n<summary>Python · "
                f"section {code_index + 1}</summary>\n\n"
                f"{code_anchor}```python\n{cell.source}\n```\n\n</details>"
            )
            code_anchor = ""
            if state == "current" and code_index < len(manifest["outputs"]):
                chunks.append(
                    output_markdown(
                        manifest["outputs"][code_index], url, config["title"]
                    )
                )
            code_index += 1
            invocation_index = len(chunks)
        elif cell.cell_type == "code":
            code_index += 1
            invocation_index = len(chunks)
    if config.get("invoke"):
        invocation = [
            f"<details>\n<summary>Explicit {config['device']} "
            "invocation</summary>\n\n"
            f"```python\n{config['invoke']}\n```\n\n</details>"
        ]
        if state == "current":
            invocation.append(
                output_markdown(manifest["outputs"][-1], url, config["title"])
            )
        # Execution appends this call after all code cells. Show its evidence
        # there too, before the author's trailing interpretation and closure.
        chunks[invocation_index:invocation_index] = invocation
    if state == "stale":
        chunks.append(
            "## Previous run outputs\n\nThese outputs belong to the executed "
            "source linked above, not the updated code sections."
        )
        chunks.extend(
            output_markdown(o, url, config["title"])
            for o in manifest["outputs"]
        )
    chunks.append(study_links)
    write_page(
        route(config["source"]) + ".md",
        config["title"],
        "\n\n".join(chunks),
        config["source"],
    )


def legacy_api_pages():
    """Load symbol anchors recorded from the pre-migration publication."""
    return json.loads((SITE / "legacy-api-anchors.json").read_text())["pages"]


def preserve_api_members(data, pages):
    """Retain documented concrete-module members in the docs-only model.

    Starlight uses __all__ as a documentation filter. The previous site also
    documented concrete-only members, so extend only the extracted model's
    exports with existing members recorded in the publication inventory.
    This does not alter package exports or invent removed symbols.
    """
    for page, anchors in pages.items():
        parts = page.removeprefix("API/").removesuffix("/index.html").split("/")
        node = data.get(parts[0], {})
        for part in parts[1:]:
            node = node.get("members", {}).get(part, {})
        if "exports" not in node or node["exports"] is None:
            continue
        prefix = ".".join(parts) + "."
        names = {
            anchor[len(prefix) :].split(".")[0].split("--")[0]
            for anchor in anchors
            if anchor.startswith(prefix)
        }
        node["exports"] = sorted(
            set(node["exports"]) | (names & node.get("members", {}).keys())
        )


def generate_api():
    """Cache Griffe data using the original generator's exclusions."""
    cache = SITE / ".cache"
    cache.mkdir(exist_ok=True)
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update((SITE / "legacy-api-anchors.json").read_bytes())
    for path in sorted((ROOT / "particula").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    digest.update(
        subprocess.check_output(  # noqa: S603 -- fixed interpreter and argv
            [sys.executable, "-m", "griffe", "--version"]
        )
    )
    stamp = cache / "api.sha256"
    if not stamp.exists() or stamp.read_text() != digest.hexdigest():
        result = subprocess.run(  # noqa: S603 -- fixed interpreter and argv
            [
                sys.executable,
                "-m",
                "griffe",
                "dump",
                "particula",
                "-s",
                str(ROOT),
                "-d",
                "google",
                "-f",
                "-r",
                "--no-resolve-external",
                "-X",
                "-o",
                str(cache / "griffe.json"),
            ],
            capture_output=True,
            text=True,
        )
        (cache / "griffe.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        data = json.loads((cache / "griffe.json").read_text())

        def prune(node):
            members = node.get("members", {})
            for name, child in list(members.items()):
                if child.get("kind") == "module" and (
                    "test" in name.lower() or name == "build"
                ):
                    del members[name]
                elif child.get("kind") == "module":
                    prune(child)

        prune(data["particula"])
        preserve_api_members(data, legacy_api_pages())
        (cache / "griffe.json").write_text(json.dumps(data))
        stamp.write_text(digest.hexdigest())
    print(
        "Griffe API extraction ready "
        "(warnings retained in website/.cache/griffe.log)"
    )


def write_catalog(catalog):
    """Save the resolved catalog for Astro components."""
    (SITE / ".cache").mkdir(exist_ok=True)
    api_backlinks = {}
    for target, links in catalog.backlinks.items():
        if target.startswith("api:"):
            symbol = target.removeprefix("api:")
            module = docs_studies.api_route(symbol, ROOT).split("#")[0]
            api_backlinks.setdefault(module, []).extend(
                {"symbol": symbol, "title": title, "url": url}
                for title, url in links
            )
    (SITE / ".cache/studies.json").write_text(
        json.dumps(
            {
                "rows": catalog.rows,
                "topics": catalog.topics,
                "categories": docs_studies.CATEGORIES,
                "navigation": catalog.navigation(),
                "apiBacklinks": api_backlinks,
            },
            ensure_ascii=False,
        )
    )


def prepare_article(page, catalog):
    """Render Markdown prose with its collection links and study resources."""
    source = ROOT / "docs" / page
    text = re.sub(r"\A---\n.*?\n---\n", "", source.read_text(), flags=re.S)
    heading = re.search(r"^# (.+)$", text, re.M)
    title = heading[1] if heading else source.stem.replace("-", " ").title()
    if heading:
        anchor = re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
        text = (
            text[: heading.start()]
            + f'<a id="{anchor}"></a>\n'
            + text[heading.end() :]
        )
    key = "docs/" + page
    if key in catalog.collections.values() or key == docs_studies.SCRIPT_INDEX:
        text = re.sub(
            r"<!-- STUDY_LIST_START -->.*?<!-- STUDY_LIST_END -->",
            "",
            text,
            flags=re.S,
        )
    text = rewrite_links(text, key).strip()
    if key == docs_studies.SCRIPT_INDEX:
        # The index contains ordinary links before filters initialize.
        text = (
            "import StudyCatalog from '@components/StudyCatalog.astro';\n\n"
            + text
            + "\n\n<StudyCatalog />"
        )
        page = str(Path(page).with_suffix(".mdx"))
    if key == "docs/studies/index.md":
        text = (
            "import StudyCarousel from '@components/StudyCarousel.astro';\n\n"
            + text
            + '\n\n<StudyCarousel label="Illustrated studies" />'
        )
        page = str(Path(page).with_suffix(".mdx"))
    # The literature gallery already links every study through a figure card.
    if (
        key in catalog.collections.values()
        and key != "docs/studies/literature/index.md"
    ):
        folder = str(Path(key).parent)
        text += "\n\n" + "\n".join(
            f"- [{catalog.title(child)}]({BASE}{route(child)}/)"
            for child in catalog.children(folder)
        )
    write_page(
        page,
        title,
        catalog.breadcrumb(key, BASE) + text + catalog.links(key, BASE),
        key,
    )


def prepare():
    """Prepare documentation pages from source files and saved outputs."""
    topic_graph = docs_topics.require_valid()
    catalog = docs_studies.StudyCatalog.load(ROOT, topic_graph)
    write_catalog(catalog)
    if CONTENT.exists():
        shutil.rmtree(CONTENT)
    CONTENT.mkdir(parents=True)
    for name in ("results", "downloads", "images"):
        target = SITE / "public" / name
        if target.exists():
            shutil.rmtree(target)
    shutil.copytree(ROOT / "docs/images", SITE / "public/images")
    for page in PAGES:
        if page == "guide/topics/index.md":
            continue  # The maintained offline entry has a generated explorer.
        prepare_article(page, catalog)
    for key, config in registry().items():
        render_example(
            key,
            config,
            catalog.links(config["source"], BASE),
            catalog.breadcrumb(config["source"], BASE),
        )
    # Replace the Material-specific home template with native Starlight.
    home = rewrite_links((ROOT / "docs/index.md").read_text(), "docs/index.md")
    write_page(
        "index.mdx",
        "Particula",
        home,
        "docs/index.md",
        {
            "template": "splash",
            "hero": {
                "title": "Small particles.\nBig questions.",
                "tagline": (
                    "Compare simulations with measurements to discover new "
                    "reactions, understand aerosol processes, and explore "
                    "their impacts."
                ),
                "actions": [
                    {
                        "text": "Get Particula Helper",
                        "link": BASE + "guide/agent-plugins/",
                        "attrs": {"data-plugin-icon": "particula-helper"},
                    },
                    {
                        "text": "Getting started",
                        "link": BASE
                        + "guide/getting-started/first-simulation/",
                        "variant": "minimal",
                    },
                ],
            },
        },
    )
    write_page(
        "api.md",
        "API reference",
        (SITE / "templates/api.md").read_text()
        + docs_topics.page_links("api", topic_graph, BASE),
    )
    write_page(
        "contribute/script-examples.md",
        "Author a Python example",
        (SITE / "AUTHORING.md").read_text(),
    )
    docs_topics.generate(write_page, topic_graph)
    generate_api()
    print(f"Prepared {len(list(CONTENT.rglob('*.*')))} authored/example pages")


class Links(HTMLParser):
    """Collect page links, local assets, and fragment anchors."""

    def __init__(self, text):
        """Parse one rendered HTML page."""
        super().__init__()
        self.links = []
        self.ids = set()
        self.is_redirect = False
        self.downloads = set()
        self.executions = []
        self.study_homes = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        """Collect only attributes relevant to link validation."""
        attrs = dict(attrs)
        if "data-particula-redirect" in attrs:
            self.is_redirect = True
        if "data-study-home" in attrs:
            self.study_homes.append(attrs["data-study-home"])
        if "data-execution" in attrs:
            self.executions.append(attrs["data-execution"])
        if tag == "a" and "download" in attrs:
            self.downloads.add(attrs.get("href"))
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag in {"a", "img", "script", "link"}:
            value = attrs.get("href" if tag in {"a", "link"} else "src")
            if value:
                self.links.append(value)


def redirects(dist):
    """Reuse applicable relocation records for retained destinations."""
    records = json.loads((ROOT / "docs/.assets/relocations.json").read_text())
    # Keep the original MkDocs API landing URL alongside the new overview.
    api_landing = dist / "API/index.html"
    api_landing.parent.mkdir(exist_ok=True)
    api_landing.write_text(
        "<!doctype html><html data-particula-redirect><head>"
        '<meta name="robots" content="noindex">'
        f'<meta http-equiv="refresh" content="0;url={BASE}api/">'
        "<title>API reference</title></head><body>"
        f'<a href="{BASE}api/">Open API reference</a></body></html>'
    )
    destinations = {}
    fragments = {}
    for old, new in records["moves"].items():
        if (
            old == new
            or not new.startswith("docs/")
            or not old.endswith((".md", ".py", ".ipynb"))
        ):
            continue
        target = route(new)
        if target not in local_routes():
            continue
        alias = route(old)
        if alias == target:
            continue
        if alias in destinations and destinations[alias] != target:
            raise ValueError(f"Conflicting redirect destinations for {alias}")
        destinations[alias] = target
    for key, value in records["fragments"].items():
        old, fragment = key.split("#", 1)
        page, anchor = value.split("#", 1)
        mapping = fragments.setdefault(route(old), {})
        destination = BASE + route(page) + "/#" + anchor
        if fragment in mapping and mapping[fragment] != destination:
            raise ValueError(f"Conflicting redirect fragments for {key}")
        mapping[fragment] = destination
    for alias, target in destinations.items():
        destination = BASE + target + "/"
        output = dist / alias / "index.html"
        if (
            output.exists()
            and "data-particula-redirect" not in output.read_text()
        ):
            raise ValueError(f"Redirect collision: {alias}")
        output.parent.mkdir(parents=True, exist_ok=True)
        mapping = fragments.get(alias, {})
        output.write_text(
            "<!doctype html><html data-particula-redirect><head>"
            '<meta name="robots" content="noindex">'
            "<title>Documentation moved</title>"
            f'<link rel="canonical" href="{destination}"></head><body>'
            f'<a href="{destination}">Open the current guide</a>'
            f"<script>const map={json.dumps(mapping)};"
            "location.replace(map[decodeURIComponent(location.hash.slice(1))]"
            f"||{json.dumps(destination)}+location.hash)</script></body></html>"
        )
    return 1 + len(destinations)


def check_rendered_links(dist, pages):
    """Check local link targets, assets, and fragments in built HTML."""
    errors = []
    for page, parsed in pages.items():
        for href in parsed.links:
            url = urlsplit(href)
            if (url.scheme or url.netloc) and not (
                url.netloc == urlsplit(PUBLISHED).netloc
                and url.path.startswith(BASE)
            ):
                continue
            if url.path.startswith("/"):
                if not url.path.startswith(BASE):
                    errors.append(
                        f"Wrong base path: {href} in {page.relative_to(dist)}"
                    )
                    continue
                target = dist / unquote(url.path.removeprefix(BASE))
            else:
                target = (
                    (page.parent / unquote(url.path)).resolve()
                    if url.path
                    else page
                )
            if target.is_dir():
                target /= "index.html"
            if not target.exists():
                errors.append(
                    f"Missing target {href} in {page.relative_to(dist)}"
                )
            elif (
                url.fragment
                and target.suffix == ".html"
                and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(
                    f"Missing anchor {href} in {page.relative_to(dist)}"
                )
    return errors


def check_study_catalog(dist, pages):
    """Count rendered catalog keys and canonical sidebar homes by identity."""
    errors = []
    catalog = docs_studies.StudyCatalog.load(ROOT, docs_topics.load(ROOT))
    home = dist / "studies/index.html"
    expected = [route(r["source"]) for r in catalog.studies]
    expected.append(route(docs_studies.SCRIPT_INDEX))
    if home not in pages or Counter(pages[home].study_homes) != Counter(
        expected
    ):
        errors.append("Each study must appear once in the primary Studies menu")
    script_index = dist / "studies/scripts/index.html"
    if script_index not in pages or Counter(
        pages[script_index].executions
    ) != Counter(list(registry())):
        errors.append(
            "Script catalog must contain every execution key exactly once"
        )
    return errors


def check_inventory(dist, pages):
    """Validate complete page, fragment, and source-download coverage."""
    errors = []
    for page_route in local_routes():
        if not (dist / page_route / "index.html").exists():
            errors.append(f"Missing inventoried page {page_route}")
    records = json.loads((ROOT / "docs/.assets/relocations.json").read_text())
    for old, destination in records["fragments"].items():
        source, anchor = destination.split("#", 1)
        target = dist / route(source) / "index.html"
        if target not in pages or anchor not in pages[target].ids:
            errors.append(f"Missing relocated anchor {old} -> {destination}")
    errors.extend(check_study_catalog(dist, pages))
    for key, config in registry().items():
        page = pages.get(dist / route(config["source"]) / "index.html")
        for suffix in ["py", "zip"] if config.get("inputs") else ["py"]:
            path = f"downloads/{key}.{suffix}"
            if not (dist / path).is_file():
                errors.append(f"Missing study download {path}")
            if not page or BASE + path not in page.downloads:
                errors.append(f"Missing download link for {path}")
    if any(".trash" in p.relative_to(dist).parts for p in dist.rglob("*")):
        errors.append("Archived files leaked into the published site")
    return errors


def check_llms_index(dist):
    """Check the published LLM index and its targets without network access."""
    index = dist / "llms.txt"
    if not index.is_file():
        return ["Missing published llms.txt"]
    text = index.read_text()
    errors = []
    if text != (SITE / "public/llms.txt").read_text():
        errors.append("Published llms.txt differs from its maintained source")
    links = re.findall(r"\[[^\]\n]+\]\(([^)\s]+)\)", text)
    if not text.startswith("# Particula\n") or not links:
        errors.append(
            "llms.txt needs a project heading and documentation links"
        )
    raw = "https://raw.githubusercontent.com/uncscode/particula/main/"
    for href in links:
        if href.startswith(raw):
            root, relative = ROOT, href.removeprefix(raw)
        elif href.startswith(PUBLISHED):
            root, relative = dist, href.removeprefix(PUBLISHED)
        else:
            errors.append(f"Unsupported llms.txt target or revision: {href}")
            continue
        target = (root / unquote(urlsplit(relative).path)).resolve()
        if not target.is_relative_to(root.resolve()):
            errors.append(f"llms.txt target escapes its root: {href}")
            continue
        if href.startswith(PUBLISHED) and target.is_dir():
            target /= "index.html"
        if ".trash" in target.parts or not target.is_file():
            errors.append(f"Missing or archived llms.txt target: {href}")
    return errors


def check_searchable_studies(urls, study_routes):
    """Each canonical study has exactly one actual Pagefind entry."""
    errors = []
    indexed = Counter(
        unquote(urlsplit(url).path).removeprefix(BASE).strip("/")
        for url in urls
    )
    for study in study_routes:
        if indexed[study] != 1:
            errors.append(f"Expected one searchable entry for {study}")
    return errors


def check_discovery(dist, pages, study_routes=()):
    """Inspect actual search fragments and sitemap URLs for retired content."""
    errors = []
    fragments = list((dist / "pagefind").rglob("*.pf_fragment"))
    if not fragments:
        errors.append("Missing Pagefind search fragments")
    urls = []
    for fragment in fragments:
        data = gzip.decompress(fragment.read_bytes())
        prefix = b"pagefind_dcd"
        if not data.startswith(prefix):
            errors.append(f"Unknown search fragment format: {fragment.name}")
            continue
        urls.append(json.loads(data[len(prefix) :])["url"])
    errors.extend(check_searchable_studies(urls, study_routes))
    for sitemap in dist.glob("sitemap-*.xml"):
        urls.extend(re.findall(r"<loc>(.*?)</loc>", sitemap.read_text()))
    for url in urls:
        path = unquote(urlsplit(url).path).removeprefix(BASE).lstrip("/")
        if path.endswith(".xml"):
            continue
        target = dist / path / "index.html"
        if ".trash" in Path(path).parts:
            errors.append(f"Archive indexed in search/sitemap: {url}")
        elif target not in pages:
            errors.append(f"Unknown page indexed in search/sitemap: {url}")
        elif pages[target].is_redirect:
            errors.append(f"Redirect indexed in search/sitemap: {url}")
    return sorted(set(errors))


def preserve_fragment_aliases(dist):
    """Keep unambiguous MkDocs heading IDs alongside Starlight heading IDs."""
    records = json.loads((ROOT / "docs/.assets/relocations.json").read_text())
    for destination in set(records["fragments"].values()):
        source, anchor = destination.split("#", 1)
        target = dist / route(source) / "index.html"
        if not target.exists():
            continue
        text = target.read_text()
        ids = Links(text).ids
        if anchor in ids:
            continue
        matches = [
            value
            for value in ids
            if re.sub("-+", "-", value).strip("-") == anchor
        ]
        if len(matches) == 1:
            pattern = (
                r'(<h[1-6]\b[^>]*\bid="' + re.escape(matches[0]) + r'"[^>]*>)'
            )
            text = re.sub(
                pattern,
                lambda m,
                anchor=anchor: f'<span id="{html.escape(anchor)}"></span>'
                + m[0],
                text,
                count=1,
            )
            target.write_text(text)


def preserve_api_anchors(dist, inventory):
    """Place old module and docstring anchors beside their current sections.

    Missing symbols are errors, never silently redirected to a page heading.
    MkDocs' nested docstring headings resolve to their owning symbol section.
    """
    errors = []
    for page, anchors in inventory.items():
        target = dist / page
        if not target.is_file():
            errors.append(f"Missing legacy API page {page}")
            continue
        text = target.read_text()
        ids = Links(text).ids
        module = page.removeprefix("API/").removesuffix("/index.html")
        module = module.replace("/", ".")
        for anchor in anchors:
            if anchor in ids:
                continue
            owner = anchor.split("--", 1)[0]
            if owner == module:
                pattern = r"(<h1\b[^>]*>)"
            elif "--" in anchor and owner in ids:
                pattern = r'(<[^>]+\bid="' + re.escape(owner) + r'"[^>]*>)'
            else:
                errors.append(f"Missing legacy API symbol {page}#{anchor}")
                continue
            text, count = re.subn(
                pattern,
                lambda match, anchor=anchor: (
                    f'<span id="{html.escape(anchor, quote=True)}"></span>'
                    + match[0]
                ),
                text,
                count=1,
            )
            if count:
                ids.add(anchor)
            else:
                errors.append(f"Missing API anchor target {page}#{anchor}")
        target.write_text(text)
    return errors


def verify():
    """Verify routes, links, API coverage, and generated assets."""
    dist = SITE / "dist"
    not_found = dist / "404.html"
    if not_found.exists():
        not_found.write_text(
            not_found.read_text().replace(
                PUBLISHED + "404/", PUBLISHED + "404.html"
            )
        )
    preserve_fragment_aliases(dist)
    api_anchors = legacy_api_pages()
    api_errors = preserve_api_anchors(dist, api_anchors)
    redirected = redirects(dist)
    pages = {p: Links(p.read_text()) for p in dist.rglob("*.html")}
    errors = check_rendered_links(dist, pages)
    errors.extend(api_errors)
    errors.extend(check_inventory(dist, pages))
    studies = (
        route(r["source"])
        for r in docs_topics.load()["resources"]
        if "study" in r
    )
    errors.extend(check_discovery(dist, pages, studies))
    errors.extend(check_llms_index(dist))
    data = json.loads((SITE / ".cache/griffe.json").read_text())
    modules = []

    def walk(node, path):
        modules.append(path)
        for name, child in node.get("members", {}).items():
            if child.get("kind") == "module":
                walk(child, path + "/" + name)

    walk(data["particula"], "API/particula")
    for module in modules:
        if not (dist / module / "index.html").exists():
            errors.append(f"Missing API module {module}")
    if not (dist / "pagefind/pagefind.js").exists():
        errors.append("Missing Pagefind search index")
    report = {
        "html_pages": len(pages),
        "api_modules": len(modules),
        "legacy_api_anchors": sum(map(len, api_anchors.values())),
        "redirects": redirected,
        "errors": sorted(set(errors)),
    }
    (SITE / ".cache/verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if errors:
        raise ValueError("\n".join(sorted(set(errors))[:60]))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify"])
    args = parser.parse_args()
    prepare() if args.command == "prepare" else verify()
