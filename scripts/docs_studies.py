"""Resolve study homes, collections, executions and precise cross-references."""

import argparse
import html
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

CATEGORIES = {
    "examples": "Worked examples",
    "tutorials": "Tutorials",
    "literature": "Literature analyses",
    "validation": "Validation studies",
    "benchmarks": "Benchmarks",
}
STUDIES = "docs/studies"
LITERATURE = f"{STUDIES}/literature"
SCRIPT_INDEX = f"{STUDIES}/scripts.md"


def route(source):
    """Return the site route for an authored page."""
    path = Path(str(source).removeprefix("docs/"))
    path = path.parent if path.name == "index.md" else path.with_suffix("")
    return "" if str(path) == "." else path.as_posix()


def download_links(key, has_inputs, base):
    """Offer downloads that also save plain-text Python files on click."""
    files = [(f"{key}.py", "Download Python script")]
    if has_inputs:
        files.append((f"{key}.zip", "Download complete ZIP"))
    return " · ".join(
        f'<a href="{base}downloads/{html.escape(name)}" '
        f'download="{html.escape(name)}">{label}</a>'
        for name, label in files
    )


def api_route(symbol, root):
    """Find the module page exposing a public symbol or class member."""
    parts = symbol.split(".")
    if parts[0] != "particula" or any(not p.isidentifier() for p in parts):
        raise ValueError(f"Invalid API symbol: {symbol}")
    for length in range(len(parts) - 1, 0, -1):
        module = Path(*parts[:length])
        if (root / module.with_suffix(".py")).is_file() or (
            root / module / "__init__.py"
        ).is_file():
            return f"API/{module.as_posix()}/#{symbol}"
    raise ValueError(f"Unknown API module for {symbol}")


class StudyCatalog:
    """Resolved study resources, collections, and executions."""

    def __init__(self, graph, runs, root):
        """Index the current graph against sources in the checkout."""
        self.root = root
        self.runs = runs
        self.studies = [r for r in graph["resources"] if "study" in r]
        self.titles = {r["source"]: r["title"] for r in runs.values()}
        self.titles.update(
            {r["source"]: r["title"] for r in graph["resources"]}
        )
        self.topics = {t["id"]: t["label"] for t in graph["topics"]}
        self.relations = defaultdict(list)
        self.backlinks = defaultdict(list)
        self.rows = []
        self.errors = []
        self.exceptions = graph.get("study_group_exceptions", {})
        self.by_collection = defaultdict(list)
        self.secondary = defaultdict(list)
        self.collections = {}
        self.study_by_source = {r["source"]: r["study"] for r in self.studies}
        self._collect()
        self._resolve()
        self.literature_groups = self._literature_groups()
        self._group()

    @classmethod
    def load(cls, root, graph):
        """Read execution declarations once alongside the topic graph."""
        runs = json.loads((root / "website/examples.json").read_text())
        return cls(graph, runs, root)

    def title(self, source):
        """Use registered titles and fall back to authored Markdown headings."""
        if source not in self.titles:
            path = self.root / source
            heading = (
                re.search(r"^# (.+)$", path.read_text(), re.M)
                if path.is_file()
                else None
            )
            self.titles[source] = heading[1] if heading else path.stem
        return self.titles[source]

    def _collect(self):
        owners = {r["source"] for r in self.studies}
        for path in (self.root / STUDIES).rglob("index.md"):
            source = path.relative_to(self.root).as_posix()
            if source not in owners:
                folder = path.parent.relative_to(self.root).as_posix()
                self.collections[folder] = source
        for folder, label in CATEGORIES.items():
            self.titles[f"{STUDIES}/{folder}/index.md"] = label
        self.titles[SCRIPT_INDEX] = "All runnable Python scripts"

    def _resolve(self):
        identifiers = Counter()
        executions = Counter()
        for resource in self.studies:
            study = resource["study"]
            source = resource["source"]
            identifiers[study["id"]] += 1
            parts = Path(source).parts
            if (
                len(parts) < 4
                or parts[:2] != ("docs", "studies")
                or parts[2] not in CATEGORIES
            ):
                self.errors.append(f"{source}: invalid canonical study home")
                continue
            self._add_study(resource, parts[2], executions)
        for key, run in self.runs.items():
            if not run["source"].startswith(STUDIES + "/"):
                self.errors.append(f"{key}: script source must live in Studies")
            if executions[key] != 1:
                self.errors.append(
                    f"{key}: expected one owning study, found {executions[key]}"
                )
        for key, count in identifiers.items():
            if not key or count != 1:
                self.errors.append(
                    f"invalid or duplicate study identifier: {key}"
                )
        self.rows.sort(key=lambda row: (row["title"].casefold(), row["key"]))

    def _add_study(self, resource, category, executions):
        source = resource["source"]
        study = resource["study"]
        secondary = study.get("secondary_category")
        if secondary:
            if secondary not in CATEGORIES or secondary == category:
                self.errors.append(
                    f"{source}: invalid secondary category {secondary}"
                )
            else:
                self.secondary[f"{STUDIES}/{secondary}"].append(source)
        executions.update(study.get("runs", []))
        self._add_runs(resource, category, secondary)
        for target in study.get("companions", []):
            self._link(source, target, self.title(target))
        for relation in study.get("relations", []):
            self._link(
                source,
                relation["target"],
                relation["label"],
                relation.get("fragment"),
            )

    def _add_runs(self, resource, category, secondary):
        source = resource["source"]
        study = resource["study"]
        for key in study.get("runs", []):
            run = self.runs.get(key)
            if not run:
                self.errors.append(f"{source}: unknown execution {key}")
                continue
            target = run["source"]
            topics = [resource["topic"], *resource["also"]]
            self.rows.append(
                {
                    "key": key,
                    "source": target,
                    "title": run["title"],
                    "study": source,
                    "category": category,
                    "secondary": secondary,
                    "device": run["device"],
                    "topics": topics,
                }
            )
            if target != source:
                self._link(source, target, f"Calculation · {run['title']}")

    def _link(self, source, target, label, fragment=None):
        if target.startswith("api:"):
            symbol = target.removeprefix("api:")
            try:
                url = api_route(symbol, self.root)
                label = f"{label} · `{symbol}`"
            except ValueError as err:
                self.errors.append(str(err))
                return
        else:
            if (
                not target.startswith("docs/")
                or not (self.root / target).is_file()
            ):
                self.errors.append(f"{source}: missing study target {target}")
                return
            url = route(target) + "/" + (f"#{fragment}" if fragment else "")
            if label != self.title(target) and not label.startswith(
                "Calculation · "
            ):
                label += f" · {self.title(target)}"
        link = (label, url)
        if link not in self.relations[source]:
            self.relations[source].append(link)
        reverse = (self.title(source), route(source) + "/")
        if reverse not in self.backlinks[target]:
            self.backlinks[target].append(reverse)

    def _literature_groups(self):
        """Read the gallery's ordered headings and illustrated study links."""
        gallery = self.root / LITERATURE / "index.md"
        expected = {
            r["source"]
            for r in self.studies
            if r["source"].startswith(LITERATURE + "/")
            or r["study"].get("secondary_category") == "literature"
        }
        if not gallery.is_file():
            return []
        by_route = {route(source): source for source in expected}
        content = gallery.read_text().split("<!-- STUDY_LIST_START -->")[0]
        sections = re.split(r"^## (.+)$", content, flags=re.M)
        groups = []
        seen = set()
        for label, section in zip(sections[1::2], sections[2::2], strict=True):
            cards = re.search(
                r'<div class="study-figures">(.*?)</div>', section, flags=re.S
            )
            if not cards:
                continue
            sources = []
            for card in cards[1].strip().split("\n\n"):
                link = re.search(r"\]\(([^()\s]+)\)\s*$", card)
                if not card.startswith("[![") or not link:
                    self.errors.append(f"{label}: malformed literature card")
                    continue
                url = urlsplit(link[1])
                if url.netloc == "uncscode.github.io" and url.path.startswith(
                    "/particula/"
                ):
                    slug = url.path.removeprefix("/particula/").rstrip("/")
                elif not url.netloc and not url.scheme:
                    target = os.path.normpath(
                        os.path.join(LITERATURE, url.path)
                    )
                    slug = route(target)
                else:
                    slug = ""
                source = by_route.get(slug)
                if not source or source in seen:
                    self.errors.append(
                        f"{label}: unknown or duplicate literature study "
                        f"{link[1]}"
                    )
                    continue
                sources.append(source)
                seen.add(source)
            self._check_choices(
                f"{LITERATURE}#{self._gallery_anchor(label)}",
                [self.title(s) for s in sources],
            )
            groups.append((label, sources))
        for source in sorted(expected - seen):
            self.errors.append(f"{source}: missing literature gallery card")
        return groups

    def _group(self):
        orders = {}
        for resource in self.studies:
            source = resource["source"]
            parent = Path(source).parent
            while parent.as_posix() not in self.collections and parent.parts:
                parent = parent.parent
            if (
                parent.as_posix() not in self.collections
                or parent.as_posix() == STUDIES
            ):
                self.errors.append(f"{source}: missing study category landing")
                continue
            self.by_collection[parent.as_posix()].append(source)
            orders[source] = resource["study"].get("order", 1000)
        for folder, source in self.collections.items():
            if folder == STUDIES:
                continue
            parent = Path(folder).parent.as_posix()
            if parent not in self.collections:
                self.errors.append(
                    f"{folder}: missing parent collection {parent}"
                )
            self.by_collection[parent].append(source)
        for folder in self.collections:
            children = self.by_collection[folder]
            children.sort(
                key=lambda s: (orders.get(s, 1000), self.title(s).casefold())
            )
            if folder == STUDIES:
                category_order = [f"{STUDIES}/{c}/index.md" for c in CATEGORIES]
                children.sort(
                    key=lambda s: category_order.index(s)
                    if s in category_order
                    else 100
                )
                children.insert(0, SCRIPT_INDEX)
            if folder == LITERATURE:
                self._check_choices(
                    folder, [g[0] for g in self.literature_groups]
                )
            else:
                self._check_choices(
                    folder, [self.title(s) for s in self.children(folder)]
                )

    def _check_choices(self, folder, titles):
        if len(titles) > 6 and not self.exceptions.get(folder):
            self.errors.append(
                f"{folder}: review {len(titles)} immediate choices: "
                + ", ".join(titles)
            )

    @staticmethod
    def _gallery_anchor(label):
        return re.sub(r"[^\w -]", "", label.lower()).replace(" ", "-")

    def children(self, folder):
        """List primary children and secondary entries in this directory."""
        return self.by_collection[folder] + sorted(
            self.secondary[folder], key=self.title
        )

    def navigation(self, folder=STUDIES):
        """Build the sidebar from primary study homes."""
        if folder == LITERATURE:
            groups = []
            for label, sources in self.literature_groups:
                items = []
                for source in sources:
                    item = self._menu_entry(source)
                    if source in self.secondary[folder]:
                        item["secondary"] = True
                    items.append(item)
                groups.append(
                    {
                        "label": label,
                        "slug": route(folder),
                        "anchor": self._gallery_anchor(label),
                        "items": items,
                    }
                )
            return groups
        return [
            self._menu_entry(source) for source in self.by_collection[folder]
        ]

    def _menu_entry(self, source):
        item = {"label": self.title(source), "slug": route(source)}
        if source in self.collections.values():
            item["items"] = self.navigation(str(Path(source).parent))
        elif study := self.study_by_source.get(source):
            # A companion can live outside its owning report's directory.
            owned = list(study.get("companions", []))
            owned.extend(
                self.runs[key]["source"]
                for key in study.get("runs", [])
                if key in self.runs
            )
            related = sorted({route(s) for s in owned if s != source})
            if related:
                item["ownedSlugs"] = related
        return item

    def breadcrumb(self, source, base):
        """Link directly to parent collections and the owning study."""
        if not source.startswith(STUDIES + "/"):
            return ""
        ancestors = [
            self.collections[p.as_posix()]
            for p in reversed(Path(source).parents)
            if p.as_posix() in self.collections
            and self.collections[p.as_posix()] != source
        ]
        parents = [
            r["source"]
            for r in self.studies
            if r["source"] != source
            and (
                source in r["study"].get("companions", [])
                or any(
                    row["source"] == source and row["study"] == r["source"]
                    for row in self.rows
                )
            )
        ]
        if len(parents) == 1:
            ancestors.append(parents[0])
        if not ancestors:
            return ""
        links = [
            f'<a href="{base}{route(page)}/">'
            f"{html.escape(self.title(page))}</a>"
            for page in ancestors
        ]
        return (
            '<nav class="study-path" aria-label="Study path">'
            + (' <span aria-hidden="true">/</span> '.join(links))
            + "</nav>\n\n"
        )

    def links(self, source, base):
        """Group companion calculations and downloads in the end matter."""
        downloads_by_link = {}
        for row in self.rows:
            if row["study"] != source or row["source"] == source:
                continue
            key = row["key"]
            url = route(row["source"]) + "/"
            link = (f"Calculation · {row['title']}", url)
            downloads = download_links(
                key, bool(self.runs[key].get("inputs")), base
            )
            downloads_by_link[link] = (
                f" · [View code]({base}{url}#python-source) · {downloads}"
            )

        resources = [
            f"- [{label}]({base}{url})"
            + downloads_by_link.get((label, url), "")
            for label, url in self.relations[source]
        ]
        related = [
            f"- [{label}]({base}{url})" for label, url in self.backlinks[source]
        ]
        sections = []
        for heading, items in [
            ("Study resources", resources),
            ("Related studies", related),
        ]:
            if items:
                sections.append(f"## {heading}\n\n" + "\n".join(items))
        return "\n\n" + "\n\n".join(sections) if sections else ""


def _portable_listing(catalog, source, links):
    """Use gallery group order for the portable literature directory."""

    def link(title, target):
        return f"[{title}]({os.path.relpath(target, Path(source).parent)})"

    if source == f"{LITERATURE}/index.md":
        groups = []
        for label, sources in catalog.literature_groups:
            items = [f"- {link(catalog.title(s), s)}" for s in sources]
            groups.append(f"**{label}**\n\n" + "\n".join(items))
        collections = [
            link(title, target)
            for title, target in links
            if target in catalog.collections.values()
        ]
        if collections:
            groups.append("Related collections: " + " · ".join(collections))
        return "\n\n".join(groups)
    return "\n".join(f"- {link(title, target)}" for title, target in links)


def offline_indexes(catalog, *, write=False):
    """Generate or verify the portable Markdown directories in the bundle."""
    errors = []
    entries = {
        source: [(catalog.title(s), s) for s in catalog.children(folder)]
        for folder, source in catalog.collections.items()
    }
    entries[SCRIPT_INDEX] = [(r["title"], r["source"]) for r in catalog.rows]
    start = "<!-- STUDY_LIST_START -->"
    end = "<!-- STUDY_LIST_END -->"
    for source, links in entries.items():
        path = catalog.root / source
        text = path.read_text() if path.is_file() else ""
        if text.count(start) != 1 or text.count(end) != 1:
            errors.append(f"{source}: missing unique offline listing markers")
            continue

        listing = _portable_listing(catalog, source, links)
        before, after = text.split(start)
        _, after = after.split(end)
        current = f"{before}{start}\n{listing}\n{end}{after}"
        if current != text:
            if write:
                path.write_text(current)
            else:
                errors.append(
                    f"{source}: refresh with scripts/docs_studies.py sync"
                )
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "sync"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    graph = json.loads((root / "website/topics.json").read_text())
    catalog = StudyCatalog.load(root, graph)
    errors = catalog.errors or offline_indexes(
        catalog, write=args.command == "sync"
    )
    if errors:
        raise SystemExit("\n".join(errors))
