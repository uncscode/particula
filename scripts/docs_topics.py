"""Validate the curated topic graph and generate its Starlight entry pages."""

import json
import re
from collections import Counter
from pathlib import Path

import docs_studies
from render_literature_tocs import is_literature_study

ROOT = Path(__file__).resolve().parents[1]
ROLES = {"Guide", "Studies", "Theory", "API", "Contribute"}


def load(root=ROOT):
    """Read the single maintained graph, including page dispositions."""
    return json.loads((root / "website/topics.json").read_text())


def topic_route(identifier):
    """Keep root and topic addresses independent of mutable labels."""
    return "guide/topics" + (
        "" if identifier == "aerosols" else "/" + identifier
    )


def inventory(root):
    """Inventory all authored pages and registered executable sources."""
    docs = root / "docs"
    paths = {
        p.relative_to(root).as_posix()
        for p in docs.rglob("*")
        if p.suffix in {".md", ".py"}
        and not any(x.startswith(".") for x in p.relative_to(docs).parts)
    }
    examples = json.loads((root / "website/examples.json").read_text())
    return paths | {v["source"] for v in examples.values()}


def check_topic_fields(t, root):
    """Check labels, searchable terms, and the local illustration."""
    errors = []
    identifier = t["id"]
    if not re.fullmatch(r"[a-z][a-z0-9-]*", identifier):
        errors.append(f"invalid topic identifier: {identifier}")
    for key in ("label", "description", "icon", "context", "investigation"):
        if not isinstance(t.get(key), str) or not t[key].strip():
            errors.append(f"{identifier}: missing {key}")
    icon = root / "website/public/topic-icons" / (t["icon"] + ".svg")
    if not re.fullmatch(r"[a-z][a-z0-9-]*", t["icon"]) or not icon.is_file():
        errors.append(f"{identifier}: missing icon {t['icon']}")
    for key in ("aliases", "search_phrases", "prerequisites", "related"):
        if not isinstance(t[key], list) or any(
            not isinstance(v, str) or not v.strip() for v in t[key]
        ):
            errors.append(f"{identifier}: invalid {key}")
    return errors


def check_relationships(t, by_id):
    """Check relationship targets and reciprocal associations."""
    errors = []
    identifier = t["id"]
    if t["parent"] is not None and t["parent"] not in by_id:
        errors.append(f"{identifier}: missing parent {t['parent']}")
    for key in ("prerequisites", "related"):
        for other in t[key]:
            if other not in by_id or other == identifier:
                errors.append(f"{identifier}: invalid {key} target {other}")
            elif key == "related" and identifier not in by_id[other][key]:
                errors.append(f"{identifier}: related link is not reciprocal")
    return errors


def check_paths(by_id):
    """Require finite hierarchy paths and an acyclic prerequisite graph."""
    errors = []
    for identifier in by_id:
        visited = set()
        current = identifier
        while current in by_id:
            if current in visited:
                errors.append(f"{identifier}: hierarchy cycle")
                break
            visited.add(current)
            current = by_id[current]["parent"]

    def visit(identifier, active, done):
        if identifier in active:
            errors.append(f"{identifier}: prerequisite cycle")
            return
        if identifier in done or identifier not in by_id:
            return
        active.add(identifier)
        for other in by_id[identifier]["prerequisites"]:
            visit(other, active, done)
        active.remove(identifier)
        done.add(identifier)

    done = set()
    for identifier in by_id:
        visit(identifier, set(), done)
    return errors


def check_toc(resource, root):
    """Check the exported card and the source that regenerates it."""
    errors = []
    source = resource["source"]
    images = (root / "docs/images").resolve()
    image = images / resource["tocFigure"]
    if (
        resource["role"] != "Studies"
        or not image.resolve().is_relative_to(images)
        or not image.is_file()
    ):
        errors.append(f"{source}: invalid study TOC figure")
    if is_literature_study(resource):
        renderer = root / resource.get("tocSource", "")
        if (
            not renderer.resolve().is_relative_to(root.resolve())
            or not renderer.is_file()
            or not renderer.name.endswith((".py", ".svg.in"))
        ):
            errors.append(f"{source}: missing or invalid TOC source")
    return errors


def check_resources(data, root, by_id):
    """Classify each authored page once with valid topic assignments."""
    errors, covered = [], []
    for resource in data["resources"]:
        source = resource["source"]
        covered.append(source.split("#", 1)[0])
        if "tocFigure" in resource:
            errors.extend(check_toc(resource, root))
        assignments = [resource["topic"], *resource["also"]]
        if len(set(assignments)) != len(assignments):
            errors.append(f"{source}: duplicate topic association")
        for identifier in assignments:
            if identifier not in by_id:
                errors.append(f"{source}: unknown topic {identifier}")
        if resource["role"] not in ROLES or not resource["title"].strip():
            errors.append(f"{source}: invalid resource title or role")
    for resource in data["supporting"]:
        covered.append(resource["source"])
        if not resource["reason"].strip():
            errors.append(f"{resource['source']}: unexplained supporting page")
    counts = Counter(covered)
    errors.extend(
        f"duplicate page disposition: {s}" for s, n in counts.items() if n > 1
    )
    expected = inventory(root) | {"api"}
    errors.extend(
        f"unclassified topic resource: {s}"
        for s in sorted(expected - set(covered))
    )
    errors.extend(
        f"missing topic resource: {s}" for s in sorted(set(covered) - expected)
    )
    return errors


def check_destinations(data, root, children):
    """Require useful leaves and collision-free generated topic routes."""
    errors = []
    used = {i for r in data["resources"] for i in [r["topic"], *r["also"]]}
    for topic in data["topics"]:
        identifier = topic["id"]
        if identifier not in used and identifier not in children:
            errors.append(f"{identifier}: empty topic leaf")
        authored = root / "docs" / (topic_route(identifier) + ".md")
        authored_index = root / "docs" / topic_route(identifier) / "index.md"
        if authored.exists() or (
            identifier != "aerosols" and authored_index.exists()
        ):
            errors.append(
                f"{identifier}: generated route collides with authored page"
            )
    return errors


def validate(data, root=ROOT):
    """Return errors for structure, semantics, assets, and complete coverage."""
    errors = []
    if data.get("version") != 1:
        errors.append("unsupported topic registry version")
    topics = data["topics"]
    by_id = {t["id"]: t for t in topics}
    if len(topics) != len(by_id):
        errors.append("duplicate topic identifier")
    if [t["id"] for t in topics if t["parent"] is None] != ["aerosols"]:
        errors.append("topic graph must have exactly one aerosols root")
    children = Counter(t["parent"] for t in topics if t["parent"])
    errors.extend(
        f"{i}: branching requires 2–4 children"
        for i, count in children.items()
        if not 2 <= count <= 4
    )
    orders = set()
    for topic in topics:
        errors.extend(check_topic_fields(topic, root))
        errors.extend(check_relationships(topic, by_id))
        order = (topic["parent"], topic["order"])
        if order in orders or not isinstance(topic["order"], int):
            errors.append(f"{topic['id']}: invalid or duplicate sibling order")
        orders.add(order)
    errors.extend(check_paths(by_id))
    errors.extend(check_resources(data, root, by_id))
    errors.extend(check_destinations(data, root, children))
    if (root / "docs/studies").is_dir():
        errors.extend(docs_studies.StudyCatalog.load(root, data).errors)
    return errors


def require_valid(root=ROOT):
    """Fail preparation before writing output when the graph has drifted."""
    data = load(root)
    errors = validate(data, root)
    if errors:
        raise ValueError("\n".join(errors))
    return data


def page_links(source, data, base):
    """Render topic associations outside scientific article content."""
    resource = next(
        (r for r in data["resources"] if r["source"] == source), None
    )
    if not resource:
        return ""
    topics = {t["id"]: t for t in data["topics"]}
    links = [
        f"[{topics[i]['label']}]({base}{topic_route(i)}/)"
        for i in [resource["topic"], *resource["also"]]
    ]
    return "\n\nExplore this topic: " + " · ".join(links) + "\n"


def generate(write_page, data):
    """Generate MDX adapters; the registry and component own their content."""
    for topic in data["topics"]:
        path = topic_route(topic["id"]) + ".mdx"
        # Generated adapters are three directories below src; use a Vite alias
        # so root and deeper topic routes share the same component import.
        body = (
            "import TopicExplorer from '@components/TopicExplorer.astro';\n\n"
            f'<TopicExplorer topicId="{topic["id"]}" />'
        )
        write_page(
            path,
            topic["label"],
            body,
            extra={
                "tableOfContents": False,
                "prev": False,
                "next": False,
                "editUrl": "https://github.com/uncscode/particula/edit/main/website/topics.json",
            },
        )


if __name__ == "__main__":
    graph = require_valid()
    print(
        f"Topic graph: {len(graph['topics'])} topics; "
        f"{len(graph['resources'])} resources; "
        f"{len(graph['supporting'])} supporting pages"
    )
