"""Topic graph invariants and publication adapters."""

import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docs_topics  # noqa: E402


def test_current_graph_covers_every_authored_page():
    """The maintained inventory must classify new pages explicitly."""
    assert docs_topics.validate(docs_topics.load()) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda d: d["topics"].append(copy.deepcopy(d["topics"][1])),
            "duplicate topic identifier",
        ),
        (lambda d: d["topics"][1].update(parent="absent"), "missing parent"),
        (lambda d: d["topics"][0].update(parent="state"), "hierarchy cycle"),
        (lambda d: d["topics"][1].update(icon="absent"), "missing icon"),
        (lambda d: d["topics"][1].pop("context"), "missing context"),
        (
            lambda d: d["topics"][1].update(investigation=" "),
            "missing investigation",
        ),
        (lambda d: d["topics"][1].update(related=["gas"]), "not reciprocal"),
        (
            lambda d: d["topics"][1].update(prerequisites=["state"]),
            "prerequisite cycle",
        ),
        (lambda d: d["resources"].pop(0), "unclassified topic resource"),
        (
            lambda d: d["supporting"][0].update(reason=""),
            "unexplained supporting",
        ),
        (
            lambda d: d["resources"][0].update(source="docs/missing.md"),
            "missing topic resource",
        ),
    ],
)
def test_invalid_graph_is_rejected(mutation, message):
    """Catch semantic drift rather than testing particular editorial wording."""
    data = docs_topics.load()
    mutation(data)
    assert any(message in error for error in docs_topics.validate(data))


def test_invalid_branching_and_empty_leaf():
    """A valid tree must remain useful to navigate."""
    data = docs_topics.load()
    for t in data["topics"]:
        if t["parent"] == "gas":
            t["parent"] = "state"
            t["order"] += 10
    errors = docs_topics.validate(data)
    assert any("branching requires" in e for e in errors)
    data = docs_topics.load()
    data["resources"] = [
        r
        for r in data["resources"]
        if r["topic"] != "transport" and "transport" not in r["also"]
    ]
    assert "transport: empty topic leaf" in docs_topics.validate(data)


def test_generated_topics_have_stable_distinct_routes():
    """Every node gets a static page that can be shared without JavaScript."""
    pages = {}

    def writer(path, title, body, **kwargs):
        assert path not in pages
        pages[path] = (title, body, kwargs)

    data = docs_topics.load()
    docs_topics.generate(writer, data)
    assert len(pages) == len(data["topics"])
    assert "guide/topics.mdx" in pages
    assert 'topicId="condensation"' in pages["guide/topics/condensation.mdx"][1]


def test_topic_associations_link_to_primary_and_related_topics():
    """A single article stays canonical while connecting across topic branches."""
    runs = json.loads((docs_topics.ROOT / "website/examples.json").read_text())
    text = docs_topics.page_links(
        runs["gpu-environment-sweep"]["source"],
        docs_topics.load(),
        "/particula/",
    )
    assert "/particula/guide/topics/atmosphere/" in text
    assert "/particula/guide/topics/gpu/" in text
    assert docs_topics.page_links("unmapped", docs_topics.load(), "/") == ""


@pytest.mark.parametrize("child_count", [1, 5])
def test_each_branch_requires_two_to_four_children(child_count):
    """Both sparse and overcrowded branches violate the same rule."""
    data = docs_topics.load()
    template = next(t for t in data["topics"] if t["id"] == "size")
    for index in range(child_count):
        node = copy.deepcopy(template)
        node.update(
            id=f"size-child-{index}", parent="size", order=index, related=[]
        )
        data["topics"].append(node)
        data["resources"][0]["also"].append(node["id"])
    assert "size: branching requires 2–4 children" in docs_topics.validate(data)


def test_deep_tree_validates_and_generates_every_destination():
    """Depth has no editorial cutoff; all branches still split in two."""
    data = docs_topics.load()
    template = next(t for t in data["topics"] if t["id"] == "size")
    parent = "size"
    for depth in range(8):
        for suffix in ("path", "leaf"):
            node = copy.deepcopy(template)
            node.update(
                id=f"deep-{depth}-{suffix}",
                parent=parent,
                order=int(suffix == "leaf"),
                related=[],
            )
            data["topics"].append(node)
            data["resources"][0]["also"].append(node["id"])
        parent = f"deep-{depth}-path"
    assert docs_topics.validate(data) == []
    destinations = []
    docs_topics.generate(
        lambda path, *args, **kwargs: destinations.append(path), data
    )
    assert "guide/topics/deep-7-path.mdx" in destinations
    assert len(destinations) == len(data["topics"])
    # A cycle at the deepest level must still be rejected.
    next(t for t in data["topics"] if t["id"] == "size")["parent"] = parent
    assert any("hierarchy cycle" in e for e in docs_topics.validate(data))


def test_topics_have_distinct_local_svg_illustrations():
    """Each topic has its own vector artwork in the shared icon style."""
    topics = docs_topics.load()["topics"]
    assert len({topic["icon"] for topic in topics}) == len(topics)
    drawings = set()
    for topic in topics:
        path = (
            docs_topics.ROOT
            / "website/public/topic-icons"
            / (topic["icon"] + ".svg")
        )
        # Only trusted SVG artwork checked into this repository is parsed.
        svg = ET.fromstring(path.read_text())  # noqa: S314
        assert svg.tag == "{http://www.w3.org/2000/svg}svg"
        assert svg.attrib["viewBox"] == "0 0 64 64"
        geometry = tuple(
            (child.tag, tuple(sorted(child.attrib.items())))
            for child in svg
            if child.tag != "{http://www.w3.org/2000/svg}title"
        )
        assert geometry and geometry not in drawings
        drawings.add(geometry)


def test_toc_figure_must_belong_to_a_study_and_exist():
    """Illustrated topic entries link to artwork copied into the site."""
    data = docs_topics.load()
    resource = next(row for row in data["resources"] if row.get("tocFigure"))
    resource["tocFigure"] = "missing-figure.svg"
    assert f"{resource['source']}: invalid study TOC figure" in (
        docs_topics.validate(data)
    )


def test_literature_card_requires_its_renderer():
    """A published illustration must retain an editable source in its study."""
    graph = docs_topics.load()
    resource = next(r for r in graph["resources"] if "tocSource" in r)
    del resource["tocSource"]
    assert any(
        "missing or invalid TOC source" in e
        for e in docs_topics.validate(graph)
    )
