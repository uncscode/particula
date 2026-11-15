"""Topic graph invariants and publication adapters."""

import copy
import sys
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
        r for r in data["resources"] if r["topic"] != "transport"
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
    text = docs_topics.page_links(
        "docs/examples/gpu/prepared/gpu_environment_sweep.py",
        docs_topics.load(),
        "/particula/",
    )
    assert "/particula/guide/topics/atmosphere/" in text
    assert "/particula/guide/topics/gpu/" in text
    assert docs_topics.page_links("unmapped", docs_topics.load(), "/") == ""
