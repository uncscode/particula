"""Study catalog identity, portable directories, and explicit evidence links."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docs_studies  # noqa: E402


def source(root, relative, title="Study"):
    """Create an authored source with a portable directory section."""
    path = root / relative
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(
        f"# {title}\n\n<!-- STUDY_LIST_START -->\n<!-- STUDY_LIST_END -->\n"
    )
    return relative


def record(path, identifier, **metadata):
    """A canonical resource with one topic and optional study relationships."""
    return {
        "source": path,
        "title": identifier,
        "topic": "transport",
        "also": ["verification"],
        "study": {"id": identifier, **metadata},
    }


def test_shared_companion_multiscript_and_prose_only_studies(tmp_path):
    """Each script appears once; a shared specification links to both owners."""
    base = docs_studies.STUDIES
    for folder in ["", *docs_studies.CATEGORIES]:
        source(tmp_path, f"{base}/{folder}/index.md")
    source(tmp_path, docs_studies.SCRIPT_INDEX)
    report = source(tmp_path, f"{base}/validation/mobility/index.md")
    comparison = source(tmp_path, f"{base}/literature/paper.md")
    spec = source(tmp_path, f"{base}/validation/mobility/criteria.md")
    guide = source(tmp_path, "docs/guide/mobility.md")
    module = tmp_path / "particula/dynamics/wall_loss.py"
    module.parent.mkdir(parents=True)
    module.touch()
    graph = {
        "topics": [],
        "resources": [
            record(
                report,
                "mobility",
                runs=["run-a", "run-b"],
                companions=[spec],
                relations=[
                    {"target": guide, "fragment": "limits", "label": "Method"},
                    {
                        "target": "api:particula.dynamics.wall_loss.Loss",
                        "label": "API",
                    },
                ],
                secondary_category="literature",
            ),
            record(comparison, "paper", companions=[spec]),
        ],
    }
    runs = {
        key: {
            "source": source(tmp_path, f"{base}/validation/mobility/{key}.py"),
            "title": key,
            "device": "cpu",
        }
        for key in ["run-a", "run-b"]
    }
    catalog = docs_studies.StudyCatalog(graph, runs, tmp_path)
    assert not catalog.errors
    assert [r["key"] for r in catalog.rows] == ["run-a", "run-b"]
    assert catalog.rows[0]["topics"] == ["transport", "verification"]
    assert catalog.children(f"{base}/literature") == [comparison, report]
    assert catalog.children(f"{base}/validation") == [report]
    assert set(catalog.backlinks[spec]) == {
        ("paper", "studies/literature/paper/"),
        ("mobility", "studies/validation/mobility/"),
    }
    assert "/guide/mobility/#limits" in catalog.links(report, "/")
    assert (
        "/API/particula/dynamics/wall_loss/#particula.dynamics.wall_loss.Loss"
        in catalog.links(report, "/")
    )
    assert catalog.backlinks["api:particula.dynamics.wall_loss.Loss"]
    assert catalog.navigation()[0]["slug"] == "studies/scripts"
    assert docs_studies.offline_indexes(catalog)
    assert docs_studies.offline_indexes(catalog, write=True) == []
    assert docs_studies.offline_indexes(catalog) == []
    assert (
        "[run-a](validation/mobility/run-a.py)"
        in (tmp_path / docs_studies.SCRIPT_INDEX).read_text()
    )
    graph["resources"][1]["study"]["runs"] = ["run-a", "absent"]
    graph["resources"][0]["study"]["relations"].append(
        {"target": "docs/missing.md", "label": "Missing"}
    )
    errors = docs_studies.StudyCatalog(graph, runs, tmp_path).errors
    assert any("missing study target" in e for e in errors)
    assert any("expected one owning study, found 2" in e for e in errors)
    assert any("unknown execution absent" in e for e in errors)


def test_six_immediate_choices_review_seventh_but_catalog_unbounded(tmp_path):
    """A seventh run in one study is still a single directory choice."""
    folder = "docs/studies/examples"
    source(tmp_path, "docs/studies/index.md")
    source(tmp_path, folder + "/index.md")
    studies = [
        record(source(tmp_path, f"{folder}/case-{i}.md"), f"case-{i}")
        for i in range(7)
    ]
    runs = {
        f"run-{i}": {
            "source": source(tmp_path, f"{folder}/run-{i}.py"),
            "device": "cuda",
            "title": f"Run {i}",
        }
        for i in range(7)
    }
    studies[0]["study"]["runs"] = list(runs)
    graph = {"topics": [], "resources": studies[:6]}
    catalog = docs_studies.StudyCatalog(graph, runs, tmp_path)
    assert len(catalog.rows) == 7
    assert catalog.errors == []
    graph["resources"] = studies
    catalog = docs_studies.StudyCatalog(graph, runs, tmp_path)
    assert any(
        "review 7 immediate choices" in err and "case-6" in err
        for err in catalog.errors
    )
    graph["study_group_exceptions"] = {
        folder: "A published seven-stage sequence."
    }
    assert docs_studies.StudyCatalog(graph, runs, tmp_path).errors == []
