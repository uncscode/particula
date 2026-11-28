"""Command-line access to the same chemistry catalog used by Python studies."""

import argparse
import json
import sqlite3

from particula.dynamics.chemistry.catalog.readers import SOURCES
from particula.dynamics.chemistry.catalog.sources import recipes
from particula.dynamics.chemistry.catalog.store import ChemistryCatalog


def main(argv=None):
    """Run particula chem import, search, show or export."""
    parser = argparse.ArgumentParser(prog="particula")
    top = parser.add_subparsers(required=True)
    chem = top.add_parser("chem")
    commands = chem.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch", help="install the release catalog or import a named source"
    )
    fetch.add_argument("recipe", nargs="?", choices=sorted(recipes()))
    ingest = commands.add_parser(
        "import", help="cache and index local model files"
    )
    ingest.add_argument("source", choices=SOURCES)
    ingest.add_argument("file")
    ingest.add_argument("--release", required=True)
    ingest.add_argument("--mechanism", required=True)
    ingest.add_argument("--aerosol")
    ingest.add_argument("--companion", action="append", default=[])
    search = commands.add_parser(
        "search", help="find channels by exact source species"
    )
    search.add_argument("--species")
    search.add_argument(
        "--snapshot", help="restrict results to this snapshot ID"
    )
    search.add_argument("--source", choices=SOURCES)
    search.add_argument("--role", choices=["reactant", "product"])
    show = commands.add_parser("show", help="inspect one reaction or snapshot")
    show.add_argument("id")
    export = commands.add_parser(
        "export", help="export selected IDs or replay a selection lock"
    )
    export.add_argument("ids", nargs="*")
    export.add_argument("--lock")
    export.add_argument("--output", required=True)
    export.add_argument(
        "--target", choices=["lock", "native"], default="native"
    )
    export.add_argument("--close-oxygen", action="store_true")
    for p in (fetch, ingest, search, show, export):
        p.add_argument(
            "--json", action="store_true", help="write machine-readable JSON"
        )
    args = parser.parse_args(argv)
    try:
        catalog = ChemistryCatalog()
        if args.command == "fetch":
            result = catalog.fetch(args.recipe)
            if args.recipe is not None:
                result = catalog.show(result)
        elif args.command == "import":
            snap = catalog.import_file(
                args.source,
                args.file,
                release=args.release,
                mechanism=args.mechanism,
                companions=args.companion,
                aerosol=args.aerosol,
            )
            result = catalog.show(snap)
        elif args.command == "search":
            result = catalog.search(
                species=args.species,
                source=args.source,
                role=args.role,
                snapshot=args.snapshot,
            )
        elif args.command == "show":
            result = catalog.show(args.id)
        else:
            catalog.export(
                args.ids,
                args.output,
                target=args.target,
                lock_path=args.lock,
                close_oxygen=args.close_oxygen,
            )
            result = {"output": args.output, "target": args.target}
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(
            2,
            (json.dumps({"error": str(exc)}) if args.json else str(exc)) + "\n",
        )
    if isinstance(result, list) and not args.json:
        for row in result:
            print(row["id"], row["equation"], sep="\t")
    else:
        print(
            json.dumps(result, indent=None if args.json else 2, allow_nan=False)
        )


if __name__ == "__main__":
    main()
