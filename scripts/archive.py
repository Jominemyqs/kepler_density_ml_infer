"""Cache public archive requests atomically with retrieval date and content digest."""

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/metadata"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def fetch(name, url, params=None):
    path = META / name
    provenance = path.with_suffix(path.suffix + ".source.json")
    if path.exists():
        if provenance.exists():
            record = json.loads(provenance.read_text())
            if record["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Cached download checksum mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    subprocess.run(
        ["curl", "-gfsSL", "--retry", "2", "--max-time", "90", url, "-o", str(tmp)]
        + (["--data-urlencode", "request=" + json.dumps(params)] if params else []),
        check=True,
    )
    if tmp.read_bytes()[:512].lower().find(b"<html") >= 0:
        raise ValueError(f"Unexpected HTML for {name}")
    tmp.replace(path)
    path.with_suffix(path.suffix + ".source.json").write_text(
        json.dumps(
            dict(
                url=url,
                request=params,
                utc=datetime.now(timezone.utc).isoformat(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )
    return path


def tap(name, query, service=TAP):
    return fetch(
        name + ".csv",
        service
        + "?"
        + urlencode(
            dict(
                query=query, format="csv", lang="ADQL", request="doQuery", maxrec=100000
            )
        ),
    )


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write(path, records):
    with path.open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=list(dict.fromkeys(k for r in records for k in r))
        )
        w.writeheader()
        w.writerows(records)


def sources():
    tree = json.loads(
        fetch(
            "posterior_tree.json",
            "https://api.github.com/repos/esandford/KnowThePlanet/git/trees/master?recursive=1",
        ).read_text()
    )
    for cat in ["occultation", "tidal", "multiplanet"]:
        name = cat + "_targets.txt"
        fetch(
            name,
            f"https://raw.githubusercontent.com/esandford/KnowThePlanet/{tree['sha']}/{name}",
        )
    tap(
        "nea_schema",
        "select table_name,column_name,unit,description from TAP_SCHEMA.columns where table_name in ('cumulative','q1_q17_dr25_koi','q1_q17_dr25_stellar','keplerstellar','ps','keplernames')",
    )


if __name__ == "__main__":
    sources()
