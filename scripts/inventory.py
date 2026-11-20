from concurrent.futures import ThreadPoolExecutor
import csv
import json
import re
from archive import META, ROOT, fetch, sources, tap, write


def koi(value):
    match = re.fullmatch(r"K?(\d{1,5})\.(\d{2})", value)
    if not match:
        raise ValueError(value)
    return f"K{int(match[1]):05}.{match[2]}"


def original():
    sources()
    tree = json.loads((META / "posterior_tree.json").read_text())
    paths = {e["path"] for e in tree["tree"]}
    result = []
    for cat, prior in [
        ("occultation", "Gaussian e cos(w), e sin(w) from eclipses"),
        ("tidal", "exponential e scale 0.00796"),
        ("multiplanet", "Rayleigh e scale 0.049"),
    ]:
        with (META / f"{cat}_targets.txt").open() as f:
            for record in csv.DictReader(f, delimiter="\t"):
                key = koi(record.pop("KOI name"))
                path = f"{cat}_targets/{key}_samples.txt"
                assert path in paths
                num = int(key[1:6])
                alerts = []
                for group, alert in [
                    ({1, 5, 10, 412}, "weak occultation detection"),
                    (
                        {1, 823, 1075, 1658, 153, 1779},
                        "anomalous limb darkening and eccentricity",
                    ),
                    ({1541}, "anomalous limb darkening"),
                    ({5157, 7430, 153}, "authors advise against adoption"),
                    ({1793, 4351, 3913}, "bimodality"),
                    ({5}, "low SNR grazing"),
                    ({1}, "seismic/RV disagreement"),
                ]:
                    if num in group:
                        alerts.append(alert)
                if float(record["b"]) + float(record["Rp/R*"]) >= 1:
                    alerts.append("marginal medians imply grazing")
                result.append(
                    dict(
                        koi=key,
                        kic=record.pop("KIC"),
                        category=cat,
                        e_prior=prior,
                        joint_fit=cat == "multiplanet",
                        paper_cadence="LC only (Fig 3)"
                        if num == 929
                        else "LC; SC if available (individual use unverified)",
                        warnings=" | ".join(alerts),
                        posterior_url=f"https://raw.githubusercontent.com/esandford/KnowThePlanet/{tree['sha']}/{path}",
                        **record,
                    )
                )
    assert (
        len(set(r["koi"] for r in result)) == 75
        and len(set(r["kic"] for r in result)) == 66
    )
    write(ROOT / "data/catalog/original_sample.csv", result)
    return result


def mast(kic):
    k = kic.zfill(9)
    url = f"https://archive.stsci.edu/missions/kepler/lightcurves/{k[:4]}/{k}/"
    # Directory listings enumerate files without accessing photometry.
    path = META / "mast" / f"{k}.html"
    if not path.exists():
        from subprocess import run

        path.parent.mkdir(exist_ok=True)
        run(
            ["curl", "-fsSL", "--max-time", "45", "--retry", "1", url, "-o", str(path)],
            check=True,
        )
    files = sorted(
        set(re.findall(r"kplr" + k + r"-\d+_[sl]lc\.fits", path.read_text()))
    )
    if not files:
        raise ValueError("Missing file listing " + url)
    return [
        dict(kic=kic, url=url + f, filename=f, cadence="LC" if "_llc" in f else "SC")
        for f in files
    ]


if __name__ == "__main__":
    records = original()
    ids = sorted({r["kic"] for r in records})
    fetch("zenodo.json", "https://zenodo.org/api/records/1028515")
    tap("table_list", "select table_name from TAP_SCHEMA.tables")
    with ThreadPoolExecutor(5) as pool:
        groups = list(pool.map(mast, ids))
    write(META / "mast_products.csv", [record for group in groups for record in group])
