"""Verify journal table values and enrich target-level comparisons."""

import csv
import json
from urllib.parse import urlencode
from archive import META, ROOT, fetch, rows, write
from inventory import koi


def table(name, source):
    path = fetch(
        name + ".tsv",
        "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?"
        + urlencode({"-source": source, "-out.all": 1}),
    )
    lines = [s for s in path.read_text().splitlines() if s and not s.startswith("#")]
    return [
        {k: v.strip() for k, v in r.items()}
        for r in csv.DictReader([lines[0], *lines[3:]], delimiter="\t")
    ]


def run():
    journal = {koi(r["KOI"]): r for r in table("published", "J/AJ/154/228/catalog")}
    hubs = {r["KIC"]: r for r in table("huber", "J/ApJ/767/127/hosts")}
    tagged = fetch(
        "release_tag.json",
        "https://api.github.com/repos/esandford/KnowThePlanet/git/trees/v1.0.1?recursive=1",
    )
    assert (
        json.loads(tagged.read_text())["tree"]
        == json.loads((META / "posterior_tree.json").read_text())["tree"]
    )
    orig = rows(ROOT / "data/catalog/original_sample.csv")
    audit = rows(ROOT / "data/catalog/current_data_audit.csv")
    assert len(journal) == len(orig) == len(audit) == 75
    solutions = rows(META / "ps.csv")
    retrieval_date = json.loads((META / "CUMULATIVE.csv.source.json").read_text())[
        "utc"
    ][:10]
    kics = sorted({r["kic"] for r in orig})
    quality = fetch(
        "berger_astrometry.tsv",
        "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=J/AJ/159/280&KIC="
        + ",".join(kics)
        + "&-out=KIC,RUWE,Ncomp&-out.max=1000",
    )
    first = quality.read_text().split("#Table")[1]
    lines = [
        line for line in first.splitlines() if line.strip() and not line.startswith("#")
    ]
    start = lines.index("KIC\tRUWE\tNcomp")
    flags = {
        r["KIC"].strip(): r
        for r in csv.DictReader(
            [
                lines[start],
                *(
                    line
                    for line in lines[start + 1 :]
                    if line.split("\t", 1)[0].strip().isdigit()
                ),
            ],
            delimiter="\t",
        )
    }
    diffs = []
    for r, v in zip(orig, audit):
        j = journal[r["koi"]]
        match = flags.get(r["kic"], {})
        for c in ["RUWE", "Ncomp"]:
            v["berger_" + c] = match.get(c, "").strip()
        assert r["koi"] == v["koi"] and j["KIC"] == r["kic"]
        assert {
            "Occultation": "occultation",
            "multis": "multiplanet",
            "tidal": "tidal",
        }[j["Target"]] == r["category"]
        for c, k in [
            ("t0[BKJD]", "t0"),
            ("P[days]", "Porb"),
            ("b", "b"),
            ("log10rhostar[kg/m^3]", "log10rho*"),
            ("Rp/R*", "Rp/R*"),
        ]:
            for pre in ("", "E_", "e_"):
                if float(r[pre + c]) != float(j[pre + k]):
                    diffs.append((r["koi"], pre + c, r[pre + c], j[pre + k]))
        if "Fig 3" not in r["paper_cadence"]:
            r["paper_cadence"] = (
                "LC and SC" if int(v["sc_files"]) else "LC only"
            ) + " (inferred from Sec 2.3 rule and DR25 inventory)"
        logr = float(r["log10rhostar[kg/m^3]"])
        for suffix, exp in [
            ("p50", logr),
            ("p16", logr - float(r["e_log10rhostar[kg/m^3]"])),
            ("p84", logr + float(r["E_log10rhostar[kg/m^3]"])),
        ]:
            r["rho_kg_m3_" + suffix] = 10**exp
        r["physical_fractional_halfwidth_68"] = (
            r["rho_kg_m3_p84"] - r["rho_kg_m3_p16"]
        ) / (2 * r["rho_kg_m3_p50"])
        if v["berger_KIC"]:
            for c in ["rho", "E_rho", "e_rho"]:
                v["berger_" + c + "_kg_m3"] = 1408 * 10 ** float(v["berger_" + c])
        h = hubs.get(r["kic"], {})
        v.update({"huber_" + c: h.get(c, "") for c in ["rho", "e_rho", "Mass", "Rad"]})
        v["release"] = "DR25 standard SOC"
        v["flux_columns"] = "SAP_FLUX PDCSAP_FLUX (standard product)"
        if v["cum_koi_period"] and v["cum_koi_time0bk"]:
            p = float(r["P[days]"])
            v["cum_minus_sk_period_s"] = (float(v["cum_koi_period"]) - p) * 86400
            delta = float(v["cum_koi_time0bk"]) - float(r["t0[BKJD]"])
            v["cum_minus_sk_epoch_mod_P_days"] = (delta + p / 2) % p - p / 2
        linked = [
            row
            for row in solutions
            if row["pl_name"] == v.get("names_pl_name") and row.get("pl_orbeccen")
        ]
        v["ps_eccentricity_solutions"] = len(linked)
        v["ps_e_with_errors_or_limit"] = sum(
            bool(
                row["pl_orbeccenerr1"]
                or row["pl_orbeccenerr2"]
                or row["pl_orbeccenlim"] in ("1", "-1")
            )
            for row in linked
        )
        if v.get("berger_rho_kg_m3"):
            v["berger_rho_over_sk"] = v["berger_rho_kg_m3"] / float(r["rho_kg_m3_p50"])
        if v.get("huber_rho"):
            v["huber_rho_over_sk"] = (
                float(v["huber_rho"]) * 1000 / float(r["rho_kg_m3_p50"])
            )
        v["retrieval_utc_date"] = retrieval_date
        k = r["kic"].zfill(9)
        v["mast_listing_url"] = (
            f"https://archive.stsci.edu/missions/kepler/lightcurves/{k[:4]}/{k}/"
        )
        if (
            r["koi"] == "K00929.01"
            and (ROOT / "results/data_quality/quarters.csv").exists()
        ):
            v["quality_status"] = "FITS verified; see results/data_quality/quarters.csv"
    write(ROOT / "data/catalog/current_data_audit.csv", audit)
    write(ROOT / "data/catalog/original_sample.csv", orig)
    (ROOT / "reports/journal_check.json").write_text(
        json.dumps(dict(rows=75, columns_checked=15, differences=diffs), indent=2)
    )


if __name__ == "__main__":
    run()
