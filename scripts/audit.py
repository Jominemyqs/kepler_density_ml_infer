"""Exact-key joins; retain every referenced PS solution separately."""

from concurrent.futures import ThreadPoolExecutor
from archive import META, ROOT, rows, tap, write, fetch
from inventory import original, mast
from urllib.parse import urlencode

KOI = "kepid kepoi_name kepler_name koi_disposition koi_pdisposition koi_srho koi_period koi_time0bk koi_count koi_num_transits koi_model_snr koi_duration koi_depth koi_eccen koi_delivname koi_parm_prov koi_limbdark_mod koi_ldm_coeff1 koi_ldm_coeff2".split()
STAR = "kepid mass radius dens st_delivname teff_prov logg_prov feh_prov dutycycle st_quarters".split()
PS = "pl_name hostname default_flag pl_refname st_refname pl_orbeccen pl_orbeccenlim st_dens st_mass st_rad pl_orbper pl_tranmid ttv_flag sy_pnum".split()


def indices(rs, key):
    out = {}
    for r in rs:
        assert r[key] not in out, r[key]
        out[r[key]] = r
    return out


def run():
    orig = original()
    ids = sorted({r["kic"] for r in orig})
    where = "kepid in (" + ",".join(ids) + ")"
    tap(
        "kepler_schema",
        "select table_name,column_name,unit,description from TAP_SCHEMA.columns where table_name in ('CUMULATIVE','Q1_Q17_DR25_KOI','Q1_Q17_DR25_KS','KEPLERSTELLAR','Q1_Q17_DR25_SUP_KS')",
    )
    schema = rows(META / "nea_schema.csv") + rows(META / "kepler_schema.csv")

    def query(table, base, condition):
        available = {r["column_name"] for r in schema if r["table_name"] == table}
        assert set(base) <= available
        cols = [
            c + s
            for c in base
            for s in ("", "err1", "err2", "_err1", "_err2")
            if c + s in available
        ]
        return rows(
            tap(
                table,
                "select " + ",".join(cols) + " from " + table + " where " + condition,
            )
        )

    tables = [
        "CUMULATIVE",
        "Q1_Q17_DR25_KOI",
        "Q1_Q17_DR25_KS",
        "Q1_Q17_DR25_SUP_KS",
        "keplernames",
    ]
    with ThreadPoolExecutor(5) as pool:
        datasets = list(
            pool.map(
                lambda t: query(
                    t,
                    (
                        ["koi_name", "kepid", "pl_name", "kepler_name"]
                        if t == "keplernames"
                        else KOI
                        if t in tables[:2]
                        else STAR
                    ),
                    where,
                ),
                tables,
            )
        )
    maps = [
        indices(rs, k)
        for rs, k in zip(
            datasets, ["kepoi_name", "kepoi_name", "kepid", "kepid", "koi_name"]
        )
    ]
    names = sorted({r["pl_name"] for r in datasets[-1] if r["pl_name"]})
    publications = query(
        "ps",
        PS,
        "pl_name in ("
        + ",".join("'" + n.replace("'", "''") + "'" for n in names)
        + ")",
    )
    defaults = indices([r for r in publications if r["default_flag"] == "1"], "pl_name")
    products = [p for k in ids for p in mast(k)]
    write(META / "mast_products.csv", products)
    # Ordered quarterly LC timestamps, mission Q0 through Q17.
    stamps = sorted(
        {
            r["filename"].split("-")[1].split("_")[0]
            for r in products
            if r["cadence"] == "LC"
        }
    )
    stamps.remove("2010009091648")  # abbreviated Q4
    assert len(stamps) == 18, stamps
    # ASU emits field descriptions and units before the records.
    gaia_path = fetch(
        "berger.tsv",
        "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?"
        + urlencode(
            {
                "-source": "J/AJ/159/280/table2",
                "KIC": ",".join(ids),
                "-out.all": "1",
                "-out.max": "1000",
            }
        ),
    )
    import csv

    lines = [
        l for l in gaia_path.read_text().splitlines() if l and not l.startswith("#")
    ]
    gaia = indices(
        [
            {k: v.strip() for k, v in r.items()}
            for r in csv.DictReader([lines[0], *lines[3:]], delimiter="\t")
        ],
        "KIC",
    )
    result = []
    for r in orig:
        k = r["kic"]
        key = r["koi"]
        pp = [p for p in products if p["kic"] == k]
        qs = [
            i
            for i, s in enumerate(stamps)
            if any(
                s + "_llc" in p["filename"]
                or (i == 4 and "2010009091648_llc" in p["filename"])
                for p in pp
            )
        ]
        row = dict(
            koi=key,
            kic=k,
            lc_quarters=" ".join(map(str, qs)),
            missing_lc_quarters=" ".join(str(i) for i in range(18) if i not in qs),
            sc_files=sum(p["cadence"] == "SC" for p in pp),
            quality_status="filename inventory only",
        )
        for prefix, lookup, idx in zip(
            ["cum", "dr25", "ks", "ks_sup", "names"], maps, [key, key, k, k, key]
        ):
            item = lookup.get(idx, {})
            if item:
                assert item["kepid"] == k, (key, k, item["kepid"])
            row.update({prefix + "_" + c: v for c, v in item.items()})
        name = row.get("names_pl_name")
        row.update({"ps_" + c: v for c, v in defaults.get(name, {}).items()})
        row.update({"berger_" + c: v for c, v in gaia.get(k, {}).items()})
        row["sk_rho_kg_m3"] = 10 ** float(r["log10rhostar[kg/m^3]"])
        for c, d in [("cum_koi_srho", 1000), ("ks_dens", 1000)]:
            if row.get(c):
                row[c + "_over_sk"] = float(row[c]) * d / row["sk_rho_kg_m3"]
        result.append(row)
    write(ROOT / "data/catalog/current_data_audit.csv", result)


if __name__ == "__main__":
    run()
