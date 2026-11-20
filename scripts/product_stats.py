"""Independent archive metadata check for quarterly coverage and crowding."""

from archive import ROOT, tap, rows, write


def run():
    audit = rows(ROOT / "data/catalog/current_data_audit.csv")
    ks = sorted({r["kic"] for r in audit})
    schema = rows(
        tap(
            "lc_archive_schema",
            "select column_name,unit,description from TAP_SCHEMA.columns where table_name='KEPLERTIMESERIES'",
        )
    )
    cols = "star_id targettype quarter start_time end_time npts data_release file_version exposure livetime cdpp3_0 cdpp6_0 crowdsap flfrcsap pdcmethod".split()
    assert set(cols) <= {r["column_name"] for r in schema}
    rs = rows(
        tap(
            "quarterly_metadata",
            "select "
            + ",".join(cols)
            + " from KEPLERTIMESERIES where star_id in ("
            + ",".join(ks)
            + ")",
        )
    )
    for r in audit:
        match = [v for v in rs if v["star_id"] == r["kic"]]
        lc = [v for v in match if v["targettype"].strip().lower() == "long cadence"]
        r["nea_lc_metadata_rows"] = len(lc)
        r["nea_sc_metadata_rows"] = len(match) - len(lc)
        r["nea_sc_quarters"] = " ".join(
            map(
                str,
                sorted(
                    {
                        int(v["quarter"])
                        for v in match
                        if v["targettype"].strip().lower() == "short cadence"
                    }
                ),
            )
        )
        if r["quality_status"].startswith("filename"):
            r["quality_status"] = (
                "quarter metadata only; individual quality flags uninspected"
            )
        for name, op in [("crowdsap", min), ("flfrcsap", min), ("cdpp6_0", max)]:
            vals = [float(v[name]) for v in lc if v[name]]
            r["nea_lc_" + name + "_" + op.__name__] = op(vals) if vals else ""
        r["nea_product_releases"] = " ".join(sorted({v["data_release"] for v in match}))
    write(ROOT / "data/catalog/current_data_audit.csv", audit)


if __name__ == "__main__":
    run()
