"""Inventory later TESS observations separately from the Kepler comparison."""

import json
from archive import ROOT, fetch, rows, write

URL = "https://mast.stsci.edu/api/v0/invoke"


def invoke(name, service, params):
    response = json.loads(
        fetch(
            name + ".json",
            URL,
            dict(service=service, params=params, format="json", pagesize=50000, page=1),
        ).read_text()
    )
    if response.get("status") != "COMPLETE":
        raise RuntimeError(str(response)[:400])
    return response["data"]


def run():
    audit = rows(ROOT / "data/catalog/current_data_audit.csv")
    kics = sorted(set(r["kic"] for r in audit))
    tic = invoke(
        "tic_crossmatch",
        "Mast.Catalogs.Filtered.Tic",
        dict(
            columns="ID,KIC,disposition", filters=[dict(paramName="KIC", values=kics)]
        ),
    )
    ids = {
        k: [
            str(t["ID"])
            for t in tic
            if str(t["KIC"]) == k
            and str(t["disposition"]).upper() not in ("DUPLICATE", "ARTIFACT", "SPLIT")
        ]
        for k in kics
    }
    targets = sorted({i[0] for i in ids.values() if len(i) == 1})
    obs = invoke(
        "tess_observations",
        "Mast.Caom.Filtered",
        dict(
            columns="target_name,obs_id,t_min,t_max,sequence_number,provenance_name,dataproduct_type",
            filters=[
                dict(paramName="obs_collection", values=["TESS"]),
                dict(paramName="target_name", values=targets),
                dict(paramName="dataproduct_type", values=["timeseries"]),
            ],
        ),
    )
    for r in audit:
        ii = ids[r["kic"]]
        r["tic_kic_matches"] = " ".join(ii)
        sel = [o for o in obs if len(ii) == 1 and o["target_name"] == ii[0]]
        r["tess_targeted_timeseries"] = len(sel) if len(ii) == 1 else ""
        r["tess_sectors"] = " ".join(
            map(
                str,
                sorted(
                    set(
                        o["sequence_number"]
                        for o in sel
                        if o["sequence_number"] is not None
                    )
                ),
            )
        )
    write(ROOT / "data/catalog/current_data_audit.csv", audit)


if __name__ == "__main__":
    run()
