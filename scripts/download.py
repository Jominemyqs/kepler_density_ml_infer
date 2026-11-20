from concurrent.futures import ThreadPoolExecutor
from archive import META, ROOT, fetch, rows

TARGET = "K00929.01"


def target():
    return next(
        r
        for r in rows(ROOT / "data/catalog/current_data_audit.csv")
        if r["koi"] == TARGET
    )


def run():
    k = target()["kic"]
    ps = [
        p
        for p in rows(META / "mast_products.csv")
        if p["kic"] == k and p["cadence"] == "LC"
    ]
    with ThreadPoolExecutor(4) as pool:
        list(
            pool.map(
                lambda p: fetch(ROOT / "data/raw" / k / p["filename"], p["url"]), ps
            )
        )
    ref = next(
        r for r in rows(ROOT / "data/catalog/original_sample.csv") if r["koi"] == TARGET
    )
    fetch("K00929.01_samples.txt", ref["posterior_url"])


if __name__ == "__main__":
    run()
