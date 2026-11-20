"""DR25 long-cadence SAP transit extraction. All time coordinates in days."""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from archive import ROOT, META, rows, write
from download import target, TARGET

OUT = ROOT / "results/data_quality"
DATA = ROOT / "data/processed"


def offset(t, t0, period):
    return (t - t0 + period / 2) % period - period / 2


def bkjd(h, t):
    assert (
        h["TIMESYS"] == "TDB" and h["TIMEUNIT"] == "d" and h["TIMEREF"] == "SOLARSYSTEM"
    )
    return t + ((h["BJDREFI"] - 2454833) + h["BJDREFF"])


def baseline(t, y, err, out):
    scale = max(abs(t))
    z = t / scale
    choices = []
    for deg in range(4):
        fit = np.polynomial.polynomial.polyfit(z[out], y[out], deg, w=1 / err[out])
        model = np.polynomial.polynomial.polyval(z, fit)
        bic = np.sum(((y[out] - model[out]) / err[out]) ** 2) + (deg + 1) * np.log(
            out.sum()
        )
        choices.append((bic, deg, model))
    return min(choices, key=lambda v: v[0])[1:]


def run():
    v = target()
    period, t0, duration = map(
        float, [v["cum_koi_period"], v["cum_koi_time0bk"], v["cum_koi_duration"]]
    )
    companions = [
        r
        for r in rows(META / "CUMULATIVE.csv")
        if r["kepid"] == v["kic"] and r["kepoi_name"] != TARGET
    ]
    segments = []
    manifest = []
    events = []
    raw = []
    for path in sorted((ROOT / "data/raw" / v["kic"]).glob("*.fits")):
        with fits.open(path) as h:
            header = h[0].header
            assert (
                header["KEPLERID"] == int(v["kic"])
                and header["OBSMODE"] == "long cadence"
            )
            d = h[1].data
            assert h[1].columns["SAP_FLUX"].unit == "e-/s"
            t = bkjd(h[1].header, d["TIME"])
            f, e, p, pe = [
                d[c].astype(float)
                for c in ("SAP_FLUX", "SAP_FLUX_ERR", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR")
            ]
            quality = d["SAP_QUALITY"]
            cad = d["CADENCENO"]
            q = header["QUARTER"]
            dt = h[1].header["TIMEDEL"]
            mask = np.isfinite(t + f + e) & (e > 0) & (quality == 0)
            raw.append((t.copy(), f.copy()))
            clean = t[mask]
            manifest.append(
                dict(
                    filename=path.name,
                    quarter=q,
                    release=header["DATA_REL"],
                    pipeline=header["PROCVER"],
                    rows=len(t),
                    flagged=int(sum(quality != 0)),
                    good=int(mask.sum()),
                    flag_bits_counts=" ".join(
                        f"{bit}:{np.count_nonzero(quality & (1 << bit))}"
                        for bit in range(31)
                        if np.any(quality.astype(np.int64) & (1 << bit))
                    ),
                    largest_good_gap_days=np.max(np.diff(clean)),
                    crowdsap=h[1].header["CROWDSAP"],
                    flfrcsap=h[1].header["FLFRCSAP"],
                    cadence_days=dt,
                )
            )
            for c in companions:
                mask &= (
                    abs(offset(t, float(c["koi_time0bk"]), float(c["koi_period"])))
                    > float(c["koi_duration"]) / 48 + dt
                )
            manifest[-1]["companion_masked_good_samples"] = int(len(clean) - mask.sum())
            half = 1.1 * duration / 48 + dt
            for n in range(
                int(np.ceil((clean.min() - t0) / period)),
                int(np.floor((clean.max() - t0) / period)) + 1,
            ):
                center = t0 + n * period
                i = np.flatnonzero(mask & (abs(t - center) < half * (1 + np.sqrt(10))))
                s = t[i] - center
                out = abs(s) > half
                event = dict(
                    quarter=q,
                    epoch=n,
                    center_bkjd=center,
                    accepted=False,
                    reason="coverage",
                )
                events.append(event)
                if out.sum() < 6:
                    continue
                # Reject isolated OOT deviations relative to a preliminary trend.
                _, trend = baseline(s, f[i], e[i], out)
                resid = (f[i] - trend) / e[i]
                scale = max(
                    1, 1.4826 * np.median(abs(resid[out] - np.median(resid[out])))
                )
                keep = (~out | (abs(resid) < 4 * scale)) & (out | (resid < 4 * scale))
                i, s, out = i[keep], s[keep], out[keep]
                if (
                    any(
                        len(side) < 3
                        or np.ptp(side) < 2 * dt
                        or min(abs(side)) > half + 2 * dt
                        for side in [s[s < -half], s[s > half]]
                    )
                    or sum(abs(s) < duration / 48) < 3
                ):
                    continue
                degree, tr = baseline(s, f[i], e[i], out)
                pd = np.full(len(i), np.nan)
                valid = np.isfinite(p[i] + pe[i]) & (pe[i] > 0)
                if sum(out & valid) >= 6:
                    _, ptr = baseline(s, p[i], pe[i], out & valid)
                    pd[valid] = p[i][valid] / ptr[valid]
                event.update(accepted=True, reason="retained")
                segments.append(
                    np.column_stack(
                        [
                            t[i],
                            s,
                            np.full(len(i), q),
                            np.full(len(i), n),
                            cad[i],
                            quality[i],
                            f[i],
                            e[i],
                            tr,
                            f[i] / tr,
                            e[i] / tr,
                            pd,
                            np.full(len(i), degree),
                        ]
                    )
                )
    a = np.concatenate(segments)
    cols = "bkjd phase_days quarter epoch cadence quality sap sap_err trend norm norm_err pdc_norm degree"
    np.savetxt(
        DATA / "segments.csv.gz",
        a,
        delimiter=",",
        header=",".join(cols.split()),
        comments="",
        fmt="%.12g",
    )
    write(OUT / "quarters.csv", manifest)
    write(OUT / "events.csv", events)
    edges = np.linspace(-0.18, 0.18, 61)
    centers = (edges[1:] + edges[:-1]) / 2
    idx = np.searchsorted(edges, a[:, 1], side="right") - 1
    bins = []
    grid = []
    weights = []
    for j, x in enumerate(centers):
        b = a[idx == j]
        w = b[:, 10] ** -2
        assert len(b) > 0
        bins.append(
            [
                x,
                np.average(b[:, 9], weights=w),
                1 / np.sqrt(w.sum()),
                len(b),
                np.average(
                    b[np.isfinite(b[:, 11]), 11], weights=w[np.isfinite(b[:, 11])]
                )
                if np.isfinite(b[:, 11]).any()
                else np.nan,
            ]
        )
        # Preserve within-bin phase sampling with four weighted centroids.
        parts = np.array_split(np.argsort(b[:, 1]), 4)
        grid.append([np.average(b[k, 1], weights=w[k]) for k in parts])
        weights.append([w[k].sum() / w.sum() for k in parts])
    bins = np.array(bins)
    np.savez(
        DATA / "folded.npz",
        phase=bins[:, 0],
        flux=bins[:, 1],
        error=bins[:, 2],
        counts=bins[:, 3],
        pdc=bins[:, 4],
        grid=grid,
        weights=weights,
        cadence=dt,
        period=period,
        t0=t0,
        ld=[float(v["cum_koi_ldm_coeff1"]), float(v["cum_koi_ldm_coeff2"])],
    )
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), layout="constrained")
    for t, f in raw:
        ax[0].plot(t, f, ".", ms=0.4, color="#355870")
    ax[0].plot(
        [u["center_bkjd"] for u in events if u["accepted"]],
        np.full(sum(u["accepted"] for u in events), ax[0].get_ylim()[0]),
        "|",
        ms=4,
        color="orange",
    )
    ax[0].set(
        xlabel="BKJD (TDB), retained events marked below", ylabel="SAP (electrons/s)"
    )
    for s in segments[:: max(1, len(segments) // 3)][:3]:
        ax[1].plot(
            s[:, 1] * 24,
            s[:, 6] / np.median(s[:, 8]),
            ".",
            ms=3,
            label=f"Q{int(s[0, 2])}, epoch {int(s[0, 3])}",
        )
        ax[1].plot(s[:, 1] * 24, s[:, 8] / np.median(s[:, 8]), lw=0.8)
        ax[2].errorbar(s[:, 1] * 24, s[:, 9], s[:, 10], fmt=".", ms=3, alpha=0.65)
    ax[1].legend()
    ax[1].set(ylabel="SAP / median baseline")
    ax[2].set(xlabel="Hours from transit center", ylabel="Normalized SAP")
    fig.savefig(OUT / "light_curve_segments.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9, 7), layout="constrained")
    ax[0].scatter(a[:, 1] * 24, a[:, 9], s=0.3, alpha=0.15)
    ax[0].errorbar(centers * 24, bins[:, 1], bins[:, 2], fmt=".", label="SAP")
    ax[0].plot(centers * 24, bins[:, 4], label="PDCSAP")
    ax[0].legend()
    ax[0].set(ylabel="Normalized flux", xlim=(-4.4, 4.4))
    ax[1].plot(centers * 24, 1e6 * (bins[:, 4] - bins[:, 1]))
    ax[1].set(ylabel="PDC minus SAP (ppm)")
    ax[2].bar(centers * 24, bins[:, 3], width=0.11, alpha=0.5)
    ax[2].set(ylabel="Samples per bin", xlabel="Hours from transit center")
    right = ax[2].twinx()
    right.plot(centers * 24, 1e6 * bins[:, 2], "k.")
    right.set_ylabel("Formal bin error (ppm)")
    fig.savefig(OUT / "folded_coverage.png", dpi=150)
    plt.close(fig)
    print(len(manifest), len(events), len(segments), len(a))


if __name__ == "__main__":
    run()
