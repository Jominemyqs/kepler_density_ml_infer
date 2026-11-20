"""Summarize the executed pilot and check exposure integration on actual cadences."""

import json
import numpy as np
import batman
from scipy.optimize import least_squares
from scipy.stats import kstest, binomtest
from archive import ROOT, META, write
from model import Simulator, LOW, HIGH
from pilot import OUT


def interval(q, j, digits):
    l, m, h = np.asarray(q)[:, j]
    return f"{m:.{digits}f} [{l:.{digits}f}, {h:.{digits}f}]"


def evaluate():
    d = json.loads((OUT / "diagnostics.json").read_text())
    s = Simulator()
    data = np.load(ROOT / "data/processed/folded.npz")
    a = np.genfromtxt(
        ROOT / "data/processed/segments.csv.gz", names=True, delimiter=","
    )
    estimates = {}
    for name, flux in [("SAP", s.flux), ("PDCSAP", data["pdc"])]:
        fit = least_squares(
            lambda t: (s.curve(t) - flux) / s.error,
            (LOW + HIGH) / 2,
            bounds=(LOW, HIGH),
        )
        estimates[name] = dict(theta=fit.x.tolist(), chi2=float(sum(fit.fun**2)))
    t = np.array(estimates["SAP"]["theta"])
    nominal = s.curve(t)
    duration = float(data["cadence"])
    # Independent high-resolution check at the actual, uncompressed phase coordinates.
    model = batman.TransitModel(
        s.p,
        a["phase_days"].copy(),
        exp_time=duration * 200 / 201,
        supersample_factor=201,
    )
    unbinned = model.light_curve(s.p)
    pos = (
        np.searchsorted(np.linspace(-0.18, 0.18, 61), a["phase_days"], side="right") - 1
    )
    exact = np.array(
        [
            np.average(unbinned[pos == j], weights=a["norm_err"][pos == j] ** -2)
            for j in range(60)
        ]
    )
    error = float(max(abs(exact - nominal) / s.error))
    # Keep a record of how much the older endpoint rule changed the optimum.
    old = batman.TransitModel(
        s.p, s.grid.ravel(), exp_time=duration, supersample_factor=11
    )

    def endpoint_residual(t):
        s.curve(t)
        m = (old.light_curve(s.p).reshape(s.grid.shape) * s.weights).sum(axis=1)
        return (m - s.flux) / s.error

    old_fit = least_squares(endpoint_residual, t, bounds=(LOW, HIGH))
    ranks = np.load(OUT / "heldout_posteriors.npz")["ranks"]
    n = len(ranks)
    calibration = []
    for level in [0.5, 0.68, 0.9, 0.95]:
        for j, name in enumerate(d["columns"][:3]):
            count = int(
                sum((ranks[:, j] > (1 - level) / 2) & (ranks[:, j] < (1 + level) / 2))
            )
            result = binomtest(count, n, level)
            ci = result.proportion_ci()
            calibration.append(
                dict(
                    nominal=level,
                    parameter=name,
                    covered=count,
                    n=n,
                    fraction=count / n,
                    ci95_low=ci.low,
                    ci95_high=ci.high,
                    binomial_p=result.pvalue,
                    rank_ks_p=kstest(ranks[:, j], "uniform").pvalue,
                )
            )
    write(OUT / "calibration.csv", calibration)
    publish = np.loadtxt(META / "K00929.01_samples.txt", skiprows=1)
    unphysical = int(sum(publish[:, 2] > 1 + publish[:, 4]))
    extra = dict(
        fits=estimates,
        max_bin_approximation_sigma=error,
        coarse_endpoint_theta=old_fit.x.tolist(),
        published_nontransiting=unphysical,
        published_rows=len(publish),
    )
    (OUT / "sensitivity.json").write_text(json.dumps(extra, indent=2))
    report = [
        "# KOI-929.01 pilot",
        "",
        "The conditional NPE and matched emcee fit were both run on a 60-bin folded SAP observation. The synthetic tests support local agreement at two test cases. Marginal coverage shows overcoverage at nominal 68%, and this target-specific pilot cannot establish reliability across stars.",
        "",
        "## Generative model",
        "",
        "Independent uniform priors in log10 density in kg/m³ [2.1, 3.4], b [0.15, 0.85] and Rp/Rstar [0.055, 0.12]. This is a restricted non-grazing giant-planet experiment. The prior is a pilot design choice. Each simulation draws fresh physical parameters from this box. Period 6.491682808 d and epoch 174.633553 BKJD are fixed at the documented catalog ephemeris. Set e = 0 and ω = 90° (irrelevant for a circle). The quadratic Kepler-band limb coefficients (0.3661, 0.2844) are fixed at the two KOI catalog values. Assume negligible planet mass, no dilution, TTVs or correlated noise.",
        "",
        "Use a/Rstar = [G ρ (86400 P)²/(3π)]^(1/3), SI units and cos(i) = b/(a/Rstar). Reject geometries that intersect the star on orbit or lack a transit. Batman integrates over a cadence-wide boxcar of 0.0204335982 d with 17 equal midpoint samples. The time span passed to batman accounts for its endpoint-inclusive sampling. Four weighted sub-bin phase centroids approximate each bin. Gaussian noise is independently drawn with the same per-bin formal uncertainty used in the MCMC likelihood.",
        "",
        f"Against 201 midpoint evaluations at every original cadence phase, the combined quadrature/binning approximation changes any bin by at most {error:.3f} formal σ at the SAP best fit. Numerical integration matters here. A preliminary 11-point endpoint average over the full cadence returned a best-fit density of {10 ** old_fit.x[0]:.0f} kg/m³; the checked midpoint calculation yields {10 ** t[0]:.0f} kg/m³. All reported samples and checkpoints use the midpoint calculation.",
        "",
        "## Estimator and reference",
        "",
        "The compact PyTorch model uses a bounded nonlinear least-squares transit fit as its observation embedding. It adds ten mean standardized residuals and uses the local Jacobian for whitening. In order, the autoregressive variables are b, log10ρ − 1.5 log10(1−b²), and radius ratio times the square root of the local quadratic limb intensity at the transit chord. These coordinates reduce curvature in the duration/depth degeneracy. A 32-unit hidden layer predicts residual locations and scales. Each conditional normal is truncated at the transformed physical bounds given previously drawn coordinates. Training minimizes normalized conditional negative log likelihood. The physical-coordinate Jacobian is constant with respect to the network weights for each training pair. Neural inference time includes fitting the observation.",
        "",
        "Training/validation/test RNG seeds 731/732/733 generate 1,600/240/160 independent simulations. Main torch, NumPy and posterior seed 730. Select the best validation checkpoint over 50 epochs. Only simulated parameters and fluxes train the network. The real data supply the observing design, bin errors and fixed ephemeris. Published samples enter only after inference. Use checkpoint.pt with Estimator, Simulator and draw in scripts/pilot.py.",
        "",
        "For each reference, emcee uses the same uniform box, batman mean function, fixed inputs, binning and diagonal Gaussian likelihood. There are 24 walkers, 400 discarded steps and thinning by four for storage. Run length adapts to estimated autocorrelation time subject to a cap. Walkers start near a least-squares fit. The chains check this restricted problem and cannot rule out separated modes in the wider published model.",
        "",
        "## Central 68% intervals",
        "",
        "| Observation | Fit | Density (kg/m³) | b | Radius ratio |",
        "|---|---|---|---|---|",
    ]
    for name in ["synthetic_0", "synthetic_1", "real", "published"]:
        if name.startswith("synthetic_"):
            truth = np.load(OUT / "simulations.npz")["test_theta"][
                int(name.split("_")[-1])
            ]
            report.append(
                f"| {name} | injected | {10 ** truth[0]:.1f} | {truth[1]:.3f} | {truth[2]:.5f} |"
            )
        for method in ["mcmc", "npe"] if name != "published" else ["SK 2017"]:
            q = d[name][method] if name != "published" else d[name]
            report.append(
                "| "
                + " | ".join(
                    [
                        name,
                        method,
                        interval(q, 3, 1),
                        interval(q, 1, 3),
                        interval(q, 2, 5),
                    ]
                )
                + " |"
            )
    report += [
        "",
        "Density quantiles are taken in physical units after exponentiating each draw. The published posterior contains 88,467 rows. Its 16th/50th/84th percentiles imply −12.5%/+15.4% in physical density. The geometry screen flags "
        + str(unphysical)
        + " samples with b > 1 + Rp/Rstar. The reference quantiles above include all rows. The paper sampled ten parameters for this planet, used eclipse priors and a flexible limb law, and applied a radius-ratio cutoff for confirmed planets. Its broader parameter space and different detrending both matter. The radius ratio shift here persists even when the matched estimators agree.",
        "",
        "## Calibration and fit quality",
        "",
        "| Nominal interval | log density covered | b covered | Radius ratio covered |",
        "|---|---:|---:|---:|",
    ]
    for level, values in d["coverage"].items():
        report.append(
            "| " + level + " | " + " | ".join(f"{v:.1%}" for v in values) + " |"
        )
    report += [
        "",
        "Each percentage uses 160 independent simulated observations with 1,000 posterior draws each. The nominal 68% binomial standard error is 3.7 percentage points. calibration.csv contains exact intervals and marginal rank tests. The three parameters are strongly correlated, so their coverage estimates are dependent. Two MCMC comparisons and marginal rank histograms leave joint and prior-boundary calibration unresolved. Figures synthetic_0.png, synthetic_1.png and real.png show paired parameter draws and posterior-predictive flux bands. The diagnostics JSON includes full correlation matrices.",
        "",
        "| Observation | Walker steps | Longest estimated τ | Acceptance | Corr(log density, b), MCMC / NPE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["synthetic_0", "synthetic_1", "real"]:
        z = d[name]
        report.append(
            f"| {name} | {z['walker_steps']} | {max(z['tau_steps']):.1f} | {z['acceptance']:.2f} | {z['corr_mcmc'][0][1]:.3f} / {z['corr_npe'][0][1]:.3f} |"
        )
    report += [
        "",
        f"The real SAP best fit gives χ² = {estimates['SAP']['chi2']:.1f} for 57 nominal degrees of freedom. Trend estimation and astrophysical noise can correlate the bins. Replacing SAP with identically sliced PDCSAP gives a conditional best-fit density of {10 ** estimates['PDCSAP']['theta'][0]:.0f} kg/m³ versus {10 ** t[0]:.0f} kg/m³ for SAP. These are likelihood optima. The pipeline makes no assertion that PDCSAP improves the density estimate.",
        "",
        "## Computational cost and next experiment",
        "",
    ]
    tt = d["timings"]
    report += [
        f"On this host, simulating the 2,000 observations took {tt['simulate_s']:.2f} s, fitting the training/validation embeddings took {tt['embed_train_val_s']:.1f} s, and neural training took {tt['training_s']:.2f} s. A new observation and 1,000 NPE draws averaged {tt['npe_per_1000_draws_s']:.3f} s. The matched real-data MCMC took {tt['real_mcmc_s']:.1f} s. These timings exclude archive I/O and process startup and do not forecast the cost of a ten-parameter or cross-KOI model.",
        "",
        "First test recovery with synthetic transits injected into actual out-of-transit SAP segments, processed from scratch. Compare the unbinned likelihood with per-event polynomial profiling to the folded likelihood. Vary eccentricity, limb darkening and quarter-level dilution separately. Repeat calibration by impact parameter and S/N on fresh test simulations. Then extend to KOI-97.01 using its independent seismic reference and a mixed-cadence simulator. A cross-target model will need conditioning on period, exposure, noise, phase coverage and eccentricity information with a star-disjoint evaluation.",
    ]
    (ROOT / "reports/pilot.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    evaluate()
