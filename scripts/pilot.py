"""Conditional autoregressive truncated-normal NPE with a local transit-fit embedding."""

import json
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri
import emcee
import torch
from torch import nn
import matplotlib.pyplot as plt
from archive import ROOT, META
from model import Simulator, LOW, HIGH

OUT = ROOT / "results/pilot"
SEED = 730
torch.set_num_threads(1)


def coordinates(t, ld):
    rho, b, r = t.T
    mu = np.sqrt(1 - b * b)
    i = 1 - ld[0] * (1 - mu) - ld[1] * (1 - mu) ** 2
    return np.stack((b, rho - 1.5 * np.log10(1 - b * b), r * np.sqrt(i)), axis=-1)


def physical(z, ld):
    b, d, k = z.T
    mu = np.sqrt(1 - b * b)
    return np.stack(
        (
            d + 1.5 * np.log10(1 - b * b),
            b,
            k / np.sqrt(1 - ld[0] * (1 - mu) - ld[1] * (1 - mu) ** 2),
        ),
        axis=-1,
    )


def feature(s, x):
    fit = least_squares(
        lambda t: (s.curve(t) - x) / s.error,
        (LOW + HIGH) / 2,
        bounds=(LOW, HIGH),
        ftol=1e-7,
        xtol=1e-7,
    )
    t = fit.x
    z = coordinates(t, s.p.u)
    dz = np.column_stack(
        [
            (coordinates(t + v, s.p.u) - coordinates(t - v, s.p.u)) / 2e-5
            for v in np.eye(3) * 1e-5
        ]
    )
    dz_jinv = dz @ np.linalg.pinv(fit.jac)
    L = np.linalg.cholesky(dz_jinv @ dz_jinv.T + np.eye(3) * 1e-16)
    return (
        z,
        L,
        np.r_[2 * (t - LOW) / (HIGH - LOW) - 1, fit.fun.reshape(10, 6).mean(axis=1)],
    )


def bounds(b, ld):
    b = np.atleast_1d(b)
    a = np.broadcast_to(LOW, (len(b), 3)).copy()
    a[:, 1] = b
    c = np.broadcast_to(HIGH, a.shape).copy()
    c[:, 1] = b
    lo = coordinates(a, ld)
    hi = coordinates(c, ld)
    lo[:, 0] = LOW[1]
    hi[:, 0] = HIGH[1]
    return lo, hi


def batch(s, ts, xs):
    xs_in = []
    targets = []
    lower = []
    upper = []
    for t, x in zip(ts, xs):
        z, L, context = feature(s, x)
        target = solve_triangular(L, coordinates(t, s.p.u) - z, lower=True)
        lo, hi = bounds(t[1], s.p.u)
        center = z + (L - np.diag(L.diagonal())) @ target
        xs_in.append(context)
        targets.append(target)
        lower.append((lo[0] - center) / L.diagonal())
        upper.append((hi[0] - center) / L.diagonal())
    return [
        torch.tensor(np.array(a), dtype=torch.float32)
        for a in [xs_in, targets, lower, upper]
    ]


class Estimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, 32), nn.Tanh(), nn.Linear(32, 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        m, logscale = self.net(x).chunk(2, dim=-1)
        return 3 * m.tanh(), torch.exp(1.5 * logscale.tanh())


def loss(net, data):
    x, y, lo, hi = data
    m, scale = net(x)
    norm = (
        torch.special.ndtr((hi - m) / scale) - torch.special.ndtr((lo - m) / scale)
    ).clamp_min(1e-10)
    return (
        (
            0.5 * ((y - m) / scale) ** 2
            + scale.log()
            + norm.log()
            + 0.5 * np.log(2 * np.pi)
        )
        .sum(-1)
        .mean()
    )


def draw(net, s, x, rng, n):
    z, L, c = feature(s, x)
    with torch.no_grad():
        m, scale = [t.numpy() for t in net(torch.tensor(c, dtype=torch.float32))]
    y = np.zeros((n, 3))
    coords = np.zeros_like(y)
    for j in range(3):
        lo, hi = bounds(np.full(n, 0.5) if j == 0 else coords[:, 0], s.p.u)
        center = z[j] + y[:, :j] @ L[j, :j]
        a, b = [
            ndtr(((v[:, j] - center) / L[j, j] - m[j]) / scale[j]) for v in (lo, hi)
        ]
        p = np.clip(rng.uniform(a, b), 1e-15, 1 - 1e-15)
        y[:, j] = m[j] + scale[j] * ndtri(p)
        coords[:, j] = center + L[j, j] * y[:, j]
    return physical(coords, s.p.u)


def mcmc(s, x, rng):
    def logp(t):
        if np.any(t <= LOW) or np.any(t >= HIGH):
            return -np.inf
        return -0.5 * np.sum(((s.curve(t) - x) / s.error) ** 2)

    fit = least_squares(
        lambda t: (s.curve(t) - x) / s.error, (LOW + HIGH) / 2, bounds=(LOW, HIGH)
    )
    init = np.clip(
        fit.x + rng.normal(size=(24, 3)) * (HIGH - LOW) * 0.003, LOW + 1e-7, HIGH - 1e-7
    )
    chain = emcee.EnsembleSampler(24, 3, logp)
    chain.run_mcmc(init, 1600)
    for _ in range(20):
        if chain.iteration - 400 > 55 * max(chain.get_autocorr_time(tol=0)):
            break
        chain.run_mcmc(None, 1200)
    return (
        chain.get_chain(discard=400, thin=4, flat=True),
        chain.get_autocorr_time(tol=0).tolist(),
        float(np.mean(chain.acceptance_fraction)),
        chain.iteration,
    )


def quant(draws):
    return np.quantile(
        np.column_stack((draws, 10 ** draws[:, 0])), [0.16, 0.5, 0.84], axis=0
    ).tolist()


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    s = Simulator()
    times = {}
    t = perf_counter()
    splits = [
        s.simulate(np.random.default_rng(k), n)
        for k, n in [(731, 1600), (732, 240), (733, 160)]
    ]
    times["simulate_s"] = perf_counter() - t
    np.savez_compressed(
        OUT / "simulations.npz",
        **{
            name + "_" + k: v
            for name, pair in zip(["train", "validation", "test"], splits)
            for k, v in zip(["theta", "flux"], pair)
        },
    )
    t = perf_counter()
    train, val = [batch(s, *pair) for pair in splits[:2]]
    times["embed_train_val_s"] = perf_counter() - t
    net = Estimator()
    opt = torch.optim.AdamW(net.parameters(), lr=0.001)
    history = []
    best = np.inf
    t = perf_counter()
    for epoch in range(50):
        for i in torch.randperm(len(train[0])).split(64):
            opt.zero_grad()
            l = loss(net, [v[i] for v in train])
            l.backward()
            opt.step()
        with torch.no_grad():
            history.append([float(loss(net, train)), float(loss(net, val))])
        if history[-1][1] < best:
            best = history[-1][1]
            torch.save(net.state_dict(), OUT / "checkpoint.pt")
    times["training_s"] = perf_counter() - t
    net.load_state_dict(torch.load(OUT / "checkpoint.pt"))
    theta, x = splits[-1]
    t = perf_counter()
    draws = np.stack([draw(net, s, xx, rng, 1000) for xx in x])
    times["npe_per_1000_draws_s"] = (perf_counter() - t) / len(x)
    ranks = (draws < theta[:, None, :]).mean(axis=1)
    report = {
        "seed": SEED,
        "columns": ["log10rho", "b", "radius_ratio", "rho_kg_m3"],
        "quantile_order": [0.16, 0.5, 0.84],
        "coverage": {
            str(p): ((ranks > (1 - p) / 2) & (ranks < (1 + p) / 2)).mean(0).tolist()
            for p in [0.5, 0.68, 0.9, 0.95]
        },
        "timings": times,
    }
    np.savez_compressed(
        OUT / "heldout_posteriors.npz", samples=draws, truth=theta, ranks=ranks
    )
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    ax[0].plot(history)
    ax[0].legend(["Train", "Validation"])
    ax[0].set(xlabel="Epoch", ylabel="Conditional NLL (transformed space)")
    for j, label in enumerate(["log density", "b", "radius ratio"]):
        ax[1].hist(
            ranks[:, j], bins=np.linspace(0, 1, 11), histtype="step", label=label
        )
    ax[1].set(xlabel="Held-out rank fraction", ylabel="Count")
    ax[1].legend()
    fig.savefig(OUT / "learning_calibration.png", dpi=150)
    plt.close(fig)
    for tag, xx in [("synthetic_0", x[0]), ("synthetic_1", x[1]), ("real", s.flux)]:
        t = perf_counter()
        ref, tau, accept, steps = mcmc(s, xx, rng)
        times[tag + "_mcmc_s"] = perf_counter() - t
        npe = draw(net, s, xx, rng, 4000)
        np.savez_compressed(OUT / (tag + ".npz"), mcmc=ref, npe=npe, flux=xx)
        report[tag] = dict(
            mcmc=quant(ref),
            npe=quant(npe),
            tau_steps=tau,
            walker_steps=steps,
            acceptance=accept,
            corr_mcmc=np.corrcoef(ref.T).tolist(),
            corr_npe=np.corrcoef(npe.T).tolist(),
        )
        fig, ax = plt.subplots(1, 4, figsize=(14, 3.4), layout="constrained")
        for a, (i, j) in zip(ax[:3], [(0, 1), (0, 2), (1, 2)]):
            for samples, label in [(ref, "MCMC"), (npe, "NPE")]:
                a.scatter(
                    samples[::10, i], samples[::10, j], s=2, alpha=0.25, label=label
                )
            a.set(xlabel=report["columns"][i], ylabel=report["columns"][j])
        ax[0].legend(markerscale=2)
        ax[3].errorbar(s.phase * 24, xx, yerr=s.error, fmt=".", ms=3, color="black")
        for samples, color in [(ref, "C0"), (npe, "C1")]:
            pred = np.array(
                [s.curve(v) + rng.normal(0, s.error) for v in samples[:300]]
            )
            l, h = np.quantile(pred, [0.025, 0.975], axis=0)
            ax[3].fill_between(s.phase * 24, l, h, color=color, alpha=0.35)
        ax[3].set(xlabel="Hours from midtransit", ylabel="Flux, 95% predictive bands")
        fig.savefig(OUT / (tag + ".png"), dpi=150)
        plt.close(fig)
    pub = np.loadtxt(META / "K00929.01_samples.txt", skiprows=1)[:, [3, 2, 4]]
    report["published"] = quant(pub)
    (OUT / "diagnostics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
