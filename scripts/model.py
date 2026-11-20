"""Three-parameter, target-conditional finite-integration forward model."""

import batman
import numpy as np
from archive import ROOT

LOW = np.array([2.1, 0.15, 0.055])
HIGH = np.array([3.4, 0.85, 0.12])
G = 6.67430e-11


def scaled_axis(logrho, period):
    return (G * 10**logrho * (period * 86400) ** 2 / (3 * np.pi)) ** (1 / 3)


class Simulator:
    def __init__(self):
        d = np.load(ROOT / "data/processed/folded.npz")
        self.grid = d["grid"]
        self.weights = d["weights"]
        self.error = d["error"]
        self.flux = d["flux"]
        self.phase = d["phase"]
        p = batman.TransitParams()
        p.per = float(d["period"])
        p.t0 = 0.0
        p.ecc = 0.0
        p.w = 90.0
        p.limb_dark = "quadratic"
        p.u = d["ld"].tolist()
        p.a = 10.0
        p.inc = 88.0
        p.rp = 0.08
        self.p = p
        # batman includes endpoints. Shorten the node span for midpoint quadrature
        # of the full cadence-wide boxcar (17 equally weighted subexposures).
        self.model = batman.TransitModel(
            p,
            self.grid.ravel(),
            supersample_factor=17,
            exp_time=float(d["cadence"]) * 16 / 17,
        )

    def curve(self, theta):
        logrho, b, r = theta
        p = self.p
        a = scaled_axis(logrho, p.per)
        if not (0 < r < 1 and 0 <= b < 1 + r < a):
            raise ValueError("Invalid transit geometry")
        p.a = a
        p.inc = np.rad2deg(np.arccos(b / a))
        p.rp = r
        return (self.model.light_curve(p).reshape(self.grid.shape) * self.weights).sum(
            axis=1
        )

    def simulate(self, rng, n):
        theta = rng.uniform(LOW, HIGH, (n, 3))
        return theta, np.array(
            [self.curve(t) + rng.normal(0, self.error) for t in theta]
        )
