import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import batman
from archive import ROOT, rows
from inventory import koi
from photometry import offset, bkjd
from model import Simulator, scaled_axis, G, LOW, HIGH
from pilot import draw, Estimator, coordinates, physical, OUT
import torch


class PipelineTests(unittest.TestCase):
    def test_identifiers(self):
        self.assertEqual(koi("929.01"), "K00929.01")
        self.assertNotEqual(koi("929.01"), koi("929.02"))
        with self.assertRaises(ValueError):
            koi("929.1")
        r = rows(ROOT / "data/catalog/current_data_audit.csv")
        self.assertEqual(
            (len({x["koi"] for x in r}), len({x["kic"] for x in r})), (75, 66)
        )
        for x in r:
            self.assertEqual(x["cum_kepid"], x["kic"])

    def test_times(self):
        h = dict(
            TIMESYS="TDB",
            TIMEUNIT="d",
            TIMEREF="SOLARSYSTEM",
            BJDREFI=2454832,
            BJDREFF=0.5,
        )
        self.assertEqual(bkjd(h, 123.5), 123.0)
        np.testing.assert_allclose(offset(np.array([13.0, 16.0, 19.0]), 13.0, 3.0), 0.0)
        with self.assertRaises(AssertionError):
            bkjd(dict(h, TIMESYS="UTC"), 1.0)

    def test_physics(self):
        s = Simulator()
        a = scaled_axis(3.0, s.p.per)
        self.assertAlmostEqual(3 * np.pi * a**3 / (G * (86400 * s.p.per) ** 2), 1000.0)
        with self.assertRaises(ValueError):
            s.curve([3.0, 1.5, 0.08])
        flux = s.curve([3.0, 0.5, 0.08])
        self.assertTrue(0.98 < min(flux) < 0.998 and np.all(flux <= 1.0000001))
        self.assertAlmostEqual(flux[0], 1.0)
        self.assertAlmostEqual(s.weights.sum(axis=1).min(), 1.0)
        np.testing.assert_allclose(physical(coordinates(LOW, s.p.u), s.p.u), LOW)

    def test_finite_exposure(self):
        s = Simulator()
        flux = s.curve([2.65, 0.625, 0.081])
        dt = float(np.load(ROOT / "data/processed/folded.npz")["cadence"])
        reference = batman.TransitModel(
            s.p, s.grid.ravel(), exp_time=dt * 200 / 201, supersample_factor=201
        )
        exact = (reference.light_curve(s.p).reshape(s.grid.shape) * s.weights).sum(
            axis=1
        )
        self.assertLess(np.max(abs(flux - exact) / s.error), 0.05)

    def test_simulated_noise(self):
        s = Simulator()
        t, x = s.simulate(np.random.default_rng(12), 40)
        noise = np.stack([(xx - s.curve(tt)) / s.error for xx, tt in zip(x, t)])
        self.assertEqual(noise.shape, (40, 60))
        self.assertTrue(0.90 < noise.std() < 1.10 and abs(noise.mean()) < 0.1)
        self.assertTrue(np.all((t >= LOW) & (t <= HIGH)))

    def test_posterior(self):
        s = Simulator()
        net = Estimator()
        net.load_state_dict(torch.load(OUT / "checkpoint.pt"))
        out = draw(net, s, s.flux, np.random.default_rng(13), 111)
        self.assertEqual(out.shape, (111, 3))
        self.assertTrue(np.all((out >= LOW) & (out <= HIGH)))
        held = np.load(OUT / "heldout_posteriors.npz")
        self.assertEqual(held["samples"].shape, (160, 1000, 3))


if __name__ == "__main__":
    unittest.main()
