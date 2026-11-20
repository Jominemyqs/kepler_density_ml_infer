# Stellar density from Kepler transits

A target-conditional NPE pilot for KOI-929.01, with a host-aware reconstruction of the Sandford & Kipping sample. Start with [the archive audit](reports/data_provenance.md), [target selection](reports/target_selection.md) and [the pilot results](reports/pilot.md).

## Reproduce

Run from this directory with CPython 3.12, curl and the pinned packages. This snapshot ran on macOS arm64. Downloads reuse local source files. For an independent new archive snapshot, run the scripts in a new copy containing the code and empty output directories. Keep dated snapshots separate.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
mkdir -p data/{catalog,metadata,processed,raw} reports results/data_quality
python scripts/inventory.py
python scripts/audit.py
python scripts/product_stats.py
python scripts/tess.py
python scripts/verification.py
python scripts/download.py
python scripts/photometry.py
python scripts/verification.py
python scripts/pilot.py
python scripts/evaluate.py
python -m unittest discover -s tests -v
```

The second verification pass records the primary's FITS inspection in the audit. Archive requests are metadata-only until download.py. Running inventory or audit alone regenerates base tables. Run the enrichment steps afterward. Every real photometric measurement used here originates in a raw FITS file. The model checkpoint is target-specific and requires the matching folded.npz grid and simulator.

|Path|Contents|
|---|---|
|data/catalog|Two analysis-ready CSVs keyed by exact KOI and KIC|
|data/metadata|Machine-readable paper tables, posterior tree and evaluation chain, NASA TAP schemas and query responses, MAST directory listings and product URLs, VizieR tables, TIC/TESS snapshots|
|data/raw/9141746|Unmodified Kepler files and retrieval metadata with SHA-256|
|data/processed|Compressed unbinned segment table and 60-bin observations with errors, counts, phase quadrature weights and fixed model inputs|
|results/data_quality|Raw light curve, segment and fold figures, quarter statistics, accepted and rejected event centers|
|results/pilot|Simulation splits, best validation checkpoint, held-out posterior draws and ranks, matched chains and draws, numerical diagnostics and predictive plots|
|scripts, tests|Archive access, extraction, physical model, conditional density estimation, matched emcee fitting, evaluation and checks|

In segments.csv.gz, bkjd and phase_days are barycentric days and time from transit center. epoch is the integer transit index, cadence the original CADENCENO, quality the original bit field. sap, sap_err and trend are electrons/s. norm, norm_err and pdc_norm are dimensionless; degree is the selected SAP polynomial order. The complete raw tables retain the rejected fluxes, quality flags and calibrated metadata. folded.npz uses days for phase/grid/cadence/period/t0. flux, pdc and error are normalized units. grid and weights are four within-bin phase centroids and their normalized inverse-variance weights.
