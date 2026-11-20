# KOI-929.01 pilot

The conditional NPE and matched emcee fit were both run on a 60-bin folded SAP observation. The synthetic tests support local agreement at two test cases. Marginal coverage shows overcoverage at nominal 68%, and this target-specific pilot cannot establish reliability across stars.

## Generative model

Independent uniform priors in log10 density in kg/m³ [2.1, 3.4], b [0.15, 0.85] and Rp/Rstar [0.055, 0.12]. This is a restricted non-grazing giant-planet experiment. The prior is a pilot design choice. Each simulation draws fresh physical parameters from this box. Period 6.491682808 d and epoch 174.633553 BKJD are fixed at the documented catalog ephemeris. Set e = 0 and ω = 90° (irrelevant for a circle). The quadratic Kepler-band limb coefficients (0.3661, 0.2844) are fixed at the two KOI catalog values. Assume negligible planet mass, no dilution, TTVs or correlated noise.

Use a/Rstar = [G ρ (86400 P)²/(3π)]^(1/3), SI units and cos(i) = b/(a/Rstar). Reject geometries that intersect the star on orbit or lack a transit. Batman integrates over a cadence-wide boxcar of 0.0204335982 d with 17 equal midpoint samples. The time span passed to batman accounts for its endpoint-inclusive sampling. Four weighted sub-bin phase centroids approximate each bin. Gaussian noise is independently drawn with the same per-bin formal uncertainty used in the MCMC likelihood.

Against 201 midpoint evaluations at every original cadence phase, the combined quadrature/binning approximation changes any bin by at most 0.035 formal σ at the SAP best fit. Numerical integration matters here. A preliminary 11-point endpoint average over the full cadence returned a best-fit density of 530 kg/m³; the checked midpoint calculation yields 449 kg/m³. All reported samples and checkpoints use the midpoint calculation.

## Estimator and reference

The compact PyTorch model uses a bounded nonlinear least-squares transit fit as its observation embedding. It adds ten mean standardized residuals and uses the local Jacobian for whitening. In order, the autoregressive variables are b, log10ρ − 1.5 log10(1−b²), and radius ratio times the square root of the local quadratic limb intensity at the transit chord. These coordinates reduce curvature in the duration/depth degeneracy. A 32-unit hidden layer predicts residual locations and scales. Each conditional normal is truncated at the transformed physical bounds given previously drawn coordinates. Training minimizes normalized conditional negative log likelihood. The physical-coordinate Jacobian is constant with respect to the network weights for each training pair. Neural inference time includes fitting the observation.

Training/validation/test RNG seeds 731/732/733 generate 1,600/240/160 independent simulations. Main torch, NumPy and posterior seed 730. Select the best validation checkpoint over 50 epochs. Only simulated parameters and fluxes train the network. The real data supply the observing design, bin errors and fixed ephemeris. Published samples enter only after inference. Use checkpoint.pt with Estimator, Simulator and draw in scripts/pilot.py.

For each reference, emcee uses the same uniform box, batman mean function, fixed inputs, binning and diagonal Gaussian likelihood. There are 24 walkers, 400 discarded steps and thinning by four for storage. Run length adapts to estimated autocorrelation time subject to a cap. Walkers start near a least-squares fit. The chains check this restricted problem and cannot rule out separated modes in the wider published model.

## Central 68% intervals

| Observation | Fit | Density (kg/m³) | b | Radius ratio |
|---|---|---|---|---|
| synthetic_0 | injected | 146.5 | 0.713 | 0.09034 |
| synthetic_0 | mcmc | 149.0 [145.0, 153.4] | 0.709 [0.702, 0.715] | 0.09023 [0.09000, 0.09046] |
| synthetic_0 | npe | 149.2 [144.8, 153.6] | 0.708 [0.701, 0.716] | 0.09022 [0.08998, 0.09048] |
| synthetic_1 | injected | 852.6 | 0.303 | 0.07841 |
| synthetic_1 | mcmc | 846.2 [760.0, 917.8] | 0.307 [0.209, 0.397] | 0.07848 [0.07804, 0.07909] |
| synthetic_1 | npe | 865.1 [775.0, 921.6] | 0.285 [0.202, 0.382] | 0.07837 [0.07801, 0.07899] |
| real | mcmc | 454.6 [424.3, 488.8] | 0.623 [0.598, 0.645] | 0.08120 [0.08073, 0.08165] |
| real | npe | 457.6 [423.2, 492.1] | 0.621 [0.596, 0.646] | 0.08117 [0.08070, 0.08168] |
| published | SK 2017 | 521.4 [456.1, 601.8] | 0.574 [0.518, 0.620] | 0.08376 [0.08252, 0.08493] |

Density quantiles are taken in physical units after exponentiating each draw. The published posterior contains 88,467 rows. Its 16th/50th/84th percentiles imply −12.5%/+15.4% in physical density. The geometry screen flags 83 samples with b > 1 + Rp/Rstar. The reference quantiles above include all rows. The paper sampled ten parameters for this planet, used eclipse priors and a flexible limb law, and applied a radius-ratio cutoff for confirmed planets. Its broader parameter space and different detrending both matter. The radius ratio shift here persists even when the matched estimators agree.

## Calibration and fit quality

| Nominal interval | log density covered | b covered | Radius ratio covered |
|---|---:|---:|---:|
| 0.5 | 56.9% | 55.6% | 57.5% |
| 0.68 | 75.0% | 78.1% | 79.4% |
| 0.9 | 91.2% | 90.0% | 94.4% |
| 0.95 | 95.6% | 95.6% | 97.5% |

Each percentage uses 160 independent simulated observations with 1,000 posterior draws each. The nominal 68% binomial standard error is 3.7 percentage points. calibration.csv contains exact intervals and marginal rank tests. The three parameters are strongly correlated, so their coverage estimates are dependent. Two MCMC comparisons and marginal rank histograms leave joint and prior-boundary calibration unresolved. Figures synthetic_0.png, synthetic_1.png and real.png show paired parameter draws and posterior-predictive flux bands. The diagnostics JSON includes full correlation matrices.

| Observation | Walker steps | Longest estimated τ | Acceptance | Corr(log density, b), MCMC / NPE |
|---|---:|---:|---:|---:|
| synthetic_0 | 2800 | 36.6 | 0.65 | -0.992 / -0.994 |
| synthetic_1 | 18400 | 301.8 | 0.43 | -0.984 / -0.976 |
| real | 4000 | 57.3 | 0.61 | -0.996 / -0.996 |

The real SAP best fit gives χ² = 68.2 for 57 nominal degrees of freedom. Trend estimation and astrophysical noise can correlate the bins. Replacing SAP with identically sliced PDCSAP gives a conditional best-fit density of 454 kg/m³ versus 449 kg/m³ for SAP. These are likelihood optima. The pipeline makes no assertion that PDCSAP improves the density estimate.

## Computational cost and next experiment

On this host, simulating the 2,000 observations took 0.33 s, fitting the training/validation embeddings took 24.6 s, and neural training took 0.48 s. A new observation and 1,000 NPE draws averaged 0.015 s. The matched real-data MCMC took 16.1 s. These timings exclude archive I/O and process startup and do not forecast the cost of a ten-parameter or cross-KOI model.

First test recovery with synthetic transits injected into actual out-of-transit SAP segments, processed from scratch. Compare the unbinned likelihood with per-event polynomial profiling to the folded likelihood. Vary eccentricity, limb darkening and quarter-level dilution separately. Repeat calibration by impact parameter and S/N on fresh test simulations. Then extend to KOI-97.01 using its independent seismic reference and a mixed-cadence simulator. A cross-target model will need conditioning on period, exposure, noise, phase coverage and eccentricity information with a star-disjoint evaluation.
