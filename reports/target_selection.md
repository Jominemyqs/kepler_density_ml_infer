# First target

Selection emphasizes the ability to identify the same cadence and inspect a worked fit, followed by external density information, eclipse constraints, transit strength, and geometry. All four candidates are confirmed occultation targets with published chains and Gaia–Kepler stellar records. None is one of the paper's nine jointly fitted pairs. S/N and transit counts below come from the cumulative catalog fit. Counts depend on the pipeline cuts. File counts measure availability, not usable events.

|Rank | KOI (KIC) | LC quarters / SC files | S/N | Catalog transits | Published b | Decision |
|---|---|---:|---:|---:|---:|---|
|1|929.01 (9141746)|15 / 0|502|189|0.57|Figs. 3–5 provide LC-only photometric and posterior checks, including 183 retained events. Moderate impact, near-zero eclipse components. Mask the second planet, 929.02.|
|2|97.01 (5780885)|18 / 18|2590|277|0.53|Best independent validation case, with a Huber seismic comparison in Fig. 7. An LC-only fit omits SC information available to the authors. Berger lists one Gaia neighbor within 4 arcsec, so dilution deserves a separate check.|
|3|186.01 (12019440)|17 / 13|1586|413|0.08|Fig. 8 demonstrates excellent limb coverage. Near-central transits have weak sensitivity to small changes in b. More slices and mixed cadence increase replication cost.|
|4|17.01 (10874614)|15 / 24|3004|335|0.15|Strong depth and a single catalog KOI. Good extension for finite-exposure tests with SC. Missing quarters and low b complicate the first comparison.|

All are practical on a binned grid. Eclipse-derived priors constrain eccentricity but leave uncertainty. A circular fit is a conditional experiment. The current Planetary Systems default entries do not supply a measured eccentricity for these four. See the reference-specific PS extract for other fits; a reported zero with no errors can be an assumption.

Defer the nine pairs until the simulator supports a shared density. Exclude current false positives from planet benchmarks. KOIs 5157, 7430 and both members of 153 carry explicit author warnings; 1793, 4351 and 3913 have bimodal solutions; 1 and 823 illustrate limb-darkening pathologies. These remain in the audit. KOI-5.01 also combines a weak eclipse prior with a low-S/N grazing solution. No shorter catalog error bar was used as a selection criterion.

For the primary, Shabram et al. Table 1 measures e cos ω to about 0.034 and e sin ω to about 0.047. The catalog count and S/N refer to 929.01 alone. Its companion has S/N 10.1 and depth 115 ppm. Six later TESS time-series records are available for the host. Neither the companion nor TESS is part of the original single-KOI posterior. The second-ranked target has a measured seismic density of 245 ± 15 kg/m³, making it the preferred next external validation. [Eclipse constraints](https://doi.org/10.3847/0004-637X/820/2/93) · [Seismic reference](https://doi.org/10.1088/0004-637X/767/2/127)

Berger reports RUWE 0.9383 and zero Gaia DR2 companions within 4 arcsec for KIC 9141746. These checks support its use as an external comparator. The Gaia neighbor count concerns resolved stellar sources and is distinct from the two transiting planets.
