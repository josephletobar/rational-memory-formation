# Rational Memory Formation

Code for the Rational Episodic Memory experiments.

The repository is split into a small reusable pipeline and a larger archive
of research experiments:

- Root scripts contain the current feature caching, probe training, scoring,
  evaluation, and reusable visualization entry points.
- [`experiments/`](experiments/) contains exploratory model variants,
  robustness sweeps, rendering scripts, remote-job launchers, and legacy
  one-off analysis code.
- Dataset files, feature caches, model checkpoints, rendered videos, and other
  runtime outputs are kept outside Git.

The scripts generally expect to be launched from the repository root. Most
files in `experiments/` preserve paths used for the original study runs; they
are retained for reproducibility and are not presented as a supported public
API.
