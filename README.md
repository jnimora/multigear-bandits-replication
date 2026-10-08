# Multi-gear bandits

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23230777.svg)](https://doi.org/10.5281/zenodo.23230777)

Algorithms and replication material for José Niño-Mora, *Fast parametric solution and index policies for structured Markov decision processes: Multi-gear bandits*.

This repository provides explicit implementations of the adaptive-greedy downshift algorithms for finite-state, finite-action bandits, including positive-work candidate selection and in-loop MPI monotonicity checks. A supplied policy family controls the feasible active-set changes. Discounted and average criteria are supported under the conditions described in the paper. Completion of a numerical path is not a general proof of indexability outside those conditions.

This replication package contains the final experiments used in the manuscript. Earlier pilots and superseded campaigns are omitted. The ordinary BLAS blocked implementations are included; the paper's fast-matrix-multiplication complexity result is theoretical and is not implemented or timed here.

## Start here

Use Julia **1.13.0** for the frozen generators and numerical experiments. From the repository root:

```sh
julia --startup-file=no --threads=1 --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --threads=1 --project=. test/runtests.jl
julia --startup-file=no --threads=1 --project=. examples/quickstart.jl
```

To reproduce figures, tables and statistical checks without rerunning the experiments, use Python 3.12:

```sh
python3 -m venv .venv-report
.venv-report/bin/python -m pip install -r paper/requirements.txt
.venv-report/bin/python paper/reproduce.py
```

Outputs go to `results/local/paper/`, which is excluded from Git. Choose a new `--output` directory for another run; published observations are never overwritten. See [reproduction instructions](docs/REPRODUCIBILITY.md) for optional per-case experiments and [data contents](docs/DATA.md) for the retained evidence.

## Implementations

| Implementation | Entry point | Purpose |
|---|---|---|
| C, R, M and unblocked FP | `src/MultiGearBandits.jl` | General algorithms, references and controls |
| Blocked multi-gear FP | `scripts/blocked_multigear/blocked_fp.jl` | Deferred BLAS matrix updates; optional reference-row cache |
| Specialized blocked binary FP | `scripts/binary_benchmark/blocked_fp.jl` | Unit active work, zero passive work; unrestricted binary candidates |
| Tailored Rayleigh BLOCK | `scripts/rayleigh_block_support.jl` | Model-specific recursions |
| Multi-project allocation | `scripts/heterogeneity_model.jl`, `scripts/heterogeneity_simulation.jl` | Six policies and exact accounting |
| Gast–Gaujal–Khun comparator | `vendor/markovianbandit/` | Unchanged released Python implementation used in the benchmark |

The multi-gear blocked API is `BlockedMultiGearFP.prepare(input; family, criterion, ...)`, `run!(workspace)` and `snapshot(workspace)`. Its family types are `UnrestrictedFamily`, `OrderedThresholdFamily`, `RowThresholdFamily`, and `ExplicitPolicyFamily`. The core C/R/M/FP APIs and independent validators are exported by `MultiGearBandits`.

The unblocked implementations remain useful as correctness references and as the comparison methods reported in the paper. Some shared numerical utilities retain their original filenames; their inclusion does not reinstate an obsolete campaign.

## Citation

Cite the archived replication package: José Niño-Mora (2026), *Multi-gear bandits: algorithms and replication material*, version 1.0.0, Zenodo, [doi:10.5281/zenodo.23230777](https://doi.org/10.5281/zenodo.23230777).

The DOI identifies the exact [v1.0.0 release](https://github.com/jnimora/multigear-bandits-replication/releases/tag/v1.0.0) used for the manuscript. The release tag remains fixed; this branch may receive later documentation updates. `CITATION.cff` supplies machine-readable citation metadata. See [release procedure](docs/RELEASE.md).

Code is distributed under the [MIT license](LICENSE). The GGK comparator retains its original [license](vendor/markovianbandit/LICENSE) and [attribution](THIRD_PARTY_NOTICES.md).
