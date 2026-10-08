# Reproduction

## 1. Reanalyse the published observations

Install `paper/requirements.txt` under Python 3.12 and run `python paper/reproduce.py`. This performs no timing campaign or simulation. It verifies the replication summaries, the multi-gear fits and whole-level holdouts, and the binary timing medians and fits, then generates the current figures and timing tables. Saved observations remain unchanged; all output is written to a fresh directory.

The principal plots have ordinary runtime axes, distinct grayscale markers/line styles, and fitted operation-count curves. Phase timings and end-to-end timings remain separate. The allocation population-scaling diagnostic retains logarithmic axes.

## 2. Verify regenerated inputs

The public runner lists cases without starting computation:

```sh
python3 scripts/reproduce/run.py binary --list
python3 scripts/reproduce/run.py multigear --list
python3 scripts/reproduce/run.py block --list
python3 scripts/reproduce/run.py allocation --list
```

Select one case and generate/hash its dense input without running a timing experiment:

```sh
python3 scripts/reproduce/run.py binary --case exponential_n1000_d5_disc --check-input
python3 scripts/reproduce/run.py multigear --case channel_N512_B2_A4_p1_average --check-input
```

Use `--julia /path/to/julia` or the `JULIA` environment variable if Julia 1.13.0 is not on PATH. Input verification creates a new per-case output directory. Specify a different `--output` base for subsequent runs.

## 3. Optional fresh experiments

Omit `--check-input` to rerun a selected case, for example:

```sh
python3 scripts/reproduce/run.py multigear --case channel_N512_B2_A4_p1_average --output results/local/new-multigear
python3 scripts/reproduce/run.py block --case N1025_A1_p1 --output results/local/new-block
```

The multi-gear runner reconstructs a fresh FP reference and audits it at three policies before timing; each blocked method receives its own audits and output comparison. Historical unresolved cases remain in the catalog and are not silently turned into timing observations. A numerical failure must remain an exclusion, not a zero runtime.

For the binary comparison, install the benchmark dependencies under Python 3.12:

```sh
python3 -m venv .venv-benchmark
.venv-benchmark/bin/python -m pip install -r scripts/binary_benchmark/requirements.txt
.venv-benchmark/bin/python scripts/reproduce/run.py binary --case exponential_n1000_d5_disc --output results/local/new-binary
```

The wrapper uses the vendored, unchanged GGK 0.4 implementation. Fresh PCL path audits precede timing. Input generation, loading, validation and Python compilation are outside the reported algorithm timings. The original worker records compilation and validity flags. Inspect the result status before treating a rerun as eligible. Binary matrices are deleted after a successful run unless `--keep-input` is supplied.

Allocation jobs can be rerun individually with `--case` using the frozen catalog. The wrapper rebuilds the exact class solutions and reruns the specified windows with their original seeds; it does not require archived serialized objects. This is deliberately a simple, sequential reproduction route and can be slower than the original shared-cache campaign. Large jobs can take hours. A failed or truncated trajectory is never an accepted replication.

Running every listed case is an optional substantial computation. The runner never launches the whole campaign by default. Large dense cases require several GiB of RAM. The published timings used an Apple M1 Max Mac Studio with 64 GiB RAM, one Julia thread and one BLAS thread. Expect machine-dependent timings, not identical seconds. Clock diagnostics support 64-bit macOS and Linux; they explicitly reject other platforms. On macOS the wrapper prevents idle sleep for each worker's lifetime; on Linux configure the host's sleep settings for long runs.

## Numerical scope

Tests cover independent Bellman/reference solves, positive-work filtering, MPI monotonicity, multiple feasible-family representations, and blocked-tableau identities. The model-specific family and validation assumptions in the paper still apply. No output flag is a claim of arbitrary-model PCL-indexability.

The dense C/R/M/FP algorithms and the blocking algebra use ordinary numerical linear algebra. No Strassen or other fast-matrix-multiplication implementation is included.

The standalone tailored BLOCK worker reports `complete_provisional` after its internal checks; the original 24 published cases additionally retain their cross-method validation. A fresh per-case BLOCK rerun does not repeat that whole comparison campaign.
