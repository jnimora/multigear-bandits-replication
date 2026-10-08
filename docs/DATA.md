# Compact research evidence

The release keeps generators **and measured outcomes**. Seeds alone cannot recover the original running times, exclusions or simulation summaries. Dense synthetic inputs and transient solver workspaces are unnecessary distribution artifacts when their primitive specifications and checksums are retained.

| Cohort | Retained evidence | Primitive specification |
|---|---|---|
| Final blocked multi-gear benchmark, 6 October 2026 | 1,196 execution records; 296 method/case summaries; fits, holdouts and audit summaries | `configs/multigear_cases.json`: 90 cases, including two unresolved cases with no timings |
| Final blocked binary benchmark, 5 October 2026 | 3,200 scalar phase records from 800 end-to-end and 800 separate phase executions; 400 method/case summaries; fits | `configs/binary_cases.json`: 80 criterion cases using 40 seeded primitive inputs |
| Tailored Rayleigh BLOCK benchmark, 3 October 2026 | 24 cases, two repetitions each, counts and original validation summaries | `paper/block_raw_measurements.json` and `configs/rayleigh_reference_production.toml` |
| Multi-project allocation confirmation | 6,528 policy/window/replication observations, summaries, paired contrasts, sensitivity and scaling results | `configs/heterogeneity_confirmation_native.toml`: frozen jobs, windows and seeds; `configs/allocation_design.json` |

Allocation windows can overlap or reuse trajectories. They are not additional independent replications. Each reported group uses 32 independent replications with shared randomness across policies. The manuscript's two timing repetitions and its 32 simulation replications concern different experiments.

`paper/blocked_multigear_evidence/raw_scalar_measurements.json` preserves all attempts, including compilation-contaminated and surplus valid attempts. Summaries select the first two eligible executions of each kind. End-to-end and loop measurements come from separate executions. The binary file `measurements.json` stores initialization, loop and output as three records from each phase execution, not three independent executions.

## Formats and provenance

JSON and CSV contain compact numerical evidence; TOML contains Julia configurations. Times are seconds unless a plot label explicitly converts them to milliseconds. `N` is state count; `A` is the count of positive gears (actions are 0 through A); `B` is channel-state count. `profile` identifies a fixed parameter profile. Figure displays use Profile 1; both profiles remain in the evidence and fitted models.

Input SHA-256 hashes describe little-endian Float64 arrays in Julia's column-major order. Reproduce them with Julia 1.13.0 before interpreting a rerun as the same input. The explicit timing primitives are stored for multi-gear cases; binary matrices are generated using their original seeds. Allocation seeds identify the original transition and initial-state streams.

`provenance/` retains measured-source checksums and hardware metadata. The general numerical kernels match the final multi-gear campaign. The binary blocked source subsequently changed its default block size from 64 to 32 and its documentation; the measured worker already explicitly passed 32, so the executed setting is unchanged. Public launchers replace private cache loading with regeneration and fresh validation. They do not reproduce the original supervisors' elapsed campaign duration.

Saved timing fits are accompanied by raw scalars and independent arithmetic checks. Class-mechanism summaries have 13,056 class/replication observations in `raw_class_replications.csv`; their means, standard errors and resource shares are checked as well. Per-project trajectories and exact intermediate ledgers are not part of this compact release.

## Excluded material

Large transition arrays, simulation shock streams, serialized workspaces, temporary caches, machine logs, generated figures and PDFs, and superseded campaigns are excluded from Git. Generated matrices can be discarded after each successful rerun. The author's full local archives are preserved separately.

No Git LFS or external dataset download is required to reproduce the published plots and principal statistics. All source and evidence files in this candidate are small enough for ordinary Git. Larger optional archives, if ever needed, belong in a separate Zenodo dataset with an explicit relation to the software version.

Release CSV files use LF line endings. Original-source hashes describe the archived files before path redaction or line-ending normalization; `provenance/FILE_SHA256.json` describes the distributed files.
