# Third-party software

`vendor/markovianbandit/` contains the Python source of **markovianbandit-pkg 0.4**, as frozen in the binary benchmark. Copyright (c) 2021 Nicolas Gast and Kimang Khun; MIT license. The original notice is in `vendor/markovianbandit/LICENSE`. The three source files are unchanged from the benchmark snapshot. They implement the Gast–Gaujal–Khun comparison used in the manuscript.

The Python worker uses the original function for primary timings. A separately instrumented copy supplies phase diagnostics; these executions are not pooled with primary timings. Dependencies are listed in `scripts/binary_benchmark/requirements.txt`; their respective licenses remain applicable.
