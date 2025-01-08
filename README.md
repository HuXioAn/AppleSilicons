# Apple M1 Matrix Multiplication Benchmarks

Repository for software for a research project aiming to answer the research question:
> What is the computational performance (in FLOP/s) and power usage (in W) of single-precision matrix-matrix multiplication on an Apple M1 SoC using pure CPU, CPU accelerators, and GPU?

Kind thanks to the open-source repositories supporting this research:
* Some Metal shaders from: https://github.com/bkvogel/metal_performance_testing
* The OpenMP block multiplication implementation from: https://github.com/dmitrydonchenko/Block-Matrix-Multiplication-OpenMP

## Prerequisites

This project requires Xcode, C++20 and `libomp`. The following software and versions have been found to work:
| Name | Version | Installation |
| --- | --- | --- |
| macOS | 12.6.7 on M1 Air | |
| CMake | 3.28 | |
| Make | 3.81 | |
| Xcode | 14.0.1 | via Apple Developer website |
| Metal | 31001.643 | bundled with Xcode |
| libomp | 19.1.4 | `brew install libomp` |

## Compilation

1. `mkdir build && cd build/`
2. `cmake -DMATRIX_N=<N> -DCMAKE_BUILD_TYPE=Release ..`, replace `<N>` with a number such that the files `./data/matrix-<N>-0.float32` and `./data/matrix-<N>-1.float32` exist
3. `make`

## Execution

For CPU-based builds, cwd into `build` and execute the binary.
For GPU-based builds, cwd into `build/XXX.app/Contents/MacOS` (where `XXX` is the name of an executable) and execute the binary.

## Benchmarking

The benchmarking can be performed automatically using the `./control` suite.
This test suite uses Python, a virtual environment with Python 3.9.6 and the packages in `./control/requirements.txt` works.

### Running the Suite

Ensure to create the `./out` and `./results` directories.
Enter the virtual environment, cwd into `./control` and run `sudo python3 benchmark.py`. 
Superuser privileges are **required** as `powermetrics` requires them.

### Analysing the Results.

Enter the virtual environment, cwd into `./control` and run `python3 make_analysis.py`.
This will create graphs inside the `./results` directory.
LaTeX tabular data will also be printed to stdout.