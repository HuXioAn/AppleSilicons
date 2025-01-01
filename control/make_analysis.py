import os
import pandas as pd

RESULTS_FOLDER = "../results"
RAW_DATA_FOLDER = "../out"

implementations = ["baseline", "omp", "blas", "dsp", "gpu_baseline", "gpu_mps", "gpu_nv"]


def setup():
    os.system(f"mkdir -p {RESULTS_FOLDER}")


def get_powers(matrix_size, trial, implementation):
    eff, pwr, gpu = [], [], []

    extract = lambda row: int(row.split(" ")[2])
    with open(RAW_DATA_FOLDER + f"/{matrix_size}x{matrix_size}/{trial}/{implementation}/power.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("E-Cluster Power"):
                eff.append(extract(line))
            elif line.startswith("P-Cluster Power"):
                pwr.append(extract(line))
            elif line.startswith("GPU Power"):
                gpu.append(extract(line))
    print(eff, pwr, gpu)


def analyze():
    get_powers(500, 1, "baseline")


def plot():
    pass


if __name__ == "__main__":
    setup()
    analyze()
    plot()
