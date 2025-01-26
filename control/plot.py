import os
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rcParams["font.family"] = "serif"

SoC = ["M1", "M2", "M3", "M4"]

sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

RESULTS_FOLDER = "results"
RAW_DATA_FOLDER = "out"

implementations = ["baseline", "omp"]

labels = ["CPU-Single", "CPU-OMP"] 

formats = [None, ".", "*", "x", "-", "/", "o", "+", "&", "v", "^", "<", ">"]



def setup(path):
    os.system(f"mkdir -p {path}")


def get_powers(matrix_size, trial, implementation):
    cpu, pwr, gpu = None, 0, None

    extract = lambda row: int(row.split(" ")[2])
    with open(RAW_DATA_FOLDER + f"/{matrix_size}x{matrix_size}/{trial}/{implementation}/power.txt", "r") as f:
        for line in f.readlines():
            if line.startswith("CPU Power"):
                cpu = int(extract(line))
            # elif line.startswith("P-Cluster Power"):
            #     pwr = int(extract(line))
            elif line.startswith("GPU Power"):
                gpu = int(extract(line))
    return cpu, pwr, gpu


def get_powers_data(implementation):
    raw = [[get_powers(size, trial, implementation) for trial in range(1, 6)] for size in sizes]
    processed = np.array(raw)
    return processed


def get_timing(matrix_size, trial, implementation):
    with open(RAW_DATA_FOLDER + f"/{matrix_size}x{matrix_size}/{trial}/{implementation}/timing.txt", "r") as f:
        return int(f.read()) / 1_000_000  # time in ms


def get_timing_data(implementation):
    raw = [[get_timing(size, trial, implementation) for trial in range(1, 6)] for size in sizes]
    processed = np.array(raw)
    return np.mean(processed, axis=1), np.std(processed, axis=1)


def plot_timing_implementation(implementation, label, fmt, color):
    y, err = get_timing_data(implementation)
    plt.errorbar(sizes, y, yerr=err, fmt=fmt, color=color, label=label)
    print(label, "&", " & ".join([f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(y, err)]), "\\\\")


def plot_timing():
    plt.cla()
    plt.figure(figsize=(7, 9))
    print("TIMING [ms]")

    plot_timing_implementation("baseline", "Naive", "o", "C0")
    plot_timing_implementation("omp", "Block Multiplication", "s", "C3")
    plot_timing_implementation("blas", "BLAS", "v", "C1")
    plot_timing_implementation("dsp", "vDSP", "^", "C2")
    plot_timing_implementation("gpu_baseline", "Naive Shader", "D", "C4")
    plot_timing_implementation("gpu_nv", "Cutlass-Style Shader", "h", "C5")
    plot_timing_implementation("gpu_mps", "MPS", "*", "C6")
    plt.xlabel("Matrix size")
    plt.xticks(sizes)
    plt.ylabel("Average multiplication time [ms], log scale, lower is better")
    plt.yscale("log")
    plt.legend(title="Implementation")
    plt.savefig(RESULTS_FOLDER + "/timing.png", dpi=300)


def plot_power_implementation(implementation, label, fmt, color):
    all_data = np.sum(get_powers_data(implementation), axis=2)
    y = np.mean(all_data, axis=1)
    err = np.std(all_data, axis=1)
    plt.errorbar(sizes, y, yerr=err, fmt=fmt, color=color, label=label)
    print(label, "&", " & ".join([f"{m:.2f} & {s:.2f}" for m, s in zip(y, err)]), "\\\\")


def plot_power():
    plt.cla()
    print("\n\nPOWER [mW]")

    plot_power_implementation("baseline", "Naive", "o", "C0")
    plot_power_implementation("omp", "Block Multiplication", "s", "C3")
    plot_power_implementation("blas", "BLAS", "v", "C1")
    plot_power_implementation("dsp", "vDSP", "^", "C2")
    plot_power_implementation("gpu_baseline", "Naive Shader", "D", "C4")
    plot_power_implementation("gpu_nv", "Cutlass-Style Shader", "h", "C5")
    plot_power_implementation("gpu_mps", "MPS", "*", "C6")
    plt.xlabel("Matrix size")
    plt.xticks(sizes)
    plt.ylabel("Average power dissipation [mW], lower is better")
    plt.legend(title="Implementation")
    plt.savefig(RESULTS_FOLDER + "/power.png", dpi=300)


def plot_granular_values(matrix_size):
    raw = [[get_powers(matrix_size, i, implementation) for i in range(1, 6)] for implementation in implementations]
    processed = np.array(raw)
    return processed[:, :, 0], processed[:, :, 1], processed[:, :, 2]


def plot_granular(matrix_size):
    plt.cla()
    print("\n\nGRANULAR [mW]", matrix_size)
    cpu, _, gpu = plot_granular_values(matrix_size)
    cpu_y, gpu_y = np.mean(cpu, axis=1), np.mean(gpu, axis=1)
    cpu_err, gpu_err = np.std(cpu, axis=1), np.std(gpu, axis=1)
    cpu_err_lower = np.clip(cpu_err, None, cpu_y)
    gpu_err_lower = np.clip(gpu_err, None, gpu_y)
    for i, implementation in enumerate(implementations):
        print(labels[i], "&",
              "%.2f" % cpu_y[i], "&", "%.2f" % cpu_err[i], "&",
              "%.2f" % gpu_y[i], "&", "%.2f" % gpu_err[i], "\\\\")

    n_categories = len(implementations)
    group_height = 0.6
    bar_height = group_height / 3
    category_spacing = 0.4
    y_positions = np.arange(n_categories) * (group_height + category_spacing)
    y_positions = y_positions[::-1]
    y_positions_eff = y_positions + bar_height / 2
    y_positions_pwr = y_positions
    y_positions_gpu = y_positions - bar_height / 2

    plt.figure(figsize=(13, 6))
    plt.barh(y_positions_eff, cpu_y, xerr=[cpu_err_lower, cpu_err], height=bar_height,
             label="CPU", hatch=None)
    
    plt.barh(y_positions_gpu, gpu_y, xerr=[gpu_err_lower, gpu_err], height=bar_height,
             label="GPU", hatch="|")
    plt.xlabel("Average power dissipation per component [mW], lower is better")
    plt.ylabel("Implementation")
    plt.yticks(y_positions, labels)
    plt.legend(title="Components", loc="lower right")
    plt.savefig(RESULTS_FOLDER + f"/granular-{matrix_size}x{matrix_size}.png", dpi=300)


def num_flop(matrix_size):
    return matrix_size * matrix_size * ((2 * matrix_size) - 1)


def plot_flop_values():
    raw = [[[get_powers(matrix_size, i, implementation) for i in range(1, 6)]
            for matrix_size in sizes] for implementation in implementations]
    watts = np.sum(np.array(raw), axis=3) / 1_000  # conversion: milli watt to watt
    flops = np.array([num_flop(matrix_size) for matrix_size in sizes])[:, np.newaxis] / 1_000_000_000
    flops_per_watt = flops / watts
    return np.mean(flops_per_watt, axis=2), np.std(flops_per_watt, axis=2)


def plot_flop():
    plt.cla()
    print("\n\nGFLOP per W")
    components = [f"{n}x{n}" for n in sizes]
    y, y_err = plot_flop_values()

    num_categories = len(labels)
    num_components = len(components)
    bar_width = 0.12
    group_spacing = 0.6
    component_spacing = bar_width
    x_base = np.arange(num_categories) * (num_components * bar_width + group_spacing)
    x_positions = [x_base + i * component_spacing for i in range(num_components)]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, component in enumerate(components):
        ax.bar(x_positions[i], y[:, i], yerr=y_err[:, i], width=bar_width, label=component, hatch=formats[i])
    for i in range(len(implementations)):
        print(labels[i], "&", " & ".join([f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(y[i, :], y_err[i, :])]), "\\\\")

    ax.set_xlabel("Implementations")
    ax.set_ylabel("GFLOP per W")
    ax.set_xticks(x_base + (num_components - 1) * bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend(title="Matrix Size", loc="upper left")
    plt.savefig(RESULTS_FOLDER + "/efficiency.png", dpi=300)


def calculate_flops():
    plt.cla()
    print("\n\nGFLOPS")
    components = [f"{n}x{n}" for n in sizes]

    num_categories = len(labels)
    num_components = len(components)
    bar_width = 0.12
    group_spacing = 0.6
    component_spacing = bar_width
    x_base = np.arange(num_categories) * (num_components * bar_width + group_spacing)
    x_positions = [x_base + i * component_spacing for i in range(num_components)]
    fig, ax = plt.subplots(figsize=(12, 6))

    y = None
    y_err = None

    for i, implementation in enumerate(implementations):
        raw = [[get_timing(size, trial, implementation) for trial in range(1, 6)] for size in sizes]
        times = np.array(raw) / 1_000  # conversion: seconds
        flop = np.array([num_flop(matrix_size) for matrix_size in sizes]) / 1_000_000_000  # conversion: giga
        flops = flop[:, np.newaxis] / times
        mean = np.mean(flops, axis=1) 
        std = np.std(flops, axis=1)
        y = np.vstack([y, mean]) if y is not None else mean
        y_err = np.vstack([y_err, std]) if y_err is not None else std

        print(labels[i], "&", " & ".join([f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(mean, std)]), "\\\\")

    for i, component in enumerate(components):
        ax.bar(x_positions[i], y[:, i], yerr=y_err[:, i], width=bar_width, label=component, hatch=formats[i])

    ax.set_xlabel("Implementations")
    ax.set_ylabel("GFLOPS")
    ax.set_xticks(x_base + (num_components - 1) * bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend(title="Matrix Size", loc="upper left")
    plt.savefig(RESULTS_FOLDER + "/GFLOPS.png", dpi=300)


def calculate_flops_allchips_cpu(chipPath, size):

    implementations = ["baseline", "omp"]
    labels = ["CPU-Single", "CPU-OMP"] 

    print("\n\nGFLOPS-all")
    component = f"{size}x{size}" 

    flop = num_flop(size) / 1_000_000_000  # conversion: giga

    getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000  # time in ms

    results = []
    for i, chip in enumerate(SoC):
        chip_results = []
        for j, implementation in enumerate(implementations):
            raw = [getTime(chipPath[i], size, trial, implementation) for trial in range(1, 6)] 
            times = np.array(raw) / 1_000  # conversion: seconds
            flops = np.full(5, flop) / times

            mean = np.mean(flops, axis=0) 
            std = np.std(flops, axis=0)
            chip_results.append((mean, std))

        results.append(chip_results)

    bar_width = 0.3
    fig, axs = plt.subplots(len(SoC), 1, figsize=(6, 6), sharex=True)
    handles = []

    for i, (chip, ax) in enumerate(zip(SoC, axs)):
        means = [results[i][j][0] for j in range(len(implementations))]
        stds = [results[i][j][1] for j in range(len(implementations))]

        # Plot horizontal bar chart with hatches
        y_positions = np.arange(len(implementations))
        for j, (mean, std) in enumerate(zip(means, stds)):
            bar = ax.barh(
                y_positions[len(y_positions)-j - 1], mean, xerr=std, height=bar_width, alpha=0.8, 
                label=labels[j] if i == 0 else None, hatch=formats[j]
            )
            if i == 0:  # Collect legend handles only for the first subplot
                handles.append(bar)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        ax.set_title(f"{chip}", fontsize=10, loc="left")

        if i == len(SoC) - 1:
            ax.set_xlabel(f"GFLOPS-{size}x{size}", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        handles=[h[0] for h in handles],
        labels=labels,
        loc="upper right", ncol=1,  frameon=True)
    

    # Global adjustments
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/GFLOPS-{component}-cpu.png", dpi=300)
    plt.show()

def calculate_flops_allchips_acc(chipPath, size):

    implementations = ["blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    labels = ["CPU-Accelerate", "GPU-Naive", "GPU-CUTLASS", "GPU-MPS"] 

    print("\n\nGFLOPS-all")
    component = f"{size}x{size}" 

    flop = num_flop(size) / 1_000_000_000  # conversion: giga

    getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000  # time in ms

    results = []
    for i, chip in enumerate(SoC):
        chip_results = []
        for j, implementation in enumerate(implementations):
            raw = [getTime(chipPath[i], size, trial, implementation) for trial in range(1, 6)] 
            times = np.array(raw) / 1_000  # conversion: seconds
            flops = np.full(5, flop) / times

            mean = np.mean(flops, axis=0) 
            std = np.std(flops, axis=0)
            chip_results.append((mean, std))

        results.append(chip_results)

    bar_width = 0.3
    fig, axs = plt.subplots(len(SoC), 1, figsize=(6, 6), sharex=True)
    handles = []

    for i, (chip, ax) in enumerate(zip(SoC, axs)):
        means = [results[i][j][0] for j in range(len(implementations))]
        stds = [results[i][j][1] for j in range(len(implementations))]

        # Plot horizontal bar chart with hatches
        y_positions = np.arange(len(implementations))
        for j, (mean, std) in enumerate(zip(means, stds)):
            bar = ax.barh(
                y_positions[len(y_positions)-j - 1], mean, xerr=std, height=bar_width, alpha=0.8, 
                label=labels[j] if i == 0 else None, hatch=formats[j]
            )
            if i == 0:  # Collect legend handles only for the first subplot
                handles.append(bar)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([])
        ax.set_title(f"{chip}", fontsize=10, loc="left")

        if i == len(SoC) - 1:
            ax.set_xlabel(f"GFLOPS-{size}x{size}", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        handles=[h[0] for h in handles],
        labels=labels,
        loc="upper right", ncol=1,  frameon=True)
    

    # Global adjustments
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/GFLOPS-{component}-ACC.png", dpi=300)
    plt.show()

def calculate_flops_allchips_best(chipPath, sizes):

    implementations = ["blas", "gpu_baseline", "gpu_nv", "gpu_mps"]
    labels = ["CPU-Accelerate", "GPU-Naive", "GPU-CUTLASS", "GPU-MPS"] 

    print("\n\nGFLOPS-all")

    # flop = num_flop(size) / 1_000_000_000  # conversion: giga

    getTime = lambda path, size, trial, implementation: int(open(f"{path}/out/{size}x{size}/{trial}/{implementation}/timing.txt", "r").read()) / 1_000_000  # time in ms

    results = [] # different cjips
    for i, chip in enumerate(SoC):
        chip_results = [] # different implementations
        for j, implementation in enumerate(implementations):
            best = [] # best size
            for k, size in enumerate(sizes):
                flop = num_flop(size) / 1_000_000_000
                raw = [getTime(chipPath[i], size, trial, implementation) for trial in range(1, 6)] 
                times = np.array(raw) / 1_000  # conversion: seconds
                flops = np.full(5, flop) / times

                mean = np.mean(flops, axis=0) 
                std = np.std(flops, axis=0)

                if len(best) == 0 or mean > best[0]:
                    best = (mean, std, size)
                
            chip_results.append(best)

        results.append(chip_results)

    bar_width = 0.3
    fig, axs = plt.subplots(len(SoC), 1, figsize=(6, 6), sharex=True)
    handles = []

    for i, (chip, ax) in enumerate(zip(SoC, axs)):
        means = [results[i][j][0] for j in range(len(implementations))]
        stds = [results[i][j][1] for j in range(len(implementations))]
        best_sizes = [results[i][j][2] for j in range(len(implementations))]

        # Plot horizontal bar chart with hatches
        y_positions = np.arange(len(implementations))
        for j, (mean, std) in enumerate(zip(means, stds)):
            bar = ax.barh(
                y_positions[len(y_positions)-j - 1], mean, xerr=std, height=bar_width, alpha=0.8, 
                label=labels[j] if i == 0 else None, hatch=formats[j]
            )
            if i == 0:  # Collect legend handles only for the first subplot
                handles.append(bar)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(best_sizes)
        ax.set_title(f"{chip}", fontsize=10, loc="left")

        if i == len(SoC) - 1:
            ax.set_xlabel(f"GFLOPS-Best", fontsize=12)

        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(
        handles=[h[0] for h in handles],
        labels=labels,
        loc="upper right", ncol=1,  frameon=True)
    

    # Global adjustments
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FOLDER}/GFLOPS-Best-acc.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    # read argument 
    if len(sys.argv) < 2:
        print("Usage: python plot.py data path")
        sys.exit(1)
    path = sys.argv[1]

    chipPath = [os.path.join(path, chip) for chip in SoC]

    RESULTS_FOLDER = os.path.join(path, RESULTS_FOLDER)
    setup(RESULTS_FOLDER)
    # plot_timing()
    # plot_power()
    # for ms in sizes:
    #     plot_granular(ms)
    # plot_flop()
    # calculate_flops()
    calculate_flops_allchips_acc(chipPath, 8192)
    calculate_flops_allchips_cpu(chipPath, 1024)
    calculate_flops_allchips_best(chipPath, sizes)
