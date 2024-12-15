import os
import subprocess
import config

# The path of the binaries for cmake and make.
CMAKE_PATH = os.environ["CMAKE_PATH"]
MAKE_PATH = os.environ["MAKE_PATH"]
# The project root, relative from both the control and bin directory.
PROJECT_ROOT = "../"
BIN_ROOT = PROJECT_ROOT + "cmake-build-release"
OUT_ROOT = PROJECT_ROOT + "out"
TRIALS = 5


class Implementation:
    def __init__(self, launch_dir, launch_bin):
        self.launch_dir = launch_dir
        self.launch_bin = launch_bin


implementations = [
    Implementation(".", "baseline"),
    Implementation(".", "omp"),
    Implementation(".", "blas"),
    Implementation(".", "dsp"),
    Implementation("./gpu_baseline.app/Contents/MacOS", "gpu_baseline"),
    Implementation("./gpu_mps.app/Contents/MacOS", "gpu_mps"),
    Implementation("./gpu_nv.app/Contents/MacOS", "gpu_nv"),
]


def compile_binaries(mat_size):
    os.system(f"rm -rf {BIN_ROOT} || true")
    os.system(f"mkdir -p {BIN_ROOT}")
    os.chdir(BIN_ROOT)
    os.system(f"\"{CMAKE_PATH}\" -DMATRIX_N={mat_size} -DCMAKE_BUILD_TYPE=Release ..")
    os.system(f"\"{MAKE_PATH}\"")


def run():
    for matrix_size in config.sizes:
        print("Matrix size:", matrix_size)
        compile_binaries(matrix_size)
        for i in range(1, TRIALS + 1):
            trial_dir = os.path.abspath(OUT_ROOT + f"/{matrix_size}x{matrix_size}/" + str(i))
            os.system(f"mkdir -p {trial_dir}")
            for impl in implementations:
                cwd_dir = os.path.abspath(BIN_ROOT + "/" + impl.launch_dir)
                log_dir = trial_dir + "/" + impl.launch_bin
                os.system(f"mkdir -p {log_dir}")
                timing_dir = log_dir + "/timing.txt"
                with open(timing_dir, "w") as timing:
                    subprocess.run(
                        [os.path.abspath(cwd_dir + "/" + impl.launch_bin)],
                        cwd=cwd_dir,
                        stdout=timing,
                        stderr=subprocess.STDOUT,
                        check=True
                    )


if __name__ == "__main__":
    run()
