import os
import subprocess
import config
import signal
import time
import dotenv

dotenv.load_dotenv()

# The path of the binaries for cmake and make.
CMAKE_PATH = os.environ["CMAKE_PATH"]
MAKE_PATH = os.environ["MAKE_PATH"]
# The project root, relative from both the control and bin directory.
PROJECT_ROOT = "../"
BIN_ROOT = PROJECT_ROOT + "cmake-build-release"
OUT_ROOT = PROJECT_ROOT + "out"
TRIALS = 5
SAMPLE_RATE = 5

if os.geteuid() != 0:
    raise RuntimeError("script must be run as root, powermetrics requires it")


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
] if config.enablePureCPU else [
    Implementation(".", "blas"),
    Implementation(".", "dsp"),
    Implementation("./gpu_baseline.app/Contents/MacOS", "gpu_baseline"),
    Implementation("./gpu_mps.app/Contents/MacOS", "gpu_mps"),
    Implementation("./gpu_nv.app/Contents/MacOS", "gpu_nv"),
]


def compile_binaries(mat_size, reset_dir_path):
    os.chdir(reset_dir_path)
    os.system(f"rm -rf {BIN_ROOT} || true")
    os.system(f"mkdir -p {BIN_ROOT}")
    os.chdir(BIN_ROOT)
    os.system(f"\"{CMAKE_PATH}\" -DMATRIX_N={mat_size} -DCMAKE_BUILD_TYPE=Release ..")
    os.system(f"\"{MAKE_PATH}\"")


def start_monitor(output_file):
    command = [
        "powermetrics",
        "-i", "0",  # only sample when process notifies
        "-a", "0",  # do not aggregate
        "-s", "cpu_power,gpu_power",
        "-o", output_file
    ]
    proc = subprocess.Popen(command)
    time.sleep(2)  # wait for the program to be running
    return proc


def stop_monitor(mon):
    mon.send_signal(signal.SIGIO)  # just in case, flush to file
    mon.send_signal(signal.SIGINT)
    mon.wait()


def run():
    checkpoint_path = os.getcwd()
    for _, matrix_size in enumerate(config.sizes):
        print("Matrix size:", matrix_size)
        compile_binaries(matrix_size, checkpoint_path)
        for i in range(1, TRIALS + 1):
            trial_dir = os.path.abspath(OUT_ROOT + f"/{matrix_size}x{matrix_size}/" + str(i))
            os.system(f"mkdir -p {trial_dir}")
            for impl in implementations:
                cwd_dir = os.path.abspath(BIN_ROOT + "/" + impl.launch_dir)
                log_dir = trial_dir + "/" + impl.launch_bin
                os.system(f"mkdir -p {log_dir}")
                timing_file = log_dir + "/timing.txt"
                power_file = os.path.abspath(log_dir + "/power.txt")
                mon = start_monitor(power_file)
                mon_pid = mon.pid
                work_env = os.environ.copy()
                work_env["POWER_MONITOR"] = str(mon_pid)
                with open(timing_file, "w") as timing:
                    subprocess.run(
                        [os.path.abspath(cwd_dir + "/" + impl.launch_bin)],
                        cwd=cwd_dir,
                        env=work_env,
                        stdout=timing,
                        stderr=subprocess.STDOUT,
                        check=True
                    )
                    stop_monitor(mon)


if __name__ == "__main__":
    run()
