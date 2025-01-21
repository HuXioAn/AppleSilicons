import os

sizes = [32, 64, 128, 150, 256, 500, 512, 1000, 1024, 1500, 2000, 2048, 2500, 4096]
# sizes = [2048, 4096, 8192, 16384]

enablePureCPU = True # for large matrix, baseline and omp can be disabled by setting this to False

def get_path(n: int, idx: int):
    return os.path.join("..", "data", f"matrix-{n}-{idx}.float32")
