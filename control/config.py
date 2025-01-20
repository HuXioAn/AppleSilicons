import os

sizes = [32, 64, 128]
# sizes = [2048, 4096, 8192, 16384, 32768, 65536]

enablePureCPU = True # for large matrix, baseline and omp can be disabled by setting this to False

def get_path(n: int, idx: int):
    return os.path.join("..", "data", f"matrix-{n}-{idx}.float32")
