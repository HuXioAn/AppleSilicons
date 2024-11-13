import config
import numpy as np
import os


def gen_and_save(n: int, idx: int):
    matrix = np.random.rand(n, n).astype("f")
    pathname = config.get_path(n, idx)
    matrix.tofile(pathname)


np.random.seed(42)
for i in config.sizes:
    gen_and_save(i, 0)
    gen_and_save(i, 1)
