import config
import numpy as np


for i in config.sizes:
    left = np.fromfile(config.get_path(i, 0), dtype=np.float32).reshape((i, i))
    print(left[0,:10])
    right = np.fromfile(config.get_path(i, 1), dtype=np.float32).reshape((i, i))
    result = left @ right
    print(f"{i}: check {result[0, 0]}")
