import os

sizes = [150, 500, 1000, 2000, 5000, 10000]


def get_path(n: int, idx: int):
    return os.path.join("..", "data", f"matrix-{n}-{idx}.float32")
