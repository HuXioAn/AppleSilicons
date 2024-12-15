import os

sizes = [150, 500, 1000, 1500, 2000, 2500]


def get_path(n: int, idx: int):
    return os.path.join("..", "data", f"matrix-{n}-{idx}.float32")
