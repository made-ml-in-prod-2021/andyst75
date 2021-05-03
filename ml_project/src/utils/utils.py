import os
import hydra


def make_path(path: str) -> str:
    """
    Create path from root to target path (fix change path by hydra)
    """
    if os.path.isabs(path):
        return path
    root = hydra.utils.get_original_cwd()
    return os.path.join(root, path)
