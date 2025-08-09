from pathlib import Path
import os

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def find_file_ci(path):
    """Return path as str; convenience to support both Path and str inputs."""
    return str(path)