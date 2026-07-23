"""
fh_persistence.py

Tiny self-contained results store (JSON) so run scripts can save stage outputs
and fh_main / the plotting layer can reload them. Kept independent of the TFIM
project's persistence.py so this Fermi-Hubbard package is drop-in.

Data goes under ./data_fh/ next to the code by default.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_fh")


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def save_stage_results(stage: str, results: dict, data_dir: str = DATA_DIR) -> str:
    os.makedirs(data_dir, exist_ok=True)
    payload = {"stage": stage, "timestamp": time.time(), "results": results}
    path = os.path.join(data_dir, f"{stage}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_default)
    return path


def load_stage_results(stage: str, data_dir: str = DATA_DIR):
    path = os.path.join(data_dir, f"{stage}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["results"]