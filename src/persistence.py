"""
persistence.py

Saves each pipeline stage's results to data/ as JSON, so past runs don't
have to be repeated just to inspect their numbers -- most importantly for
run_h2_emulator.py, since re-running it spends metered qnexus quota.

Layout per stage (e.g. stage="h2_emulator"):
    data/h2_emulator_<UTC timestamp>.json   -- permanent, one per run
    data/h2_emulator_latest.json            -- overwritten each run, convenience pointer

Numpy arrays/scalars and non-string dict keys (e.g. float h-values) are
coerced into JSON-safe plain Python.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _to_jsonable(obj):
    """Recursively convert numpy types/arrays and dict keys into JSON-safe
    plain Python so json.dump doesn't choke on them."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_stage_results(stage: str, results, data_dir=DATA_DIR):
    """Write `results` to data/<stage>_<timestamp>.json and refresh
    data/<stage>_latest.json. Returns the timestamped file's path.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "stage": stage,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "results": _to_jsonable(results),
    }
    text = json.dumps(payload, indent=2)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = data_dir / f"{stage}_{timestamp}.json"
    path.write_text(text)
    (data_dir / f"{stage}_latest.json").write_text(text)

    print(f"[persistence] Saved {stage} results to {path}")
    return path


def load_latest(stage: str, data_dir=DATA_DIR):
    """Load the most recently saved payload for `stage`, or None if there
    isn't one yet. Returns the full payload dict (stage/saved_at/results)."""
    path = Path(data_dir) / f"{stage}_latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
