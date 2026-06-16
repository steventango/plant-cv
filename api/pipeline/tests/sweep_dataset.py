"""Compare SAM3 plant-propagation configs across a whole dataset.

For each config we run the full frame sequence and, per pot, record the plant
mask pixel-area trajectory. We then surface:
  - healthy growth (area rising over time), and
  - REGRESSIONS: pots whose area drops sharply frame-to-frame (mask collapse /
    leak correction), which is the failure mode reported for larger plants.

Run inside the pipeline container:
  docker compose run --rm pipeline uv run python tests/sweep_dataset.py
Env: DATASET (default E14Z1)
"""

import base64
import os
from pathlib import Path

import cv2
import numpy as np
import requests

DATASET = os.environ.get("DATASET", "E14Z1")
FRAMES_DIR = Path(__file__).parent / "test_data" / "tracking_frames" / DATASET


def _base():
    for host in ["pipeline", "localhost"]:
        try:
            requests.get(f"http://{host}:8800/health", timeout=2)
            return f"http://{host}:8800/pipeline"
        except Exception:
            continue
    return "http://localhost:8800/pipeline"


BASE = _base()
DETECT = f"{BASE}/detect"
PROPAGATE = f"{BASE}/propagate"


def encode(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def poly_area(contour):
    if not contour:
        return 0
    pts = np.array(contour, dtype=np.int32)
    if pts.ndim != 2 or len(pts) < 3:
        return 0
    return cv2.contourArea(pts)


# Grid: reconditioning frequency (n) x memory depth (keep), recond_on_trk=True.
CONFIGS = {
    f"n={n} keep={k}": {
        "recondition_on_trk_masks": True,
        "recondition_every_nth_frame": n,
        "prune_keep_frames": k,
    }
    for k in (2, 4, 8)
    for n in (1, 2, 3)
}


def run_config(overrides, frames):
    """Return {pot_id: [area_per_frame]}."""
    state = None
    pot_areas = {}
    for j, fp in enumerate(frames):
        img_b64 = encode(fp)
        if state is None:
            payload = {"image_data": img_b64, "debug_raw_plants": True}
            r = requests.post(DETECT, json=payload, timeout=300)
        else:
            payload = {
                "image_data": img_b64,
                "state": state,
                "debug_raw_plants": True,
                "plant_params": overrides,
            }
            r = requests.post(PROPAGATE, json=payload, timeout=300)
        if r.status_code != 200:
            print(f"  FAIL f{j}: {r.text[:200]}")
            return pot_areas
        res = r.json()
        state = res.get("state")
        assoc = res.get("associations", {})  # plant_id -> pot_id
        raw = {m["object_id"]: m for m in res.get("debug_raw_plant_masks", [])}
        for pid_str, pot_id in assoc.items():
            m = raw.get(int(pid_str))
            area = poly_area(m.get("contour")) if m else 0
            pot_areas.setdefault(int(pot_id), [None] * len(frames))[j] = area
    return pot_areas


def summarize(pot_areas):
    """Return (n_tracked, n_growing, n_vanished). A plant 'vanishes' if it once
    reached a sizable area but is 0 for the final frames."""
    n_grow = n_vanish = 0
    for areas in pot_areas.values():
        peak = max((a for a in areas if a), default=0)
        if len([a for a in areas if a]) < 3:
            continue
        ended_zero = len([a for a in areas[-3:] if a]) == 0
        if peak > 3000 and ended_zero:
            n_vanish += 1
        first = next((a for a in areas if a), 0)
        last = next((a for a in reversed(areas) if a), 0)
        if last > first * 1.3:
            n_grow += 1
    return len(pot_areas), n_grow, n_vanish


def main():
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    target_pot = int(os.environ["TARGET_POT"]) if os.environ.get("TARGET_POT") else None
    print(f"Dataset {DATASET}: {len(frames)} frames\n")

    rows = []
    results = {}
    for name, overrides in CONFIGS.items():
        pot_areas = run_config(overrides, frames)
        results[name] = pot_areas
        tracked, grow, vanish = summarize(pot_areas)
        rows.append((name, tracked, grow, vanish))
        print(f"{name:14s} tracked={tracked:3d} growing={grow:3d} vanished={vanish:3d}")

    print(f"\n{'config':14s} {'tracked':>7} {'growing':>7} {'vanished':>8}")
    for name, tracked, grow, vanish in rows:
        print(f"{name:14s} {tracked:7d} {grow:7d} {vanish:8d}")

    # Plants present (non-zero area) per frame — exposes a last-frame dropout.
    print("\n=== plants present per frame (count with area>0) ===")
    for name in CONFIGS:
        pa = results[name]
        per_frame = [
            sum(1 for areas in pa.values() if areas[j]) for j in range(len(frames))
        ]
        print(f"  {name:14s}: " + " ".join(f"{c:3d}" for c in per_frame))

    if target_pot is not None:
        print(f"\n=== pot {target_pot} area trajectory per config ===")
        for name in CONFIGS:
            seq = results[name].get(target_pot, [])
            seq_str = " ".join(f"{(v if v else 0):5.0f}" for v in seq)
            print(f"  {name:14s}: {seq_str}")


if __name__ == "__main__":
    main()
