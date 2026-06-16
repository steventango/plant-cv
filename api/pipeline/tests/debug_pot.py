"""Debug tooling: trace the plant inside a target POT across frames.

Pot IDs are stable (row-major id_map); the plant associated with a pot can be
re-issued by SAM3, so we follow the pot and inspect whichever plant maps to it.

For each frame we record + visualise:
  - which plant_id associates to the target pot
  - the RAW SAM3 plant contour (green) vs the Otsu-REFINED contour (red)
    overlaid on a crop, so under-segmentation is obvious
  - raw vs refined pixel areas, and the reported (calibrated) area

Run inside the pipeline container:
  docker compose run --rm pipeline uv run python tests/debug_pot.py
Env: TARGET_POT (default 34), DATASET (default E16Z1)
"""

import base64
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

TARGET_POT = int(os.environ.get("TARGET_POT", "34"))
DATASET = os.environ.get("DATASET", "E16Z1")
# Optional JSON dict of SAM3 plant-param overrides, e.g. PLANT_PARAMS='{"det_nms_thresh":0.5}'
PLANT_PARAMS = json.loads(os.environ.get("PLANT_PARAMS", "{}"))

FRAMES_DIR = Path(__file__).parent / "test_data" / "tracking_frames" / DATASET
OUT_DIR = Path(__file__).parent / "test_output" / "debug" / f"{DATASET}_pot{TARGET_POT}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def draw_contour(img, contour, color, thickness=2):
    if not contour:
        return
    pts = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def main():
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"Tracking POT {TARGET_POT} across {len(frames)} frames\n")

    state = None
    summary = []
    last_box = None  # crop region around target pot

    for j, fp in enumerate(frames):
        img_b64 = encode(fp)
        endpoint = DETECT if state is None else PROPAGATE
        payload = {"image_data": img_b64, "debug_raw_plants": True}
        if state is not None:
            payload["state"] = state
            if PLANT_PARAMS:
                payload["plant_params"] = PLANT_PARAMS
        r = requests.post(endpoint, json=payload, timeout=300)
        if r.status_code != 200:
            print(f"Frame {j} FAILED: {r.text[:500]}")
            break

        res = r.json()
        state = res.get("state")

        assoc = res.get("associations", {})  # plant_id(str) -> pot_id(int)
        plant_masks = res.get("plant_masks", [])
        pot_masks = res.get("pot_masks", [])
        plant_stats = res.get("plant_stats", {})
        raw = res.get("debug_raw_plant_masks", [])

        # plant associated to target pot
        plant_id = None
        for pid_str, pot_id in assoc.items():
            if int(pot_id) == TARGET_POT:
                plant_id = int(pid_str)
                break

        pot_obj = next(
            (m for m in pot_masks if int(m.get("object_id")) == TARGET_POT), None
        )
        if pot_obj and pot_obj.get("box"):
            last_box = pot_obj["box"]

        refined = next((m for m in plant_masks if m.get("object_id") == plant_id), None)
        raw_obj = next((m for m in raw if m.get("object_id") == plant_id), None)

        refined_contour = None
        if refined:
            refined_contour = refined.get("contour")
        raw_contour = raw_obj.get("contour") if raw_obj else None

        raw_px = poly_area(raw_contour)
        refined_px = poly_area(refined_contour)
        stat = plant_stats.get(str(TARGET_POT)) or {}
        reported_area = stat.get("area")
        clean_area = stat.get("clean_area")

        shrink = ""
        if raw_px > 0:
            ratio = refined_px / raw_px
            shrink = f" refined/raw={ratio:.2f}"

        rec = {
            "frame": j,
            "file": fp.name,
            "plant_id": plant_id,
            "raw_px_area": raw_px,
            "refined_px_area": refined_px,
            "reported_area_mm2": reported_area,
            "clean_area": clean_area,
        }
        summary.append(rec)
        print(
            f"[f{j}] plant_id={plant_id} raw_px={raw_px:.0f} refined_px={refined_px:.0f}"
            f"{shrink} reported={reported_area} clean={clean_area}"
        )

        # Overlay crop: raw SAM3 (green) vs refined (red)
        if last_box:
            img = np.array(
                Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cx = (last_box[0] + last_box[2]) / 2
            cy = (last_box[1] + last_box[3]) / 2
            half = 160
            x0, y0 = int(cx - half), int(cy - half)
            x1, y1 = int(cx + half), int(cy + half)
            draw_contour(img, raw_contour, (0, 255, 0))  # green raw
            draw_contour(img, refined_contour, (0, 0, 255), 1)  # red refined
            crop = img[max(0, y0) : y1, max(0, x0) : x1]
            cv2.putText(
                crop,
                f"f{j} raw={raw_px:.0f} ref={refined_px:.0f}",
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imwrite(str(OUT_DIR / f"f{j:02d}_pot{TARGET_POT}.jpg"), crop)

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nGreen=raw SAM3 mask, Red=Otsu-refined mask. Crops -> {OUT_DIR}")


if __name__ == "__main__":
    main()
