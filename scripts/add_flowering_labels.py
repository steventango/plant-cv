#!/usr/bin/env python3
"""Add flowering labels to all.csv files.

Usage:
  python add_flowering_labels.py --root /data/online/E13/P1 --threshold-date 2025-10-24 --cutoff-date 2025-11-02

Adds a new column 'flowered_label' with boolean values based on whether the date parsed
from the 'image_name' field (first 10 characters, YYYY-MM-DD) is >= threshold date.
Drops rows where the date is after the cutoff date.
Operates on files matching pattern: <root>/*/*/processed/v5.0.0/all.csv
Creates a backup file alongside each original named all.csv.bak before overwriting.
Skips files that already contain 'flowered_label' unless --force is provided.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
from pathlib import Path
from typing import List

DEFAULT_PATTERN = "{root}/*/*/processed/v5.0.0/all.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/online/E13/P1", help="Root directory containing experiment data")
    p.add_argument("--threshold-date", default="2025-10-24", help="Date (YYYY-MM-DD) from which flowering is True")
    p.add_argument("--cutoff-date", default="2025-11-02", help="Date (YYYY-MM-DD) after which rows are dropped")
    p.add_argument("--dry-run", action="store_true", help="Show planned changes without writing files")
    p.add_argument("--force", action="store_true", help="Rewrite even if flowered_label already exists")
    return p.parse_args()


def find_files(root: str) -> List[str]:
    pattern = DEFAULT_PATTERN.format(root=root.rstrip("/"))
    return sorted(glob.glob(pattern))


def process_file(path: Path, threshold: dt.date, cutoff: dt.date, dry_run: bool, force: bool) -> None:
    with path.open("r", newline="") as rf:
        reader = csv.reader(rf)
        try:
            header = next(reader)
        except StopIteration:
            print(f"[SKIP] Empty file: {path}")
            return
        has_column = "flowered_label" in header
        if has_column and not force:
            print(f"[SKIP] Already labeled: {path}")
            return
        rows = list(reader)

    # Add/replace column
    if not has_column:
        header.append("flowered_label")
    else:
        # Replace existing values by truncating existing column values (we'll rebuild)
        flower_idx = header.index("flowered_label")
        for r in rows:
            if len(r) > flower_idx:
                r.pop(flower_idx)

    threshold_str = threshold.isoformat()
    cutoff_str = cutoff.isoformat()
    changed_true = 0
    changed_false = 0
    dropped = 0

    filtered_rows = []
    for r in rows:
        if not r:
            continue
        image_name = r[0]
        date_part = image_name[:10]  # YYYY-MM-DD
        try:
            img_date = dt.date.fromisoformat(date_part)
        except ValueError:
            # If parsing fails, mark False (conservative) and report
            label = False
        else:
            # Drop rows after cutoff date
            if img_date >= cutoff:
                dropped += 1
                continue
            label = img_date >= threshold
        if label:
            changed_true += 1
        else:
            changed_false += 1
        r.append("True" if label else "False")
        filtered_rows.append(r)

    print(f"[PROCESS] {path} -> True: {changed_true} False: {changed_false} Dropped: {dropped} (threshold {threshold_str}, cutoff {cutoff_str})")
    if dry_run:
        return

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        os.replace(path, backup)
    else:
        # If backup exists, just read from backup (we already moved original before?)
        # Ensure we are writing new file fresh.
        pass

    with path.open("w", newline="") as wf:
        writer = csv.writer(wf)
        writer.writerow(header)
        writer.writerows(filtered_rows)
    print(f"[WRITE] Updated file written: {path} (backup: {backup})")


def main():
    args = parse_args()
    try:
        threshold_date = dt.date.fromisoformat(args["threshold_date"] if isinstance(args, dict) else args.threshold_date)
        cutoff_date = dt.date.fromisoformat(args["cutoff_date"] if isinstance(args, dict) else args.cutoff_date)
    except Exception:
        threshold_date = dt.date.fromisoformat(args.threshold_date)
        cutoff_date = dt.date.fromisoformat(args.cutoff_date)
    files = find_files(args.root)
    if not files:
        print(f"[INFO] No files found under root {args.root}")
        return
    print(f"[INFO] Found {len(files)} all.csv files.")
    for f in files:
        process_file(Path(f), threshold_date, cutoff_date, args.dry_run, args.force)


if __name__ == "__main__":
    main()
