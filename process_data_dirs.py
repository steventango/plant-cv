import argparse
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from scripts.bulk_pipeline_client import BulkPipelineProcessor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def find_alliance_zones(base_dirs: List[str], filter_p1: bool = False) -> List[Path]:
    """Find all alliance-zone directories in the given base directories."""
    zones = []
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning(f"Base directory does not exist: {base_dir}")
            continue

        # Find all alliance-zone* directories
        for zone in base_path.rglob("alliance-zone*"):
            if zone.is_dir():
                # Filter for P1 directories if requested
                if filter_p1:
                    # Check if P1 is in the path (e.g., E13/P1/...)
                    parts = zone.parts
                    if "P1" not in parts:
                        continue
                zones.append(zone)

    return sorted(zones)


def parse_image_timestamp(image_name: str) -> Optional[datetime]:
    """
    Parse timestamp from image filename.

    Expected format: YYYY-MM-DDTHHMM+ZZZZ_position.ext
    Example: 2025-10-08T153000+0000_left.jpg

    Args:
        image_name: Image filename

    Returns:
        datetime object in UTC, or None if parsing fails
    """
    try:
        # Extract timestamp part (before first underscore)
        timestamp_str = image_name.split('_')[0]
        # Parse as UTC datetime
        dt = datetime.fromisoformat(timestamp_str)
        return dt
    except (ValueError, IndexError) as e:
        logger.debug(f"Could not parse timestamp from {image_name}: {e}")
        return None


def find_first_9am_index(image_names: List[str], timezone: str = "America/Edmonton") -> int:
    """
    Find the index of the first image taken at or after 9:30 AM in the specified timezone.

    Args:
        image_names: List of image filenames
        timezone: Timezone string (default: America/Edmonton)

    Returns:
        Index of first 9:30 AM image, or 0 if none found
    """
    tz = ZoneInfo(timezone)

    for idx, img_name in enumerate(image_names):
        dt = parse_image_timestamp(img_name)
        if dt is None:
            continue

        # Convert to target timezone
        local_dt = dt.astimezone(tz)

        # Check if this is after 9:30 AM but before 9:30 PM
        if local_dt.hour == 9 and local_dt.minute >= 30 or 9 < local_dt.hour < 21:
            logger.info(f"Found first 9:30 AM image at index {idx}: {img_name} (local time: {local_dt})")
            return idx

    logger.warning(f"No 9:30 AM image found in {len(image_names)} images, starting from beginning")
    return 0


def filter_images_by_time(
    image_names: List[str],
    timezone: str = "America/Edmonton",
    exclude_night: bool = False,
    specific_time: Optional[str] = None,
) -> List[str]:
    """
    Filter images based on time criteria.

    Args:
        image_names: List of image filenames
        timezone: Timezone string (default: America/Edmonton)
        exclude_night: If True, exclude images from 21:00 to 9:00
        specific_time: If provided, only include images at this specific time (format: "HH:MM")

    Returns:
        Filtered list of image filenames
    """
    if not exclude_night and not specific_time:
        return image_names

    tz = ZoneInfo(timezone)
    filtered = []

    for img_name in image_names:
        dt = parse_image_timestamp(img_name)
        if dt is None:
            # If we can't parse timestamp, include it by default
            filtered.append(img_name)
            continue

        # Convert to target timezone
        local_dt = dt.astimezone(tz)

        # Check night time filter (21:00 to 9:00)
        if exclude_night:
            hour = local_dt.hour
            # Night is 21:00 (9 PM) to 9:00 (9 AM)
            if hour >= 21 or hour < 9:
                continue

        # Check specific time filter
        if specific_time:
            target_hour, target_minute = map(int, specific_time.split(':'))
            if local_dt.hour != target_hour or local_dt.minute != target_minute:
                continue

        filtered.append(img_name)

    logger.info(f"Filtered {len(image_names)} images to {len(filtered)} images")
    if exclude_night:
        logger.info("  Excluded nighttime images (21:00 to 9:00)")
    if specific_time:
        logger.info(f"  Filtered to specific time: {specific_time}")

    return filtered


def process_zone_directory(
    zone_dir: Path,
    processor: BulkPipelineProcessor,
    version: str = "v5.0.0",
    text_prompt: str = "pot",
    threshold: float = 0.03,
    margin: float = 0.25,
    output_size: int = 400,
    visualize: bool = False,
    dry_run: bool = False,
    embedding_url: Optional[str] = None,
    embedding_workers: int = 10,
    timezone: str = "America/Edmonton",
    exclude_night: bool = False,
    specific_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single alliance-zone directory.

    Args:
        zone_dir: Path to alliance-zone directory
        processor: BulkPipelineProcessor instance
        version: Version string for output directory
        text_prompt: Text prompt for detection
        threshold: Detection threshold
        margin: Warp margin
        output_size: Fixed output size
        visualize: Whether to save visualization images
        dry_run: If True, don't actually process, just print what would be done
        embedding_url: URL of embedding service (if None, skip embeddings)
        embedding_workers: Number of parallel workers for embeddings
        timezone: Timezone string for time filtering
        exclude_night: If True, exclude images from 21:00 to 9:00
        specific_time: If provided, only process images at this specific time (format: "HH:MM")

    Returns:
        Dictionary with processing results and metadata
    """
    images_dir = zone_dir / "images"
    raw_csv = zone_dir / "raw.csv"

    # Check if images directory exists
    if not images_dir.exists():
        logger.warning(f"Images directory does not exist: {images_dir}")
        return {"error": "Images directory not found", "zone_dir": str(zone_dir)}

    # Check if raw.csv exists
    if not raw_csv.exists():
        logger.warning(f"raw.csv not found: {raw_csv}")
        return {"error": "raw.csv not found", "zone_dir": str(zone_dir)}

    # Read raw.csv to get image names
    try:
        df = pd.read_csv(raw_csv)
        if "image_name" not in df.columns:
            logger.error(f"raw.csv missing 'image_name' column: {raw_csv}")
            return {"error": "raw.csv missing image_name column", "zone_dir": str(zone_dir)}

        # Get unique image names in order
        image_names = df["image_name"].unique().tolist()

        # Apply time filtering
        image_names = filter_images_by_time(
            image_names,
            timezone=timezone,
            exclude_night=exclude_night,
            specific_time=specific_time,
        )

        if not image_names:
            logger.warning("No images remaining after time filtering")
            return {"error": "No images after time filtering", "zone_dir": str(zone_dir)}

        # Find the first 9:30 AM image to use as reference for detection
        reference_idx = find_first_9am_index(image_names, timezone=timezone)
        if reference_idx > 0:
            logger.info(f"Using image at index {reference_idx} as reference for pot detection")

    except Exception as e:
        logger.error(f"Error reading raw.csv: {e}")
        return {"error": f"Error reading raw.csv: {e}", "zone_dir": str(zone_dir)}

    # Build image paths
    image_paths = []
    for img_name in image_names:
        img_path = images_dir / img_name
        if img_path.exists():
            image_paths.append(img_path)
        else:
            logger.warning(f"Image file not found: {img_path}")

    if not image_paths:
        logger.warning(f"No valid images found in {images_dir}")
        return {"error": "No valid images found", "zone_dir": str(zone_dir)}

    # Setup output directory
    output_base = zone_dir / "processed" / version
    output_images_dir = output_base / "images"
    output_csv = output_base / "all.csv"

    logger.info(f"Processing zone: {zone_dir}")
    logger.info(f"  Found {len(image_paths)} images")
    logger.info(f"  Output directory: {output_base}")

    if dry_run:
        print("\n[DRY RUN] Would process:")
        print(f"  Zone: {zone_dir}")
        print(f"  Images: {len(image_paths)}")
        print(f"  Output: {output_base}")
        return {"zone_dir": str(zone_dir), "num_images": len(image_paths), "dry_run": True}

    # Create output directory
    output_images_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    try:
        result = processor.process_bulk(
            image_paths,
            text_prompt=text_prompt,
            threshold=threshold,
            margin=margin,
            output_size=output_size,
            output_dir=output_images_dir,
            save_metadata=True,
            visualize=visualize,
            reference_image_index=reference_idx,
        )

        # Create all.csv with processed image paths and embeddings
        create_output_csv(
            result,
            image_paths,
            output_csv,
            output_images_dir,
            embedding_url=embedding_url,
            embedding_workers=embedding_workers,
        )

        return {
            "zone_dir": str(zone_dir),
            "num_images": len(image_paths),
            "num_pots": len(result["pot_info"]["quadrilaterals"]),
            "output_csv": str(output_csv),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error processing zone {zone_dir}: {e}")
        return {
            "zone_dir": str(zone_dir),
            "error": str(e),
            "success": False,
        }


def get_embedding(image_path: Path, embedding_url: str = "http://embeddings:8000") -> Optional[List[float]]:
    """
    Get CLS token embedding for an image.

    Args:
        image_path: Path to image file
        embedding_url: URL of embedding service

    Returns:
        List of embedding values, or None if error
    """
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Request CLS token embedding
        response = requests.post(
            f"{embedding_url}/predict",
            json={
                "image_data": image_data,
                "embedding_types": ["cls_token"]
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            return result["cls_token"]
        else:
            logger.warning(f"Embedding request failed for {image_path}: {response.status_code}")
            return None

    except Exception as e:
        logger.warning(f"Error getting embedding for {image_path}: {e}")
        return None


def create_output_csv(
    result: Dict[str, Any],
    image_paths: List[Path],
    output_csv: Path,
    output_images_dir: Path,
    embedding_url: Optional[str] = None,
    embedding_workers: int = 10,
) -> None:
    """
    Create all.csv with processed image paths and embeddings.

    Args:
        result: Result from process_bulk
        image_paths: Original image paths
        output_csv: Output CSV path
        output_images_dir: Base directory for processed images
        embedding_url: URL of embedding service (if None, skip embeddings)
        embedding_workers: Number of parallel workers for embeddings
    """
    num_pots = len(result["pot_info"]["quadrilaterals"])
    tracker_ids = result["pot_info"].get("tracker_ids", list(range(num_pots)))

    # Ensure we have tracker IDs for all pots
    if len(tracker_ids) < num_pots:
        tracker_ids = list(range(num_pots))

    rows = []
    for img_path, warp_result in zip(image_paths, result["warp_results"]):
        if "error" in warp_result:
            logger.warning(f"Skipping image with error: {img_path}")
            continue

        image_stem = img_path.stem

        for pot_idx in range(num_pots):
            plant_id = tracker_ids[pot_idx]
            pot_image_path = output_images_dir / image_stem / f"pot_{pot_idx:02d}.jpg"

            # Check if the pot image exists
            if pot_image_path.exists():
                row = {
                    "image_name": img_path.name,
                    "plant_id": plant_id,
                    "pot_index": pot_idx,
                    "pot_image_path": str(pot_image_path.relative_to(output_csv.parent)),
                }
                rows.append(row)

    # Get embeddings in parallel if service is available
    if embedding_url and rows:
        logger.info(f"Getting embeddings for {len(rows)} images using {embedding_workers} workers...")

        # Create a mapping from pot_image_path to row index
        pot_paths = [Path(output_csv.parent / row["pot_image_path"]) for row in rows]

        # Use ThreadPoolExecutor to get embeddings in parallel
        with ThreadPoolExecutor(max_workers=embedding_workers) as executor:
            # Submit all embedding tasks
            future_to_idx = {
                executor.submit(get_embedding, pot_path, embedding_url): idx
                for idx, pot_path in enumerate(pot_paths)
            }

            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Embeddings"):
                idx = future_to_idx[future]
                try:
                    embedding = future.result()
                    if embedding:
                        rows[idx]["embedding"] = json.dumps(embedding)
                    else:
                        rows[idx]["embedding"] = None
                except Exception as e:
                    logger.warning(f"Error getting embedding for {pot_paths[idx]}: {e}")
                    rows[idx]["embedding"] = None

    # Write CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        logger.info(f"Wrote {len(rows)} rows to {output_csv}")
    else:
        logger.warning(f"No rows to write to {output_csv}")


def main():
    """Command-line interface for processing data directories."""
    parser = argparse.ArgumentParser(
        description="Process data directories with CSV integration"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["E11", "E12", "E13", "E14"],
        help="Experiment directories to process (e.g., E11 E12)",
    )
    parser.add_argument(
        "--base-dir",
        default="/data/online",
        help="Base directory containing experiment folders",
    )
    parser.add_argument(
        "--url",
        default="http://pipeline:8000",
        help="Pipeline service URL",
    )
    parser.add_argument(
        "--embedding-url",
        default="http://embeddings:8000",
        help="Embedding service URL (set to empty string to disable)",
    )
    parser.add_argument(
        "--version",
        default="v5.0.0",
        help="Version string for output directory",
    )
    parser.add_argument(
        "--prompt",
        default="pot",
        help="Detection text prompt",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Detection threshold",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="Warp margin",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=400,
        help="Fixed output size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--embedding-workers",
        type=int,
        default=256,
        help="Number of parallel workers for embeddings",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - don't actually process",
    )
    parser.add_argument(
        "--zone",
        help="Process only a specific zone directory (full path)",
    )
    parser.add_argument(
        "--p1-only",
        action="store_true",
        help="Process only P1 directories (E*/P1/...)",
    )
    parser.add_argument(
        "--timezone",
        default="America/Edmonton",
        help="Timezone for time filtering (default: America/Edmonton)",
    )
    parser.add_argument(
        "--exclude-night",
        action="store_true",
        help="Exclude images taken at night (21:00 to 9:00)",
    )
    parser.add_argument(
        "--specific-time",
        help="Only process images at a specific time (format: HH:MM, e.g., 09:30)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize processor
    processor = BulkPipelineProcessor(args.url, args.workers)

    # Check service health
    if not args.dry_run:
        try:
            health = processor.client.health()
            print(f"Pipeline service status: {health['status']}")
        except Exception as e:
            print(f"Error connecting to pipeline service: {e}")
            return

        # Check embedding service if enabled
        if args.embedding_url:
            try:
                response = requests.get(f"{args.embedding_url}/health", timeout=5)
                if response.status_code == 200:
                    print("Embedding service status: available")
                else:
                    print("Embedding service status: unavailable (will skip embeddings)")
                    args.embedding_url = None
            except Exception as e:
                print(f"Warning: Embedding service not available: {e}")
                print("Will process without embeddings")
                args.embedding_url = None

    # Find zones to process
    if args.zone:
        zones = [Path(args.zone)]
    else:
        base_dirs = [f"{args.base_dir}/{exp}" for exp in args.experiments]
        zones = find_alliance_zones(base_dirs, filter_p1=args.p1_only)

    print(f"\nFound {len(zones)} alliance-zone directories to process")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===\n")

    # Process each zone
    results = []
    for i, zone in enumerate(zones, 1):
        print(f"\n{'='*80}")
        print(f"Processing zone {i}/{len(zones)}: {zone}")
        print(f"{'='*80}")

        result = process_zone_directory(
            zone,
            processor,
            version=args.version,
            text_prompt=args.prompt,
            threshold=args.threshold,
            margin=args.margin,
            output_size=args.size,
            visualize=args.visualize,
            dry_run=args.dry_run,
            embedding_url=args.embedding_url if args.embedding_url else None,
            embedding_workers=args.embedding_workers,
            timezone=args.timezone,
            exclude_night=args.exclude_night,
            specific_time=args.specific_time,
        )
        results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success") and not r.get("dry_run")]

    print(f"\nTotal zones: {len(zones)}")
    if args.dry_run:
        print("Dry run - no processing performed")
    else:
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print("\nFailed zones:")
            for r in failed:
                print(f"  {r['zone_dir']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
