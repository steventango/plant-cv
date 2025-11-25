"""Bulk pipeline client for processing multiple images."""

import argparse
import base64
import io
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _process_warp_image(args):
    """
    Process a single image for warping (module-level function for ProcessPoolExecutor).

    Args:
        args: Tuple of (image_path, base_url, quadrilaterals, margin, output_size, output_dir)

    Returns:
        Dictionary with warping results
    """
    image_path, base_url, quadrilaterals, margin, output_size, output_dir = args
    image_path = Path(image_path)

    # Create a new client for this process
    client = PipelineClient(base_url)

    # Load and process image
    image = Image.open(image_path).convert("RGB")
    warp_result = client.warp(image, quadrilaterals, margin, output_size)

    result = {
        "image_path": str(image_path),
        "warped_images": warp_result["warped_images"],
        "homographies": warp_result["homographies"],
    }

    # Save warped images if output_dir provided
    max_dimensions = (0, 0)
    if output_dir:
        output_dir = Path(output_dir)
        stem = image_path.stem
        image_output_dir = output_dir / stem
        image_output_dir.mkdir(exist_ok=True)

        saved_paths = []
        for idx, warped_b64 in enumerate(warp_result["warped_images"]):
            if warped_b64:
                warped_img = client.decode_image(warped_b64)
                save_path = image_output_dir / f"pot_{idx:02d}.jpg"
                warped_img.save(save_path)
                saved_paths.append(str(save_path))
                # Track largest output dimensions
                img_size = warped_img.size[0] * warped_img.size[1]
                max_size = max_dimensions[0] * max_dimensions[1]
                if img_size > max_size:
                    max_dimensions = warped_img.size
            else:
                saved_paths.append(None)
        result["saved_paths"] = saved_paths
        result["max_dimensions"] = max_dimensions

    return result


class PipelineClient:
    """Client for the pot cropping pipeline service."""

    def __init__(self, base_url="http://localhost:8000"):
        """
        Initialize the pipeline client.

        Args:
            base_url: Base URL of the pipeline service
        """
        self.base_url = base_url.rstrip("/")

    def encode_image(self, image):
        """Encode PIL Image to base64."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def decode_image(self, image_data):
        """Decode base64 to PIL Image."""
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def decode_masks(self, masks_b64):
        """Decode base64 numpy array."""
        masks_bytes = io.BytesIO(base64.b64decode(masks_b64))
        return np.load(masks_bytes)

    def health(self):
        """Check service health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def detect(self, image, text_prompt="pot", threshold=0.03, visualize=False):
        """Detect pots in an image."""
        payload = {
            "image_data": self.encode_image(image),
            "text_prompt": text_prompt,
            "threshold": threshold,
            "visualize": visualize,
        }
        response = requests.post(f"{self.base_url}/pot/detect", json=payload)
        response.raise_for_status()
        return response.json()

    def segment(self, image, boxes, visualize=False):
        """Segment pot boxes."""
        payload = {
            "image_data": self.encode_image(image),
            "boxes": boxes,
            "visualize": visualize,
        }
        response = requests.post(f"{self.base_url}/pot/segment", json=payload)
        response.raise_for_status()
        return response.json()

    def quad(self, masks_b64, image=None, visualize=False):
        """Compute quadrilaterals from masks."""
        payload = {
            "masks": masks_b64,
            "visualize": visualize,
        }
        if image is not None:
            payload["image_data"] = self.encode_image(image)
        response = requests.post(f"{self.base_url}/pot/quad", json=payload)
        response.raise_for_status()
        return response.json()

    def warp(self, image, quadrilaterals, margin=0.25, output_size=None):
        """Warp pot regions to squares."""
        payload = {
            "image_data": self.encode_image(image),
            "quadrilaterals": quadrilaterals,
            "margin": margin,
            "output_size": output_size,
        }
        response = requests.post(f"{self.base_url}/pot/warp", json=payload)
        response.raise_for_status()
        return response.json()

    def pipeline(
        self,
        image,
        text_prompt="pot",
        threshold=0.03,
        margin=0.25,
        output_size=None,
        visualize=False,
    ):
        """Run full pipeline on an image."""
        payload = {
            "image_data": self.encode_image(image),
            "text_prompt": text_prompt,
            "threshold": threshold,
            "margin": margin,
            "output_size": output_size,
            "visualize": visualize,
        }
        response = requests.post(f"{self.base_url}/pot/pipeline", json=payload)
        response.raise_for_status()
        return response.json()


class BulkPipelineProcessor:
    """Bulk processor for multiple images using the pipeline service."""

    def __init__(self, base_url="http://localhost:8000", max_workers=4):
        """
        Initialize bulk processor.

        Args:
            base_url: Base URL of the pipeline service
            max_workers: Maximum number of parallel workers
        """
        self.base_url = base_url
        self.client = PipelineClient(base_url)
        self.max_workers = max_workers

    def process_first_image(
        self,
        image_path,
        text_prompt="pot",
        threshold=0.03,
        visualize=False,
        output_dir=None,
    ):
        """
        Process first image to get pot positions (detect, segment, quad).

        Args:
            image_path: Path to first image
            text_prompt: Text prompt for detection
            threshold: Detection threshold
            visualize: Whether to save visualization images
            output_dir: Output directory for visualizations

        Returns:
            dict with boxes, masks, quadrilaterals
        """
        image = Image.open(image_path).convert("RGB")

        # Detect pots
        detect_result = self.client.detect(
            image, text_prompt, threshold, visualize=visualize
        )
        boxes = detect_result["boxes"]
        tracker_ids = detect_result.get("tracker_ids", [])

        # Save detection visualization
        if visualize and output_dir and "visualization" in detect_result:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            vis_img = self.client.decode_image(detect_result["visualization"])
            vis_path = output_dir / f"{Path(image_path).stem}_detect.jpg"
            vis_img.save(vis_path)
            print(f"   Saved detection visualization to {vis_path}")

        if not boxes:
            return {
                "boxes": [],
                "masks": None,
                "quadrilaterals": [],
                "image_shape": np.array(image).shape[:2],
            }

        # Segment pots
        segment_result = self.client.segment(image, boxes, visualize=visualize)
        masks_b64 = segment_result["masks"]

        # Save segmentation visualization
        if visualize and output_dir and "visualization" in segment_result:
            vis_img = self.client.decode_image(segment_result["visualization"])
            vis_path = output_dir / f"{Path(image_path).stem}_segment.jpg"
            vis_img.save(vis_path)
            print(f"   Saved segmentation visualization to {vis_path}")

        # Compute quadrilaterals
        quad_result = self.client.quad(
            masks_b64, image=image if visualize else None, visualize=visualize
        )
        quadrilaterals = quad_result["quadrilaterals"]

        # Save quad visualization
        if visualize and output_dir and "visualization" in quad_result:
            vis_img = self.client.decode_image(quad_result["visualization"])
            vis_path = output_dir / f"{Path(image_path).stem}_quad.jpg"
            vis_img.save(vis_path)
            print(f"   Saved quadrilateral visualization to {vis_path}")

        return {
            "boxes": boxes,
            "tracker_ids": tracker_ids,
            "masks": masks_b64,
            "quadrilaterals": quadrilaterals,
            "image_shape": np.array(image).shape[:2],
        }

    def warp_sequence(
        self,
        image_paths,
        quadrilaterals,
        margin=0.25,
        output_size=None,
        output_dir=None,
    ):
        """
        Warp pot regions for a sequence of images using fixed quadrilaterals.

        Args:
            image_paths: List of image paths
            quadrilaterals: Fixed quadrilaterals from first image
            margin: Warp margin
            output_size: Optional fixed output size
            output_dir: Optional output directory to save warped images

        Returns:
            List of results per image
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare arguments for parallel processing
        process_args = [
            (str(path), self.base_url, quadrilaterals, margin, output_size, str(output_dir) if output_dir else None)
            for path in image_paths
        ]

        results = []
        max_dimensions = (0, 0)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_process_warp_image, args): args[0]
                for args in process_args
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Warping images"
            ):
                try:
                    result = future.result()
                    results.append(result)

                    # Track largest output dimensions
                    if "max_dimensions" in result:
                        img_size = result["max_dimensions"][0] * result["max_dimensions"][1]
                        max_size = max_dimensions[0] * max_dimensions[1]
                        if img_size > max_size:
                            max_dimensions = result["max_dimensions"]
                except Exception as e:
                    image_path = futures[future]
                    print(f"Error processing {image_path}: {e}")
                    results.append(
                        {
                            "image_path": image_path,
                            "error": str(e),
                        }
                    )

        # Sort results by original order
        path_to_result = {r["image_path"]: r for r in results}
        ordered_results = [path_to_result[str(p)] for p in image_paths]

        if max_dimensions[0] > 0 and max_dimensions[1] > 0:
            logger.info(
                f"Largest warped output dimensions: {max_dimensions[0]}x{max_dimensions[1]} pixels"
            )
            print(
                f"Largest warped output dimensions: {max_dimensions[0]}x{max_dimensions[1]} pixels"
            )

        return ordered_results

    def process_bulk(
        self,
        image_paths,
        text_prompt="pot",
        threshold=0.03,
        margin=0.25,
        output_size=None,
        output_dir=None,
        save_metadata=True,
        visualize=False,
        reference_image_index=0,
    ):
        """
        Process a sequence of images in bulk.

        Uses reference image to detect pot positions, then applies warping to all images.

        Args:
            image_paths: List of image paths (sorted by time)
            text_prompt: Text prompt for detection
            threshold: Detection threshold
            margin: Warp margin
            output_size: Optional fixed output size
            output_dir: Output directory for warped images
            save_metadata: Whether to save metadata JSON
            visualize: Whether to save visualization images
            reference_image_index: Index of image to use for pot detection (default: 0)

        Returns:
            dict with pot_info and warp_results
        """
        if not image_paths:
            raise ValueError("No images provided")

        image_paths = [Path(p) for p in image_paths]

        print(f"Processing {len(image_paths)} images...")
        print(f"Reference image (index {reference_image_index}): {image_paths[reference_image_index]}")

        # Step 1: Process reference image to get pot positions
        print(f"\n1. Detecting and segmenting pots in reference image (index {reference_image_index})...")
        pot_info = self.process_first_image(
            image_paths[reference_image_index],
            text_prompt,
            threshold,
            visualize=visualize,
            output_dir=output_dir,
        )

        if not pot_info["quadrilaterals"]:
            print("No pots detected in first image!")
            return {"pot_info": pot_info, "warp_results": []}

        print(f"   Detected {len(pot_info['quadrilaterals'])} pots")

        # Step 2: Warp all images using fixed pot positions
        print("\n2. Warping all images using fixed pot positions...")
        warp_results = self.warp_sequence(
            image_paths, pot_info["quadrilaterals"], margin, output_size, output_dir
        )

        # Save metadata
        if output_dir and save_metadata:
            output_dir = Path(output_dir)
            metadata = {
                "pot_info": {
                    "boxes": pot_info["boxes"],
                    "tracker_ids": pot_info.get("tracker_ids", []),
                    "quadrilaterals": pot_info["quadrilaterals"],
                    "num_pots": len(pot_info["quadrilaterals"]),
                    "reference_image": str(image_paths[reference_image_index]),
                    "reference_image_index": reference_image_index,
                },
                "processing_params": {
                    "text_prompt": text_prompt,
                    "threshold": threshold,
                    "margin": margin,
                    "output_size": output_size,
                },
                "num_images": len(image_paths),
            }
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"\n✓ Saved metadata to {metadata_path}")

        print(f"\n✓ Processed {len(warp_results)} images")
        return {"pot_info": pot_info, "warp_results": warp_results}


def main():
    """Command-line interface for bulk processing."""
    parser = argparse.ArgumentParser(description="Bulk pot cropping pipeline")
    parser.add_argument("images", nargs="+", help="Image paths or directory")
    parser.add_argument(
        "--url", default="http://pipeline:8000", help="Pipeline service URL"
    )
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--prompt", default="pot", help="Detection text prompt")
    parser.add_argument(
        "--threshold", type=float, default=0.03, help="Detection threshold"
    )
    parser.add_argument("--margin", type=float, default=0.25, help="Warp margin")
    parser.add_argument("--size", type=int, default=400, help="Fixed output size")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Save visualization images"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Collect image paths
    image_paths = []
    for path_str in args.images:
        path = Path(path_str)
        if path.is_dir():
            image_paths.extend(sorted(path.glob("*.jpg")))
            image_paths.extend(sorted(path.glob("*.jpeg")))
            image_paths.extend(sorted(path.glob("*.png")))
        else:
            image_paths.append(path)

    if not image_paths:
        print("No images found!")
        return

    image_paths = sorted(image_paths)

    # Initialize processor
    processor = BulkPipelineProcessor(args.url, args.workers)

    # Check service health
    try:
        health = processor.client.health()
        print(f"Service status: {health['status']}")
    except Exception as e:
        print(f"Error connecting to service: {e}")
        return

    # Process images
    try:
        processor.process_bulk(
            image_paths,
            text_prompt=args.prompt,
            threshold=args.threshold,
            margin=args.margin,
            output_size=args.size,
            output_dir=args.output,
            visualize=args.visualize,
        )
    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
