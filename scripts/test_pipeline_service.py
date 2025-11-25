"""Test script to verify pipeline service is working."""
import sys
from pathlib import Path

from PIL import Image

from api.pipeline.bulk_pipeline_client import PipelineClient

# Initialize client
client = PipelineClient(base_url="http://localhost:8002")

print("Testing pipeline service...\n")

# 1. Health check
print("1. Health check...")
try:
    health = client.health()
    print(f"   ✓ Service is {health['status']}\n")
except Exception as e:
    print(f"   ✗ Failed to connect: {e}")
    print("\n   Make sure the service is running:")
    print("   docker-compose up -d pipeline\n")
    sys.exit(1)

# 2. Test with a sample image
test_image_path = Path("test_images")
if not test_image_path.exists():
    print("   ⚠ No test_images directory found, skipping image test")
    sys.exit(0)

image_files = list(test_image_path.glob("*.jpg"))
if not image_files:
    print("   ⚠ No .jpg files found in test_images/, skipping image test")
    sys.exit(0)

test_image = image_files[0]
print(f"2. Testing with image: {test_image.name}")

try:
    image = Image.open(test_image).convert("RGB")

    # Test detect endpoint
    print("   - Testing /pot/detect...")
    detect_result = client.detect(image, text_prompt="pot", threshold=0.03)
    num_pots = len(detect_result["boxes"])
    print(f"     ✓ Detected {num_pots} pots")

    if num_pots == 0:
        print("     ⚠ No pots detected, cannot test further")
        sys.exit(0)

    # Test full pipeline endpoint
    print("   - Testing /pot/pipeline (full pipeline)...")
    pipeline_result = client.pipeline(
        image, text_prompt="pot", threshold=0.03, margin=0.25
    )
    num_warped = len([w for w in pipeline_result["warped_images"] if w])
    print(f"     ✓ Generated {num_warped} warped pot images")

    print("\n✓ All tests passed! Service is working correctly.")

except Exception as e:
    print(f"   ✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
