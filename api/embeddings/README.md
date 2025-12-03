# Embeddings API

A LitServe-based API for extracting image embeddings using Meta's DINOv3 models.

## Features

- Fast batch processing with GPU acceleration
- Support for DINOv3 pretrained models
- Flexible embedding extraction: CLS token, patch features, or both
- Base64 image input/output
- Automatic image resizing to multiples of 16 for ViT

## Models

Default model: `facebook/dinov3-vitb16-pretrain-lvd1689m`

This model produces 768-dimensional embeddings and is pretrained on the LVD-142M dataset.

## API Usage

### Request Format

```json
{
  "image_data": "base64_encoded_image_string",
  "embedding_types": ["cls_token"]
}
```

**Parameters:**
- `image_data` (required): Base64-encoded image (JPEG, PNG, etc.)
- `embedding_types` (optional, default: `["cls_token"]`): Array of embedding types to extract
  - `"cls_token"`: Returns the CLS token embedding (768-dim vector)
  - `"patch_features"`: Returns all patch feature embeddings (N×768 matrix, where N = number of image patches)
  - Can request one or both: `["cls_token"]`, `["patch_features"]`, or `["cls_token", "patch_features"]`

### Response Format

**For CLS token only:**
```json
{
  "cls_token": [0.123, -0.456, ...]
}
```

**For patch features only:**
```json
{
  "patch_features": [[0.123, -0.456, ...], [0.789, 0.012, ...]]
}
```

**For both:**
```json
{
  "cls_token": [0.123, -0.456, ...],
  "patch_features": [[0.123, -0.456, ...], [0.789, 0.012, ...]]
}
```

**Notes:**
- CLS token is a 768-dimensional vector
- Patch features is a 2D array with shape `[num_patches, 768]`
- The number of patches depends on the image size. For a 16×16 patch size, an image resized to 256×256 will have 256 patches (16×16)

### Example Usage

#### Python

```python
import base64
import requests
from PIL import Image
from io import BytesIO

# Load and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8803/predict",
    json={
        "image_data": image_data,
        "embedding_types": ["cls_token", "patch_features"]
    }
)

result = response.json()
cls_embedding = result["cls_token"]
patch_embeddings = result["patch_features"]
print(f"CLS token length: {len(cls_embedding)}")
print(f"Number of patches: {len(patch_embeddings)}")
print(f"Feature dimension: {len(patch_embeddings[0])}")
```

#### cURL

```bash
IMAGE_BASE64=$(base64 -w 0 image.jpg)
curl -X POST http://localhost:8803/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_data\": \"$IMAGE_BASE64\", \"embedding_types\": [\"cls_token\", \"patch_features\"]}"
```

## Development

### Local Setup

1. Install dependencies:
```bash
uv sync --dev
```

2. Run the server:
```bash
uv run python app/main.py
```

### Docker Setup

Build and run with Docker Compose:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8803`.

## Configuration

Edit `app/main.py` to configure:
- `max_batch_size`: Maximum number of images per batch (default: 16)
- `batch_timeout`: Time to wait for batch completion in seconds (default: 0.01)
- `num_api_servers`: Number of parallel API server instances (default: 4)
- `pretrained_model_name_or_path`: HuggingFace model identifier

## Performance

The API uses:
- BFloat16 precision on compatible GPUs
- TF32 on Ampere+ GPUs
- Parallel image preprocessing with ThreadPoolExecutor
- Batch inference for improved throughput

## Use Cases

- Image similarity search
- Visual clustering
- Transfer learning features
- Image retrieval systems
- Visual recommendation engines
