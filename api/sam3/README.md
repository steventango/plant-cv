# SAM 3.1 Video Tracking API

Stateless online video tracking API using the native `facebookresearch/sam3` package
(SAM 3.1 Object Multiplex) and LitServe. Frames are processed one per request; tracking
memory is held resident per session and written through to disk (`checkpoints/`) so
sessions survive restarts. The gated `facebook/sam3.1` checkpoint downloads once at
startup into the mounted HuggingFace cache (requires HF access).

## Running

### With Docker Compose (Recommended)
```bash
docker compose up -d sam3
```

### Local Development
```bash
uv run python app/main.py
```

The server runs on port **8805**.

## Endpoints

All requests should be sent to `POST /predict`. The specific action is determined by the `endpoint` field in the JSON payload.

### 1. Detect (Initialize Tracking)

Initialize a tracking session with the first frame and a text prompt.

**Request:**
```json
{
    "endpoint": "detect",
    "image_data": "base64_encoded_image_string",
    "text_prompt": "plant" 
}
```

**Response:**
```json
{
    "session_id": "uuid_of_tracking_session",
    "masks": [
        {
            "object_id": 1,
            "contour": [[x1, y1], [x2, y2], ...],
            "box": [x_min, y_min, x_max, y_max],
            "score": 0.95
        },
        ...
    ]
}
```

### 2. Propagate (Track Next Frame)

Advance the tracking session by one frame using the `session_id` from a previous response.

**Request:**
```json
{
    "endpoint": "propagate",
    "image_data": "base64_encoded_image_string_of_next_frame",
    "session_id": "uuid_from_previous_response"
}
```

**Response:**
```json
{
    "session_id": "uuid_of_tracking_session",
    "masks": [
        {
            "object_id": 1,
            "contour": [[x1, y1], [x2, y2], ...],
            "box": [x_min, y_min, x_max, y_max],
            "score": 0.98
        },
        ...
    ]
}
```

### 3. Health Check

**Request:**
```json
{
    "endpoint": "health"
}
```

**Response:**
```json
{
    "status": "healthy"
}
```
