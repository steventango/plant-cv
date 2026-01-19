# SAM3 Video Tracking API

Stateless SAM3 video tracking API using LitServe.

## Running

### With Docker Compose (Recommended)
```bash
docker compose up -d sam3
```

### Local Development
```bash
uv run lightning deploy app/main.py
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
    "state": "base64_encoded_serialized_session_state",
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

Propagate tracking to the next frame using the state from the previous frame.

**Request:**
```json
{
    "endpoint": "propagate",
    "image_data": "base64_encoded_image_string_of_next_frame",
    "state": "base64_encoded_serialized_session_state_from_previous_response"
}
```

**Response:**
```json
{
    "state": "base64_encoded_serialized_session_state_updated",
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
