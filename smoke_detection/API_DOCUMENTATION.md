# Smoke Detection System - API Documentation

## Overview

This document describes the REST API endpoints provided by the Flask application for smoke detection.

**Base URL:** `http://localhost:5000`

---

## Endpoints

### 1. Home Page

**Endpoint:** `/`  
**Method:** `GET`  
**Description:** Renders the main web interface

**Response:**
- HTML page with smoke detection interface

**Example:**
```bash
curl http://localhost:5000/
```

---

### 2. Upload Image

**Endpoint:** `/upload`  
**Method:** `POST`  
**Description:** Upload an image for smoke detection

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:**
  - `file`: Image file (JPG, PNG, GIF)
  - Max size: 16MB

**Response:**
```json
{
  "success": true,
  "smoke_detected": true,
  "confidence": 0.8523,
  "timestamp": "2025-10-27 14:30:45",
  "image_url": "/static/uploads/20251027_143045_image.jpg",
  "threshold": 0.75
}
```

**Response Fields:**
- `success` (boolean): Request success status
- `smoke_detected` (boolean): Whether smoke was detected
- `confidence` (float): Prediction confidence (0.0 - 1.0)
- `timestamp` (string): Detection timestamp
- `image_url` (string): URL to uploaded image
- `threshold` (float): Current detection threshold

**Error Response:**
```json
{
  "error": "No file uploaded"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (no file, invalid type, etc.)
- `500`: Server error

**Example:**
```bash
curl -X POST \
  -F "file=@/path/to/image.jpg" \
  http://localhost:5000/upload
```

**JavaScript Example:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/upload', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Smoke detected:', data.smoke_detected);
  console.log('Confidence:', data.confidence);
});
```

---

### 3. Predict Frame

**Endpoint:** `/predict_frame`  
**Method:** `POST`  
**Description:** Analyze a camera frame for smoke detection

**Request:**
- **Content-Type:** `application/json`
- **Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Request Fields:**
- `image` (string): Base64-encoded image data URL

**Response:**
```json
{
  "success": true,
  "smoke_detected": false,
  "confidence": 0.3421,
  "timestamp": "2025-10-27 14:31:12",
  "threshold": 0.75
}
```

**Response Fields:**
- `success` (boolean): Request success status
- `smoke_detected` (boolean): Whether smoke was detected
- `confidence` (float): Prediction confidence (0.0 - 1.0)
- `timestamp` (string): Detection timestamp
- `threshold` (float): Current detection threshold

**Error Response:**
```json
{
  "error": "No image data"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request
- `500`: Server error

**Example:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQ..."}' \
  http://localhost:5000/predict_frame
```

**JavaScript Example:**
```javascript
// Capture frame from video
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
canvas.getContext('2d').drawImage(video, 0, 0);

const imageData = canvas.toDataURL('image/jpeg');

fetch('/predict_frame', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ image: imageData })
})
.then(response => response.json())
.then(data => {
  console.log('Smoke detected:', data.smoke_detected);
});
```

---

### 4. Model Information

**Endpoint:** `/model_info`  
**Method:** `GET`  
**Description:** Get information about the loaded model

**Response:**
```json
{
  "architecture": "MobileNetV2",
  "input_shape": [224, 224, 3],
  "classes": ["no_smoke", "smoke"],
  "training_date": "2025-10-27T10:15:30",
  "total_params": 2257984
}
```

**Response Fields:**
- `architecture` (string): Model architecture name
- `input_shape` (array): Expected input dimensions
- `classes` (array): Class labels
- `training_date` (string): When model was trained
- `total_params` (integer): Total model parameters

**Error Response:**
```json
{
  "error": "Model info not found",
  "model_loaded": false
}
```

**Status Codes:**
- `200`: Success
- `404`: Model info not found

**Example:**
```bash
curl http://localhost:5000/model_info
```

---

### 5. Set Threshold

**Endpoint:** `/set_threshold`  
**Method:** `POST`  
**Description:** Update the detection confidence threshold

**Request:**
- **Content-Type:** `application/json`
- **Body:**
```json
{
  "threshold": 0.85
}
```

**Request Fields:**
- `threshold` (float): New threshold value (0.0 - 1.0)

**Response:**
```json
{
  "success": true,
  "threshold": 0.85
}
```

**Response Fields:**
- `success` (boolean): Update success status
- `threshold` (float): New threshold value

**Error Response:**
```json
{
  "error": "Threshold must be between 0.0 and 1.0"
}
```

**Status Codes:**
- `200`: Success
- `400`: Invalid threshold value

**Example:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"threshold":0.85}' \
  http://localhost:5000/set_threshold
```

**JavaScript Example:**
```javascript
fetch('/set_threshold', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ threshold: 0.85 })
})
.then(response => response.json())
.then(data => {
  console.log('New threshold:', data.threshold);
});
```

---

### 6. Health Check

**Endpoint:** `/health`  
**Method:** `GET`  
**Description:** Check application health status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "threshold": 0.75
}
```

**Response Fields:**
- `status` (string): Application status
- `model_loaded` (boolean): Whether model is loaded
- `threshold` (float): Current detection threshold

**Status Codes:**
- `200`: Healthy

**Example:**
```bash
curl http://localhost:5000/health
```

---

## Data Types

### Confidence Score
- **Type:** Float
- **Range:** 0.0 - 1.0
- **Description:** Probability that smoke is present
- **Example:** 0.8523 = 85.23% confidence

### Timestamp
- **Type:** String
- **Format:** `YYYY-MM-DD HH:MM:SS`
- **Example:** `2025-10-27 14:30:45`

### Image Data URL
- **Type:** String
- **Format:** `data:image/jpeg;base64,<base64_data>`
- **Description:** Base64-encoded image with data URL prefix

---

## Error Handling

### Error Response Format
```json
{
  "error": "Error message description"
}
```

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 413 | Payload Too Large - File exceeds 16MB |
| 415 | Unsupported Media Type - Invalid file type |
| 500 | Internal Server Error - Server-side error |

---

## Rate Limiting

**Current Implementation:** No rate limiting

**Recommended for Production:**
- 100 requests per minute per IP
- 1000 requests per hour per IP
- Implement using Flask-Limiter

**Example:**
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per minute"]
)
```

---

## Authentication

**Current Implementation:** No authentication

**Recommended for Production:**
- JWT tokens
- API keys
- OAuth 2.0

**Example with API Key:**
```bash
curl -X POST \
  -H "X-API-Key: your_api_key_here" \
  -F "file=@image.jpg" \
  http://localhost:5000/upload
```

---

## CORS Configuration

**Current Implementation:** No CORS headers

**For Cross-Origin Requests:**
```python
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

---

## WebSocket Support (Future)

For real-time bidirectional communication:

**Endpoint:** `/ws/stream`  
**Protocol:** WebSocket  
**Description:** Real-time video stream processing

**Example:**
```javascript
const ws = new WebSocket('ws://localhost:5000/ws/stream');

ws.onopen = () => {
  // Send video frames
  ws.send(frameData);
};

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Smoke detected:', result.smoke_detected);
};
```

---

## Testing the API

### Using cURL

**Upload Image:**
```bash
curl -X POST \
  -F "file=@test_image.jpg" \
  http://localhost:5000/upload
```

**Get Model Info:**
```bash
curl http://localhost:5000/model_info
```

**Set Threshold:**
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"threshold":0.8}' \
  http://localhost:5000/set_threshold
```

### Using Python Requests

```python
import requests

# Upload image
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/upload', files=files)
    print(response.json())

# Get model info
response = requests.get('http://localhost:5000/model_info')
print(response.json())

# Set threshold
data = {'threshold': 0.8}
response = requests.post('http://localhost:5000/set_threshold', json=data)
print(response.json())
```

### Using Postman

1. **Upload Image:**
   - Method: POST
   - URL: `http://localhost:5000/upload`
   - Body: form-data
   - Key: `file` (type: File)
   - Value: Select image file

2. **Predict Frame:**
   - Method: POST
   - URL: `http://localhost:5000/predict_frame`
   - Body: raw (JSON)
   - Content: `{"image": "data:image/jpeg;base64,..."}`

---

## Performance Considerations

### Response Times
- Image upload: 100-500ms
- Frame prediction: 50-200ms
- Model info: <10ms
- Health check: <5ms

### Optimization Tips
1. Use GPU for faster inference
2. Implement caching for repeated requests
3. Compress images before upload
4. Use WebSocket for continuous streaming
5. Implement request queuing for high load

---

## Security Best Practices

1. **Input Validation:**
   - Validate file types
   - Check file sizes
   - Sanitize filenames

2. **Rate Limiting:**
   - Prevent abuse
   - Limit requests per IP

3. **Authentication:**
   - Require API keys
   - Implement user sessions

4. **HTTPS:**
   - Use SSL/TLS in production
   - Encrypt data in transit

5. **Error Handling:**
   - Don't expose stack traces
   - Log errors securely

---

## Versioning

**Current Version:** v1  
**API Prefix:** None (future: `/api/v1/`)

**Future Versioning:**
```
/api/v1/upload
/api/v1/predict_frame
/api/v1/model_info
```

---

## Support

For API issues or questions:
1. Check this documentation
2. Review Flask application logs
3. Test with provided examples
4. Verify model is loaded

---

## Changelog

### Version 1.0.0 (2025-10-27)
- Initial API release
- Image upload endpoint
- Frame prediction endpoint
- Model info endpoint
- Threshold configuration
- Health check endpoint

---

*Last Updated: 2025-10-27*
