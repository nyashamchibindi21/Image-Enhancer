from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64

app = FastAPI(
    title="Document Image Enhancer",
    description="Enhances document images for OCR and returns Base64-safe JSON",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )

    # Read image bytes
    image_bytes = await file.read()

    # Decode image using OpenCV
    np_image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image data"
        )

    # ===============================
    # IMAGE ENHANCEMENT PIPELINE
    # ===============================

    # 1. Normalize contrast
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # 2. Increase contrast and brightness
    enhanced = cv2.convertScaleAbs(
        normalized,
        alpha=1.4,   # contrast
        beta=15     # brightness
    )

    # 3. Noise reduction
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 4. Adaptive threshold (OCR-friendly)
    enhanced = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    # Encode enhanced image to JPEG
    success, encoded = cv2.imencode(".jpg", enhanced)
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode image"
        )

    # Convert to Base64 (n8n-safe)
    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

    # Return JSON-safe response
    return JSONResponse(
        content={
            "image_base64": image_base64,
            "mime_type": "image/jpeg",
            "filename": "enhanced.jpg"
        }
    )

