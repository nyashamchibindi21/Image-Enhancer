from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import cv2
import numpy as np

app = FastAPI(
    title="Image Enhancer API",
    description="Enhances document images for OCR",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )

    # Read file bytes
    image_bytes = await file.read()

    # Convert bytes to OpenCV image
    np_image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # =========================
    # IMAGE ENHANCEMENT PIPELINE
    # =========================

    # 1. Normalize contrast
    normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # 2. Increase contrast + brightness
    enhanced = cv2.convertScaleAbs(
        normalized,
        alpha=1.4,   # contrast
        beta=15     # brightness
    )

    # 3. Reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 4. Adaptive threshold (excellent for OCR)
    enhanced = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    # Encode image back to JPEG
    success, encoded_image = cv2.imencode(".jpg", enhanced)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode enhanced image"
        )

    return Response(
        content=encoded_image.tobytes(),
        media_type="image/jpeg"
    )
