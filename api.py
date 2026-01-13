from fastapi import FastAPI, Request
import cv2
import numpy as np

app = FastAPI()

@app.post("/enhance")
async def enhance_image(request: Request):
    # Read raw binary body
    contents = await request.body()

    # Convert to OpenCV image
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # OCR-optimized enhancement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=15)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(contrast, -1, kernel)

    _, buffer = cv2.imencode(".jpg", sharpened)
    return buffer.tobytes()
