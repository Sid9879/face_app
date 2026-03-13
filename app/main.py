from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from .face_service import compare_faces

app = FastAPI(
    title="Face Verification API",
    version="1.0"
)

def read_image(file):

    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return image


@app.post("/compare")
async def compare(img1: UploadFile = File(...), img2: UploadFile = File(...)):

    image1 = read_image(img1)
    image2 = read_image(img2)

    score, error = compare_faces(image1, image2)

    if error:
        return {
            "success": False,
            "message": error
        }

    return {
        "success": True,
        "matchScore": round(score, 2),
        "isMatch": score > 60
    }