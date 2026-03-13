import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .utils import is_blurry

model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)

def get_embedding(image):

    faces = model.get(image)

    if len(faces) == 0:
        return None

    return faces[0].embedding


def compare_faces(img1, img2):

    if is_blurry(img1) or is_blurry(img2):
        return None, "Image is blurry"

    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    if emb1 is None or emb2 is None:
        return None, "Face not detected"

    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )

    score = float(similarity * 100)

    return score, None