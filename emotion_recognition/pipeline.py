"""Un frame in ingresso, una lista di volti con emozione in uscita."""

import numpy as np

from . import config
from .models import EmotionClassifier, FaceDetector
from .tracking import FaceTracker, MainFaceSelector


def crop_face(gray: np.ndarray, x: int, y: int, w: int, h: int):
    """Ritaglio quadrato attorno al volto, limitato ai bordi dell'immagine."""
    side = int(max(w, h) * (1 + config.CROP_MARGIN))
    cx, cy = x + w // 2, y + h // 2
    x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x2, y2 = min(cx + side // 2, gray.shape[1]), min(cy + side // 2, gray.shape[0])
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return gray[y1:y2, x1:x2]


class EmotionPipeline:
    """Rilevamento -> ritaglio -> classificazione -> media nel tempo -> volto principale."""

    def __init__(self):
        self.detector = FaceDetector(config.FACE_MODEL, config.FACE_SCORE_THRESHOLD)
        self.classifier = EmotionClassifier(config.EMOTION_MODEL, config.LABELS)
        self.tracker = FaceTracker()
        self.selector = MainFaceSelector()

    def analyze(self, img: np.ndarray, gray: np.ndarray) -> list:
        """Ritorna una lista di dict, uno per volto:
        box, landmarks, score, small (volto 48x48), probs, main.
        """
        faces = []
        for face in self.detector.detect(img):
            x, y, w, h = map(int, face[:4])
            if min(w, h) < config.FACE_MIN_SIZE:
                continue
            roi = crop_face(gray, x, y, w, h)
            if roi is None:
                continue
            small = self.classifier.model_view(roi)
            faces.append({
                "box": (x, y, w, h),
                "landmarks": face[4:14].reshape(5, 2).astype(int),
                "score": float(face[14]),
                "small": small,
                "probs": self.classifier.predict(small),
            })

        centers = [(f["box"][0] + f["box"][2] / 2, f["box"][1] + f["box"][3] / 2) for f in faces]
        smoothed = self.tracker.smooth(list(zip(centers, [f["probs"] for f in faces])))
        for f, probs in zip(faces, smoothed):
            f["probs"] = probs
        self.selector.select(faces)
        return faces
