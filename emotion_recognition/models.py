"""Le due reti neurali: rilevamento volti (YuNet) e classificazione delle emozioni (TFLite)."""

from pathlib import Path

import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

LFS_POINTER_MAX_SIZE = 1024  # un puntatore Git LFS e' un file di testo di ~130 byte


def check_model_file(path: Path, hint: str = "") -> None:
    if not path.exists():
        raise SystemExit(f"File non trovato: {path}\n{hint}")
    if path.stat().st_size < LFS_POINTER_MAX_SIZE:
        raise SystemExit(
            f"{path.name} e' un puntatore Git LFS, non il modello vero.\n"
            "Esegui: git lfs install && git lfs pull"
        )


class FaceDetector:
    """Trova i volti con YuNet. Ogni volto: box, 5 punti chiave, punteggio."""

    def __init__(self, model_path: Path, score_threshold: float):
        check_model_file(
            model_path,
            "Scaricalo da https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet "
            "(vedi README).",
        )
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold=score_threshold
        )

    def detect(self, img: np.ndarray) -> np.ndarray:
        """Ritorna un array N x 15: x, y, w, h, 5 punti (x, y), punteggio."""
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))  # la risoluzione reale puo' variare
        _, faces = self.detector.detect(img)
        return np.empty((0, 15), np.float32) if faces is None else faces


class EmotionClassifier:
    """CNN TFLite che riceve un volto in grigio e restituisce le probabilita' delle emozioni."""

    def __init__(self, model_path: Path, labels: list):
        check_model_file(model_path)
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

        _, self.in_h, self.in_w, self.in_c = self.input["shape"]   # es. (1, 48, 48, 1)
        self.in_dtype = self.input["dtype"]                         # es. np.float32 o np.int8
        self.in_scale, self.in_zero = self.input["quantization"]
        self.out_scale, self.out_zero = self.output["quantization"]

        n_classes = int(self.output["shape"][-1])
        if n_classes != len(labels):
            raise SystemExit(f"Il modello ha {n_classes} classi ma LABELS ne contiene {len(labels)}")

    def model_view(self, face_gray: np.ndarray) -> np.ndarray:
        """Il volto esattamente come lo vede il modello (es. 48x48 in grigio)."""
        return cv2.resize(face_gray, (self.in_w, self.in_h), interpolation=cv2.INTER_AREA)

    def _to_tensor(self, face_small: np.ndarray) -> np.ndarray:
        if self.in_c == 1:
            face = face_small[..., np.newaxis]
        else:
            face = cv2.cvtColor(face_small, cv2.COLOR_GRAY2RGB)

        face = face.astype(np.float32) / 255.0
        if self.in_dtype != np.float32:
            # Modello quantizzato: valore_reale = scale * (q - zero_point)
            if self.in_scale:
                face = face / self.in_scale + self.in_zero
            info = np.iinfo(self.in_dtype)
            face = np.clip(np.round(face), info.min, info.max)
        return np.expand_dims(face.astype(self.in_dtype), axis=0)

    def predict(self, face_small: np.ndarray) -> np.ndarray:
        """Restituisce le probabilita' (somma = 1) per ogni emozione."""
        self.interpreter.set_tensor(self.input["index"], self._to_tensor(face_small))
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        if self.out_scale:
            out = self.out_scale * (out - self.out_zero)
        if not np.isclose(out.sum(), 1.0, atol=1e-2) or out.min() < 0:
            out = np.exp(out - out.max())  # logits -> softmax
            out /= out.sum()
        return out
