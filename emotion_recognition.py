"""
Riconoscimento delle espressioni facciali in tempo reale.

Pipeline:
  1. webcam  ->  2. rilevamento volto (YuNet)  ->  3. ritaglio + preprocessing
  ->  4. classificazione (CNN TFLite)  ->  5. barre di probabilita' a schermo

Tasti:
  ESC / q   esci
  f         schermo intero on/off
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

# --- Percorsi (relativi allo script, cosi' funziona da qualunque cartella) ---
BASE_DIR = Path(__file__).resolve().parent
EMOTION_MODEL = BASE_DIR / "expression_tf_lite.tflite"
FACE_MODEL = BASE_DIR / "face_detection_yunet_2023mar.onnx"
# PHOTO_DIR = BASE_DIR.parent / "images"

# --- Parametri ---
LABELS = ["Neutro", "Felice", "Triste", "Sorpreso", "Arrabbiato"]
# LABELS = ["Neutral", "Happy", "Sad", "Surprised", "Angry"]
BAR_COLORS = [  # BGR, un colore per emozione
    (180, 180, 180),  # Neutro
    (60, 200, 255),   # Felice
    (200, 120, 40),   # Triste
    (80, 220, 120),   # Sorpreso
    (60, 60, 230),    # Arrabbiato
]
BOX_COLOR = (245, 135, 66)

CAM_WIDTH, CAM_HEIGHT = 1280, 720
FACE_SCORE_THRESHOLD = 0.8   # confidenza minima di YuNet
FACE_MIN_SIZE = 60           # px: ignora volti troppo piccoli/lontani
CROP_MARGIN = 0.0            # margine extra attorno al box (0.1 = +10%)
SMOOTHING_FRAMES = 8         # media delle ultime N predizioni (anti-sfarfallio)
TRACK_MAX_DIST = 80          # px: distanza max per associare un volto al frame prima
WINDOW_NAME = "Riconoscimento emozioni"


# --- Caricamento modelli ---
def check_file(path: Path, hint: str = "") -> None:
    if not path.exists():
        raise SystemExit(f"File non trovato: {path}\n{hint}")


check_file(EMOTION_MODEL)
check_file(
    FACE_MODEL,
    "Scaricalo da https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet "
    "(vedi README).",
)

interpreter = Interpreter(model_path=str(EMOTION_MODEL))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

_, in_h, in_w, in_c = input_details["shape"]   # es. (1, 48, 48, 1)
in_dtype = input_details["dtype"]               # es. np.float32 o np.int8/uint8
in_scale, in_zero = input_details["quantization"]
out_scale, out_zero = output_details["quantization"]

n_classes = int(output_details["shape"][-1])
if n_classes != len(LABELS):
    raise SystemExit(f"Il modello ha {n_classes} classi ma LABELS ne contiene {len(LABELS)}")

face_detector = cv2.FaceDetectorYN.create(
    str(FACE_MODEL), "", (CAM_WIDTH, CAM_HEIGHT), score_threshold=FACE_SCORE_THRESHOLD
)


# --- Funzioni ---
def preprocess(face_gray: np.ndarray) -> np.ndarray:
    """Porta il volto in scala di grigi al formato atteso dal modello."""
    face = cv2.resize(face_gray, (in_w, in_h), interpolation=cv2.INTER_AREA)
    if in_c == 1:
        face = face[..., np.newaxis]
    else:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

    face = face.astype(np.float32) / 255.0
    if in_dtype != np.float32:
        # Modello quantizzato: valore_reale = scale * (q - zero_point)
        if in_scale:
            face = face / in_scale + in_zero
        info = np.iinfo(in_dtype)
        face = np.clip(np.round(face), info.min, info.max)
    return np.expand_dims(face.astype(in_dtype), axis=0)


def predict(face_gray: np.ndarray) -> np.ndarray:
    """Restituisce le probabilita' (somma = 1) per ogni emozione."""
    interpreter.set_tensor(input_details["index"], preprocess(face_gray))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details["index"])[0].astype(np.float32)
    if out_scale:
        out = out_scale * (out - out_zero)
    if not np.isclose(out.sum(), 1.0, atol=1e-2) or out.min() < 0:
        out = np.exp(out - out.max())  # logits -> softmax
        out /= out.sum()
    return out


def crop_face(gray: np.ndarray, x: int, y: int, w: int, h: int):
    """Ritaglio quadrato attorno al volto, limitato ai bordi dell'immagine."""
    side = int(max(w, h) * (1 + CROP_MARGIN))
    cx, cy = x + w // 2, y + h // 2
    x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x2, y2 = min(cx + side // 2, gray.shape[1]), min(cy + side // 2, gray.shape[0])
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return gray[y1:y2, x1:x2]


class FaceTracker:
    """Associa i volti tra frame consecutivi per mediare le predizioni nel tempo."""

    def __init__(self):
        self.tracks = []  # lista di (centro, deque di probabilita')

    def smooth(self, detections):
        """detections: lista di (centro, probs). Ritorna probs mediate, stesso ordine."""
        new_tracks, smoothed, used = [], [], set()
        for center, probs in detections:
            best, best_d = None, TRACK_MAX_DIST
            for i, (c, _) in enumerate(self.tracks):
                d = np.hypot(center[0] - c[0], center[1] - c[1])
                if i not in used and d < best_d:
                    best, best_d = i, d
            if best is None:
                history = deque(maxlen=SMOOTHING_FRAMES)
            else:
                used.add(best)
                history = self.tracks[best][1]
            history.append(probs)
            new_tracks.append((center, history))
            smoothed.append(np.mean(history, axis=0))
        self.tracks = new_tracks
        return smoothed


def draw_label(img, text, x, y):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    y_top = max(y - th - 12, 0)
    cv2.rectangle(img, (x, y_top), (x + tw + 12, y_top + th + 12), BOX_COLOR, -1)
    cv2.putText(img, text, (x + 6, y_top + th + 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_bars(img, probs, x, y, w, h):
    """Barre di probabilita' a destra del volto (a sinistra se non c'e' spazio)."""
    bar_w, bar_h, gap, label_w = 150, 22, 6, 120
    panel_w = label_w + bar_w + 70
    panel_h = len(LABELS) * (bar_h + gap) + gap
    img_h, img_w = img.shape[:2]
    if x + w + 10 + panel_w <= img_w:        # a destra
        px, py = x + w + 10, y
    elif x - panel_w - 10 >= 0:              # a sinistra
        px, py = x - panel_w - 10, y
    else:                                    # sotto (o sopra) il volto
        px = x
        py = y + h + 10 if y + h + 10 + panel_h <= img_h else y - panel_h - 40
    px = min(max(px, 0), max(img_w - panel_w, 0))
    py = min(max(py, 0), max(img_h - panel_h, 0))

    # Sfondo semitrasparente
    overlay = img.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)

    best = int(np.argmax(probs))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (label, p) in enumerate(zip(LABELS, probs)):
        by = py + gap + i * (bar_h + gap)
        bx = px + label_w
        weight = 2 if i == best else 1
        cv2.putText(img, label, (px + 6, by + bar_h - 6), font, 0.55, (255, 255, 255), weight, cv2.LINE_AA)
        cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), 1)
        cv2.rectangle(img, (bx, by), (bx + int(bar_w * float(p)), by + bar_h), BAR_COLORS[i], -1)
        cv2.putText(img, f"{p * 100:4.0f}%", (bx + bar_w + 6, by + bar_h - 6), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


# --- Main ---
def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Impossibile aprire la webcam (permessi fotocamera?)")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False
    tracker = FaceTracker()
    # photo_count = 0

    while True:
        ret, img = camera.read()
        if not ret:
            print("Frame non ricevuto dalla webcam, esco.")
            break
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # La risoluzione reale puo' differire da quella richiesta
        frame_h, frame_w = img.shape[:2]
        face_detector.setInputSize((frame_w, frame_h))
        _, faces = face_detector.detect(img)
        faces = [] if faces is None else faces

        boxes, detections = [], []
        for face in faces:
            x, y, w, h = map(int, face[:4])
            if min(w, h) < FACE_MIN_SIZE:
                continue
            roi = crop_face(gray, x, y, w, h)
            if roi is None:
                continue
            boxes.append((x, y, w, h))
            detections.append(((x + w / 2, y + h / 2), predict(roi)))

        for (x, y, w, h), probs in zip(boxes, tracker.smooth(detections)):
            best = int(np.argmax(probs))
            cv2.rectangle(img, (x, y), (x + w, y + h), BOX_COLOR, 2)
            draw_label(img, f"{LABELS[best]} {probs[best] * 100:.0f}%", x, y)
            draw_bars(img, probs, x, y, w, h)

        cv2.imshow(WINDOW_NAME, img)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):  # ESC o q
            break
        if k == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

        # --- Salvataggio foto (disattivato) ---
        # Riattivare solo con il consenso dei partecipanti (minorenni: serve
        # quello dei genitori). Nessun dato personale nel nome del file.
        # if k == ord("s"):
        #     PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        #     photo_count += 1
        #     path = PHOTO_DIR / f"foto_{photo_count:03d}.png"
        #     if cv2.imwrite(str(path), img):
        #         print(f"Foto salvata: {path}")
        #     else:
        #         print(f"Errore nel salvataggio di {path}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
