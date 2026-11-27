"""
Riconoscimento delle espressioni facciali in tempo reale.

Pipeline:
  1. webcam  ->  2. rilevamento volto (YuNet)  ->  3. ritaglio + preprocessing
  ->  4. classificazione (CNN TFLite)  ->  barre di probabilita' a schermo

Tasti:
  1-4       mostra un passo della pipeline
  0         tour automatico (cambia passo ogni TOUR_SECONDS secondi)
  f         schermo intero on/off
  ESC / q   esci
"""

import time
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
LANDMARK_COLORS = [(0, 255, 255), (0, 255, 255), (0, 200, 0), (255, 0, 255), (255, 0, 255)]

CAMERA_INDEX = 1             # 0 = webcam integrata, 1 = webcam esterna (dipende dal PC)
CAM_WIDTH, CAM_HEIGHT = 1280, 720
FACE_SCORE_THRESHOLD = 0.8   # confidenza minima di YuNet
FACE_MIN_SIZE = 60           # px: ignora volti troppo piccoli/lontani
CROP_MARGIN = 0.0            # margine extra attorno al box (0.1 = +10%)
SMOOTHING_FRAMES = 8         # media delle ultime N predizioni (anti-sfarfallio)
TRACK_MAX_DIST = 80          # px: distanza max per associare un volto al frame prima
MAIN_SWITCH_RATIO = 1.2      # un altro volto diventa "principale" se piu' grande del 20%
TOUR_SECONDS = 6            # durata di ogni passo nel tour automatico
START_MODE = 4               # passo mostrato all'avvio
WINDOW_NAME = "Riconoscimento emozioni"

# Titolo e spiegazione di ogni passo (niente lettere accentate: i font di OpenCV non le supportano)
STEPS = {
    1: ("Passo 1: l'immagine",
        "Per il computer una foto e' solo una griglia di numeri (0 = nero, 255 = bianco)"),
    2: ("Passo 2: trovo il volto",
        "Una prima rete neurale (YuNet) cerca i volti e 5 punti chiave: occhi, naso, bocca"),
    3: ("Passo 3: cosa vede il modello",
        "Il volto diventa 48 x 48 pixel in bianco e nero: in tutto 2304 numeri"),
    4: ("Passo 4: la predizione",
        "Una seconda rete neurale da' una probabilita' a ogni emozione"),
}


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


# --- Elaborazione ---
def model_view(face_gray: np.ndarray) -> np.ndarray:
    """Il volto esattamente come lo vede il modello (es. 48x48 in grigio)."""
    return cv2.resize(face_gray, (in_w, in_h), interpolation=cv2.INTER_AREA)


def preprocess(face_small: np.ndarray) -> np.ndarray:
    """Porta il volto ridimensionato al formato tensore atteso dal modello."""
    if in_c == 1:
        face = face_small[..., np.newaxis]
    else:
        face = cv2.cvtColor(face_small, cv2.COLOR_GRAY2RGB)

    face = face.astype(np.float32) / 255.0
    if in_dtype != np.float32:
        # Modello quantizzato: valore_reale = scale * (q - zero_point)
        if in_scale:
            face = face / in_scale + in_zero
        info = np.iinfo(in_dtype)
        face = np.clip(np.round(face), info.min, info.max)
    return np.expand_dims(face.astype(in_dtype), axis=0)


def predict(face_small: np.ndarray) -> np.ndarray:
    """Restituisce le probabilita' (somma = 1) per ogni emozione."""
    interpreter.set_tensor(input_details["index"], preprocess(face_small))
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


def analyze(img, gray, tracker):
    """Trova i volti e classifica l'espressione. Ritorna una lista di dict, uno per volto."""
    frame_h, frame_w = img.shape[:2]
    face_detector.setInputSize((frame_w, frame_h))  # la risoluzione reale puo' variare
    _, faces = face_detector.detect(img)
    faces = [] if faces is None else faces

    results = []
    for face in faces:
        x, y, w, h = map(int, face[:4])
        if min(w, h) < FACE_MIN_SIZE:
            continue
        roi = crop_face(gray, x, y, w, h)
        if roi is None:
            continue
        small = model_view(roi)
        results.append({
            "box": (x, y, w, h),
            "landmarks": face[4:14].reshape(5, 2).astype(int),
            "score": float(face[14]),
            "small": small,
            "probs": predict(small),
        })

    smoothed = tracker.smooth([((r["box"][0] + r["box"][2] / 2, r["box"][1] + r["box"][3] / 2),
                                r["probs"]) for r in results])
    for r, probs in zip(results, smoothed):
        r["probs"] = probs
        r["main"] = False
    return results


class MainFaceSelector:
    """Sceglie il volto "principale" (il piu' grande, cioe' il piu' vicino).

    Solo lui riceve il pannello completo; gli altri una vista compatta.
    Isteresi: il principale cambia solo se un altro volto e' piu' grande di
    MAIN_SWITCH_RATIO, cosi' il pannello non salta da una persona all'altra.
    """

    def __init__(self):
        self.center = None

    def select(self, faces):
        if not faces:
            self.center = None
            return

        def area(f):
            return f["box"][2] * f["box"][3]

        def center(f):
            x, y, w, h = f["box"]
            return x + w / 2, y + h / 2

        largest = max(faces, key=area)
        current = None
        if self.center is not None:  # il principale del frame prima, se e' ancora li'
            dists = [np.hypot(center(f)[0] - self.center[0], center(f)[1] - self.center[1]) for f in faces]
            i = int(np.argmin(dists))
            if dists[i] < TRACK_MAX_DIST:
                current = faces[i]

        if current is None or area(largest) > MAIN_SWITCH_RATIO * area(current):
            current = largest
        current["main"] = True
        self.center = center(current)


# --- Disegno: elementi comuni ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)


def shade(img, x1, y1, x2, y2, alpha=0.6, color=(30, 30, 30)):
    """Rettangolo semitrasparente."""
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, dst=roi)


def place_panel(img, box, panel_w, panel_h, avoid=()):
    """Posizione per un pannello vicino al volto: destra, sinistra, sotto o sopra.

    Sceglie la posizione che copre meno gli altri volti (`avoid`) e il volto stesso;
    a parita', vale l'ordine destra > sinistra > sotto > sopra.
    """
    x, y, w, h = box
    img_h, img_w = img.shape[:2]
    candidates = [
        (x + w + 10, y),              # destra
        (x - panel_w - 10, y),        # sinistra
        (x, y + h + 10),              # sotto
        (x, y - panel_h - 40),        # sopra (lascia spazio all'etichetta)
    ]

    def overlap(px, py, bx, by, bw, bh):
        return max(0, min(px + panel_w, bx + bw) - max(px, bx)) * max(0, min(py + panel_h, by + bh) - max(py, by))

    best, best_cost = None, None
    for px, py in candidates:
        px = min(max(px, 0), max(img_w - panel_w, 0))
        py = min(max(py, 0), max(img_h - panel_h, 0))
        cost = sum(overlap(px, py, *b) for b in (box, *avoid))
        if best_cost is None or cost < best_cost:
            best, best_cost = (px, py), cost
    return best


def draw_label(img, text, x, y):
    scale, thick = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    y_top = max(y - th - 12, 0)
    cv2.rectangle(img, (x, y_top), (x + tw + 12, y_top + th + 12), BOX_COLOR, -1)
    cv2.putText(img, text, (x + 6, y_top + th + 5), FONT, scale, WHITE, thick, cv2.LINE_AA)


def draw_header(img, step, tour, tour_progress):
    """Titolo del passo in alto e legenda dei tasti in basso."""
    img_h, img_w = img.shape[:2]
    s = img_w / 1280  # scala il testo con la risoluzione
    title, subtitle = STEPS[step]
    bar_h = int(95 * s)
    shade(img, 0, 0, img_w, bar_h, alpha=0.65)
    cv2.putText(img, title, (int(20 * s), int(42 * s)), FONT, 1.2 * s, WHITE, max(int(3 * s), 1), cv2.LINE_AA)
    cv2.putText(img, subtitle, (int(20 * s), int(78 * s)), FONT, 0.75 * s, (220, 220, 220),
                max(int(2 * s), 1), cv2.LINE_AA)

    # Indicatore dei passi: 1 2 3 4 con quello attivo evidenziato
    for i in range(1, 5):
        cx, cy, r = img_w - int((5 - i) * 50 * s) - int(20 * s), int(bar_h / 2), int(18 * s)
        active = i == step
        cv2.circle(img, (cx, cy), r, BOX_COLOR if active else (90, 90, 90), -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(str(i), FONT, 0.7 * s, 2)
        cv2.putText(img, str(i), (cx - tw // 2, cy + th // 2), FONT, 0.7 * s, WHITE, 2, cv2.LINE_AA)

    if tour:  # barra di avanzamento del tour
        cv2.rectangle(img, (0, bar_h - 4), (int(img_w * tour_progress), bar_h), BOX_COLOR, -1)

    hint = "1-4 passi   0 tour automatico   f schermo intero   q esci"
    if tour:
        hint = "TOUR AUTOMATICO  -  " + hint
    (tw, th), _ = cv2.getTextSize(hint, FONT, 0.5 * s, 1)
    shade(img, 0, img_h - th - int(16 * s), tw + int(24 * s), img_h, alpha=0.5)
    cv2.putText(img, hint, (int(12 * s), img_h - int(8 * s)), FONT, 0.5 * s, WHITE, 1, cv2.LINE_AA)


# --- Disegno: un passo per modalita' ---
def draw_step1_pixels(img, gray, faces):
    """Ingrandisce una griglia 8x8 di pixel e ne scrive i valori numerici."""
    n, cell = 8, 44
    img_h, img_w = img.shape[:2]
    if faces:  # zona tra gli occhi del primo volto, altrimenti centro dell'immagine
        cx, cy = faces[0]["landmarks"][:2].mean(axis=0).astype(int)
    else:
        cx, cy = img_w // 2, img_h // 2
    x0 = int(np.clip(cx - n // 2, 0, img_w - n))
    y0 = int(np.clip(cy - n // 2, 0, img_h - n))
    patch = gray[y0:y0 + n, x0:x0 + n]

    panel = n * cell
    px, py = img_w - panel - 30, img_h - panel - 60
    # Collegamento tra la zona sorgente e il pannello ingrandito
    cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + n + 2, y0 + n + 2), (0, 255, 255), 2)
    cv2.line(img, (x0 + n + 2, y0 + n // 2), (px, py + panel // 2), (0, 255, 255), 1, cv2.LINE_AA)

    for r in range(n):
        for c in range(n):
            v = int(patch[r, c])
            x1, y1 = px + c * cell, py + r * cell
            cv2.rectangle(img, (x1, y1), (x1 + cell, y1 + cell), (v, v, v), -1)
            txt_color = (0, 0, 0) if v > 127 else WHITE
            (tw, th), _ = cv2.getTextSize(str(v), FONT, 0.42, 1)
            cv2.putText(img, str(v), (x1 + (cell - tw) // 2, y1 + (cell + th) // 2), FONT, 0.42,
                        txt_color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (px, py), (px + panel, py + panel), (0, 255, 255), 2)
    shade(img, px, py - 32, px + panel, py - 4, alpha=0.6)
    cv2.putText(img, f"8 x 8 pixel ingranditi (su {img_w} x {img_h})", (px + 8, py - 12), FONT, 0.5,
                WHITE, 1, cv2.LINE_AA)


def draw_step2_detection(img, face):
    x, y, w, h = face["box"]
    cv2.rectangle(img, (x, y), (x + w, y + h), BOX_COLOR, 2)
    for (lx, ly), color in zip(face["landmarks"], LANDMARK_COLORS):
        cv2.circle(img, (int(lx), int(ly)), 5, color, -1, cv2.LINE_AA)
    draw_label(img, f"Volto: {face['score'] * 100:.0f}% sicuro", x, y)


def draw_step3_model_view(img, face):
    x, y, w, h = face["box"]
    cv2.rectangle(img, (x, y), (x + w, y + h), BOX_COLOR, 2)
    if not face["main"]:  # miniatura solo per il volto principale
        return
    size = 240
    # INTER_NEAREST: ogni pixel diventa un quadretto ben visibile
    big = cv2.resize(face["small"], (size, size), interpolation=cv2.INTER_NEAREST)
    px, py = place_panel(img, face["box"], size, size + 36, face["others"])
    img[py:py + size, px:px + size] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (px, py), (px + size, py + size), BOX_COLOR, 2)
    shade(img, px, py + size, px + size, py + size + 36, alpha=0.7)
    cv2.putText(img, f"{in_w} x {in_h} pixel", (px + 8, py + size + 25), FONT, 0.65, WHITE, 2, cv2.LINE_AA)


def draw_step4_prediction(img, face):
    x, y, w, h = face["box"]
    probs = face["probs"]
    best = int(np.argmax(probs))
    cv2.rectangle(img, (x, y), (x + w, y + h), BOX_COLOR, 2)
    draw_label(img, f"{LABELS[best]} {probs[best] * 100:.0f}%", x, y)

    if not face["main"]:
        # Vista compatta: barretta sotto il box, colore dell'emozione, lunga quanto la confidenza
        by = min(y + h + 4, img.shape[0] - 8)
        cv2.rectangle(img, (x, by), (x + w, by + 8), (60, 60, 60), -1)
        cv2.rectangle(img, (x, by), (x + int(w * float(probs[best])), by + 8), BAR_COLORS[best], -1)
        return

    bar_w, bar_h, gap, label_w = 150, 22, 6, 120
    panel_w = label_w + bar_w + 70
    panel_h = len(LABELS) * (bar_h + gap) + gap
    px, py = place_panel(img, face["box"], panel_w, panel_h, face["others"])
    shade(img, px, py, px + panel_w, py + panel_h)
    for i, (label, p) in enumerate(zip(LABELS, probs)):
        by = py + gap + i * (bar_h + gap)
        bx = px + label_w
        weight = 2 if i == best else 1
        cv2.putText(img, label, (px + 6, by + bar_h - 6), FONT, 0.55, WHITE, weight, cv2.LINE_AA)
        cv2.rectangle(img, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), 1)
        cv2.rectangle(img, (bx, by), (bx + int(bar_w * float(p)), by + bar_h), BAR_COLORS[i], -1)
        cv2.putText(img, f"{p * 100:4.0f}%", (bx + bar_w + 6, by + bar_h - 6), FONT, 0.5, WHITE, 1,
                    cv2.LINE_AA)


def render(img, gray, faces, step, tour=False, tour_progress=0.0):
    """Disegna sul frame il passo della pipeline scelto."""
    if step == 1:
        draw_step1_pixels(img, gray, faces)
    else:
        draw = {2: draw_step2_detection, 3: draw_step3_model_view, 4: draw_step4_prediction}[step]
        for face in faces:  # box degli altri volti, che i pannelli devono evitare
            face["others"] = [f["box"] for f in faces if f is not face]
        for face in sorted(faces, key=lambda f: f["main"]):  # principale per ultimo, sopra gli altri
            draw(img, face)
    draw_header(img, step, tour, tour_progress)


# --- Main ---
def main():
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise SystemExit(f"Impossibile aprire la webcam {CAMERA_INDEX} (permessi fotocamera? "
                         "prova a cambiare CAMERA_INDEX)")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False
    tracker = FaceTracker()
    selector = MainFaceSelector()
    step, tour, tour_start = START_MODE, False, 0.0
    # photo_count = 0

    while True:
        ret, img = camera.read()
        if not ret:
            print("Frame non ricevuto dalla webcam, esco.")
            break
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Il riconoscimento gira sempre: cambia solo cosa viene disegnato
        faces = analyze(img, gray, tracker)
        selector.select(faces)

        tour_progress = 0.0
        if tour:
            elapsed = (time.monotonic() - tour_start) / TOUR_SECONDS
            step = 1 + int(elapsed) % 4
            tour_progress = elapsed % 1
        render(img, gray, faces, step, tour, tour_progress)

        cv2.imshow(WINDOW_NAME, img)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):  # ESC o q
            break
        if ord("1") <= k <= ord("4"):
            step, tour = k - ord("0"), False
        elif k == ord("0"):
            tour, tour_start = True, time.monotonic()
        elif k == ord("f"):
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
