"""Parametri della demo: e' l'unico file da modificare per adattarla."""

from pathlib import Path

# --- Percorsi ---
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
EMOTION_MODEL = MODELS_DIR / "expression_tf_lite.tflite"
FACE_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
# PHOTO_DIR = ROOT_DIR.parent / "images"

# --- Emozioni (stesso ordine delle uscite del modello) ---
LABELS = ["Neutro", "Felice", "Triste", "Sorpreso", "Arrabbiato"]
# LABELS = ["Neutral", "Happy", "Sad", "Surprised", "Angry"]

# --- Webcam ---
CAMERA_INDEX = 1             # 0 = webcam integrata, 1 = webcam esterna (dipende dal PC)
CAM_WIDTH, CAM_HEIGHT = 1280, 720
WINDOW_NAME = "Riconoscimento emozioni"

# --- Rilevamento e classificazione ---
FACE_SCORE_THRESHOLD = 0.8   # confidenza minima di YuNet
FACE_MIN_SIZE = 60           # px: ignora volti troppo piccoli/lontani
CROP_MARGIN = 0.0            # margine extra attorno al box (0.1 = +10%)

# --- Stabilita' tra un frame e l'altro ---
SMOOTHING_FRAMES = 8         # media delle ultime N predizioni (anti-sfarfallio)
TRACK_MAX_DIST = 80          # px: distanza max per associare un volto al frame prima
MAIN_SWITCH_RATIO = 1.2      # un altro volto diventa "principale" se piu' grande del 20%

# --- Passi della pipeline ---
TOUR_SECONDS = 6             # durata di ogni passo nel tour automatico
START_MODE = 4               # passo mostrato all'avvio

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

# --- Colori (BGR) ---
BAR_COLORS = [
    (180, 180, 180),  # Neutro
    (60, 200, 255),   # Felice
    (200, 120, 40),   # Triste
    (80, 220, 120),   # Sorpreso
    (60, 60, 230),    # Arrabbiato
]
BOX_COLOR = (245, 135, 66)
LANDMARK_COLORS = [(0, 255, 255), (0, 255, 255), (0, 200, 0), (255, 0, 255), (255, 0, 255)]
