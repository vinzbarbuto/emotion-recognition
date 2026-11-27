"""Loop principale: webcam, tasti e finestra."""

import time

import cv2

from . import config
from .drawing import render
from .pipeline import EmotionPipeline


def main():
    pipeline = EmotionPipeline()

    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    if not camera.isOpened():
        raise SystemExit(f"Impossibile aprire la webcam {config.CAMERA_INDEX} (permessi fotocamera? "
                         "prova a cambiare CAMERA_INDEX in emotion_recognition/config.py)")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False
    step, tour, tour_start = config.START_MODE, False, 0.0
    # photo_count = 0

    while True:
        ret, img = camera.read()
        if not ret:
            print("Frame non ricevuto dalla webcam, esco.")
            break
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Il riconoscimento gira sempre: cambia solo cosa viene disegnato
        faces = pipeline.analyze(img, gray)

        tour_progress = 0.0
        if tour:
            elapsed = (time.monotonic() - tour_start) / config.TOUR_SECONDS
            step = 1 + int(elapsed) % 4
            tour_progress = elapsed % 1
        render(img, gray, faces, step, tour, tour_progress)

        cv2.imshow(config.WINDOW_NAME, img)
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
                config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )

        # --- Salvataggio foto (disattivato) ---
        # Riattivare solo con il consenso dei partecipanti (minorenni: serve
        # quello dei genitori). Nessun dato personale nel nome del file.
        # Serve anche decommentare PHOTO_DIR in config.py.
        # if k == ord("s"):
        #     config.PHOTO_DIR.mkdir(parents=True, exist_ok=True)
        #     photo_count += 1
        #     path = config.PHOTO_DIR / f"foto_{photo_count:03d}.png"
        #     if cv2.imwrite(str(path), img):
        #         print(f"Foto salvata: {path}")
        #     else:
        #         print(f"Errore nel salvataggio di {path}")

    camera.release()
    cv2.destroyAllWindows()
