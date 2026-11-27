"""Tutto cio' che viene disegnato sul frame: intestazione e i 4 passi della pipeline."""

import cv2
import numpy as np

from .config import BAR_COLORS, BOX_COLOR, LABELS, LANDMARK_COLORS, STEPS

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)


# --- Elementi comuni ---
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


# --- Un passo per modalita' ---
def draw_step1_pixels(img, gray, faces):
    """Ingrandisce una griglia 8x8 di pixel e ne scrive i valori numerici."""
    n, cell = 8, 44
    img_h, img_w = img.shape[:2]
    if faces:  # zona tra gli occhi del volto principale, altrimenti centro dell'immagine
        main = next((f for f in faces if f.get("main")), faces[0])
        cx, cy = main["landmarks"][:2].mean(axis=0).astype(int)
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
    small_h, small_w = face["small"].shape[:2]
    # INTER_NEAREST: ogni pixel diventa un quadretto ben visibile
    big = cv2.resize(face["small"], (size, size), interpolation=cv2.INTER_NEAREST)
    px, py = place_panel(img, face["box"], size, size + 36, face["others"])
    img[py:py + size, px:px + size] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (px, py), (px + size, py + size), BOX_COLOR, 2)
    shade(img, px, py + size, px + size, py + size + 36, alpha=0.7)
    cv2.putText(img, f"{small_w} x {small_h} pixel", (px + 8, py + size + 25), FONT, 0.65, WHITE, 2,
                cv2.LINE_AA)


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


STEP_DRAWERS = {2: draw_step2_detection, 3: draw_step3_model_view, 4: draw_step4_prediction}


def render(img, gray, faces, step, tour=False, tour_progress=0.0):
    """Disegna sul frame il passo della pipeline scelto."""
    if step == 1:
        draw_step1_pixels(img, gray, faces)
    else:
        for face in faces:  # box degli altri volti, che i pannelli devono evitare
            face["others"] = [f["box"] for f in faces if f is not face]
        for face in sorted(faces, key=lambda f: f["main"]):  # principale per ultimo, sopra gli altri
            STEP_DRAWERS[step](img, face)
    draw_header(img, step, tour, tour_progress)
