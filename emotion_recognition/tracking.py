"""Coerenza tra un frame e l'altro: media delle predizioni e scelta del volto principale."""

from collections import deque

import numpy as np

from . import config


def _center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


class FaceTracker:
    """Associa i volti tra frame consecutivi per mediare le predizioni nel tempo."""

    def __init__(self):
        self.tracks = []  # lista di (centro, deque di probabilita')

    def smooth(self, detections):
        """detections: lista di (centro, probs). Ritorna probs mediate, stesso ordine."""
        new_tracks, smoothed, used = [], [], set()
        for center, probs in detections:
            best, best_d = None, config.TRACK_MAX_DIST
            for i, (c, _) in enumerate(self.tracks):
                d = np.hypot(center[0] - c[0], center[1] - c[1])
                if i not in used and d < best_d:
                    best, best_d = i, d
            if best is None:
                history = deque(maxlen=config.SMOOTHING_FRAMES)
            else:
                used.add(best)
                history = self.tracks[best][1]
            history.append(probs)
            new_tracks.append((center, history))
            smoothed.append(np.mean(history, axis=0))
        self.tracks = new_tracks
        return smoothed


class MainFaceSelector:
    """Sceglie il volto "principale" (il piu' grande, cioe' il piu' vicino).

    Solo lui riceve il pannello completo; gli altri una vista compatta.
    Isteresi: il principale cambia solo se un altro volto e' piu' grande di
    MAIN_SWITCH_RATIO, cosi' il pannello non salta da una persona all'altra.
    """

    def __init__(self):
        self.center = None

    def select(self, faces):
        """Imposta face["main"] = True sul volto principale."""
        for f in faces:
            f["main"] = False
        if not faces:
            self.center = None
            return

        def area(f):
            return f["box"][2] * f["box"][3]

        largest = max(faces, key=area)
        current = None
        if self.center is not None:  # il principale del frame prima, se e' ancora li'
            centers = [_center(f["box"]) for f in faces]
            dists = [np.hypot(cx - self.center[0], cy - self.center[1]) for cx, cy in centers]
            i = int(np.argmin(dists))
            if dists[i] < config.TRACK_MAX_DIST:
                current = faces[i]

        if current is None or area(largest) > config.MAIN_SWITCH_RATIO * area(current):
            current = largest
        current["main"] = True
        self.center = _center(current["box"])
