# Riconoscimento delle emozioni in tempo reale

Demo didattica di **machine learning** usata nelle attività di orientamento per
studenti delle scuole superiori: la webcam inquadra i volti e una rete neurale
riconosce l'espressione facciale, mostrando in tempo reale le probabilità per
ciascuna emozione.

Emozioni riconosciute: **Neutro, Felice, Triste, Sorpreso, Arrabbiato**.

## Come funziona

```
webcam ──► rilevamento volto ──► ritaglio 48×48 ──► CNN ──► probabilità
           (YuNet, OpenCV)       scala di grigi     (TFLite)   per emozione
```

1. **Acquisizione**: ogni frame della webcam viene specchiato (effetto specchio).
2. **Rilevamento del volto**: [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
   una piccola rete neurale inclusa in OpenCV, trova i volti nell'immagine.
3. **Preprocessing**: ogni volto viene ritagliato, convertito in scala di grigi,
   ridimensionato a 48×48 pixel e normalizzato in [0, 1].
4. **Classificazione**: una rete convoluzionale (`models/expression_tf_lite.tflite`)
   restituisce una probabilità per ciascuna delle 5 emozioni.
5. **Stabilizzazione**: le predizioni vengono mediate sugli ultimi frame per
   evitare che l'etichetta "sfarfalli".
6. **Visualizzazione**: riquadro sul volto, emozione più probabile e barre con
   tutte le probabilità.

## Requisiti

- Python 3.11
- Webcam
- [Git LFS](https://git-lfs.com) (il modello `.tflite` è salvato con Git LFS)

### Clonare il repository

Installa Git LFS **prima** di clonare, altrimenti al posto del modello
(circa 4 MB) scarichi un file di testo di 132 byte e lo script non parte:

```bash
brew install git-lfs   # macOS; su Linux: sudo apt install git-lfs
git lfs install
git clone https://github.com/vinzbarbuto/emotion-recognition.git
cd emotion-recognition
```

Se hai già clonato senza Git LFS, recupera il modello con:

```bash
git lfs install
git lfs pull
```

### Installare le dipendenze

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In alternativa con conda:

```bash
conda create -n emotion python=3.11
conda activate emotion
pip install -r requirements.txt
```

## Avvio

```bash
python main.py
```

Su macOS, al primo avvio, concedi al terminale l'accesso alla fotocamera
(Impostazioni di Sistema → Privacy e sicurezza → Fotocamera).

### Tasti

| Tasto       | Azione                                             |
|-------------|----------------------------------------------------|
| `1`         | Passo 1: l'immagine come griglia di numeri          |
| `2`         | Passo 2: rilevamento del volto e punti chiave       |
| `3`         | Passo 3: il volto 48×48 come lo vede il modello      |
| `4`         | Passo 4: predizione con barre di probabilità (default) |
| `0`         | tour automatico: passa da un passo all'altro da solo |
| `f`         | schermo intero on/off                               |
| `ESC` / `q` | esci                                                |

### I passi, per spiegare la demo

1. **L'immagine**: un riquadro ingrandisce 8×8 pixel presi tra gli occhi e ne
   mostra i valori (0 = nero, 255 = bianco). Per il computer la foto è solo questo.
2. **Trovo il volto**: il riquadro di YuNet, i 5 punti chiave (occhi, naso,
   angoli della bocca) e quanto la rete è sicura che sia un volto.
3. **Cosa vede il modello**: il ritaglio 48×48 in scala di grigi, ingrandito a
   "quadretti". Tutto quello che la rete ha a disposizione sono 2304 numeri.
4. **La predizione**: l'emozione più probabile e le barre con tutte le probabilità.

Con più persone, il pannello completo (barre al passo 4, miniatura al passo 3)
va solo al volto **principale**, cioè il più grande e quindi il più vicino. Gli
altri volti hanno etichetta e una barretta colorata sotto il riquadro. Il volto
principale cambia solo quando un altro diventa più grande del 20%
(`MAIN_SWITCH_RATIO`), e i pannelli si posizionano evitando gli altri volti.

Il riconoscimento gira sempre in sottofondo: cambiando passo cambia solo cosa
viene disegnato.

## Parametri

Tutti in [`emotion_recognition/config.py`](emotion_recognition/config.py), l'unico file da toccare per adattare la demo:

| Parametro              | Default | Significato                                        |
|------------------------|---------|----------------------------------------------------|
| `CAMERA_INDEX`         | 1       | webcam da usare (0 = integrata, 1 = esterna; dipende dal PC) |
| `TOUR_SECONDS`         | 6       | secondi per ogni passo nel tour automatico         |
| `START_MODE`           | 4       | passo mostrato all'avvio                           |
| `FACE_SCORE_THRESHOLD` | 0.8     | confidenza minima per considerare un volto         |
| `FACE_MIN_SIZE`        | 60      | dimensione minima del volto in pixel               |
| `CROP_MARGIN`          | 0.0     | margine extra attorno al volto prima del ritaglio  |
| `SMOOTHING_FRAMES`     | 8       | numero di frame su cui mediare le predizioni       |
| `MAIN_SWITCH_RATIO`    | 1.2     | quanto deve essere più grande un volto per diventare principale |
| `LABELS`               | italiano | etichette (c'è anche la versione inglese commentata) |

## Struttura del progetto

```
emotion-recognition/
├── main.py                    # avvio: python main.py
├── emotion_recognition/       # codice
│   ├── config.py              # parametri modificabili (webcam, soglie, testi, colori)
│   ├── models.py              # le due reti: YuNet (volti) e CNN TFLite (emozioni)
│   ├── pipeline.py            # frame -> volti ritagliati -> probabilità
│   ├── tracking.py            # media nel tempo e scelta del volto principale
│   ├── drawing.py             # intestazione e disegno dei 4 passi
│   └── app.py                 # loop della webcam e tasti
├── models/
│   ├── expression_tf_lite.tflite          # emozioni: input 48×48×1, 5 classi (Git LFS)
│   └── face_detection_yunet_2023mar.onnx  # rilevamento volti YuNet
├── requirements.txt
└── README.md
```

In alternativa a `python main.py` si può avviare anche con `python -m emotion_recognition`.

Se il modello YuNet manca, scaricalo con:

```bash
curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## Privacy

Lo script **non salva immagini né dati**: tutto viene elaborato in memoria, frame
per frame. Il codice per salvare una foto (tasto `s`) è presente ma commentato:
se lo riattivi, raccogli prima il consenso dei partecipanti (per i minorenni,
quello dei genitori) e non usare dati personali nei nomi dei file.

## Limiti (spunti per la discussione)

- Il modello vede solo un'immagine 48×48 in bianco e nero: molti dettagli si perdono.
- Occhiali, mani davanti al viso, testa girata o poca luce peggiorano le predizioni.
- Un'espressione non è un'emozione: il modello riconosce come appare il volto,
  non cosa prova la persona.
- Le prestazioni dipendono dal dataset di addestramento, che può non
  rappresentare allo stesso modo tutte le persone.

## Crediti

- Rilevamento volti: YuNet, [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (licenza MIT).
