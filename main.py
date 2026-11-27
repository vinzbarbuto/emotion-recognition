"""
Riconoscimento delle espressioni facciali in tempo reale.

Avvio:  python main.py

Tasti:
  1-4       mostra un passo della pipeline
  0         tour automatico
  f         schermo intero on/off
  ESC / q   esci

I parametri (webcam, soglie, testi) sono in emotion_recognition/config.py.
"""

from emotion_recognition.app import main

if __name__ == "__main__":
    main()
