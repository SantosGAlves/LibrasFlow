"""
benchmark_fps.py
-----------------
Mede a latencia media e o FPS (quadros por segundo) de cada etapa do
pipeline de reconhecimento (captura -> MediaPipe Hands -> classificacao),
usando a webcam e o modelo reais.

Como usar:
    1. Coloque este arquivo na mesma pasta de main.py / app.py (ou ajuste
       CAMINHO_MODELO abaixo).
    2. Execute: python benchmark_fps.py
    3. Fique com a mao visivel para a camera durante a coleta.
    4. Ao final, copie os valores impressos para a Tabela 1 do artigo
       (colunas "Latencia media (ms)" e "FPS equivalente").

Este script NAO precisa de dataset nem de internet, apenas do arquivo
models/modelo_libras.p e de uma webcam conectada.
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pickle

# ----------------------------------------------------------------------
# Configuracao
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_MODELO = os.path.join(BASE_DIR, "models", "modelo_libras.p")
N_FRAMES = 300  # quantidade de quadros usados no benchmark

# ----------------------------------------------------------------------
# Carregamento do modelo e do MediaPipe (mesma configuracao do app.py)
# ----------------------------------------------------------------------
with open(CAMINHO_MODELO, "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir a webcam (indice 0).")

tempos_captura = []
tempos_mediapipe = []
tempos_classificacao = []
tempos_totais = []

print(f"Iniciando benchmark com {N_FRAMES} quadros.")
print("Mantenha a mao visivel para a camera. Pressione 'q' para interromper antes do fim.\n")

frames_processados = 0
t_inicio_geral = time.perf_counter()

while frames_processados < N_FRAMES:
    t0 = time.perf_counter()
    success, frame = cap.read()
    t1 = time.perf_counter()
    if not success:
        break
    tempos_captura.append((t1 - t0) * 1000)

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    t2 = time.perf_counter()
    results = hands.process(frame_rgb)
    t3 = time.perf_counter()
    tempos_mediapipe.append((t3 - t2) * 1000)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            coordenadas = []
            id0_x = hand_landmarks.landmark[0].x
            id0_y = hand_landmarks.landmark[0].y
            for lm in hand_landmarks.landmark:
                coordenadas.extend([lm.x - id0_x, lm.y - id0_y])

            max_val = max(list(map(abs, coordenadas)))
            if max_val != 0:
                coordenadas = [c / max_val for c in coordenadas]

            t4 = time.perf_counter()
            model.predict([coordenadas])
            t5 = time.perf_counter()
            tempos_classificacao.append((t5 - t4) * 1000)

    t_fim = time.perf_counter()
    tempos_totais.append((t_fim - t0) * 1000)
    frames_processados += 1

    cv2.imshow("Benchmark - pressione q para sair", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


def resumo(nome, lista):
    if not lista:
        print(f"{nome}: sem amostras (mao nao detectada nesses quadros?)")
        return
    arr = np.array(lista)
    media = arr.mean()
    fps = 1000 / media if media > 0 else float("inf")
    print(
        f"{nome}:\n"
        f"  media   = {media:.2f} ms\n"
        f"  mediana = {np.median(arr):.2f} ms\n"
        f"  p95     = {np.percentile(arr, 95):.2f} ms\n"
        f"  FPS equivalente = {fps:.1f}\n"
    )


print("=" * 70)
print(f"RESULTADOS DO BENCHMARK ({frames_processados} quadros processados)")
print("=" * 70)
resumo("Captura de quadro (OpenCV)", tempos_captura)
resumo("Deteccao de landmarks (MediaPipe Hands)", tempos_mediapipe)
resumo("Classificacao (Random Forest)", tempos_classificacao)
resumo("Pipeline completo (captura + MediaPipe + classificacao)", tempos_totais)

tempo_total = time.perf_counter() - t_inicio_geral
if tempo_total > 0:
    fps_real = frames_processados / tempo_total
    print(f"FPS real observado de ponta a ponta (quadros/segundo): {fps_real:.1f}")

print(
    "\nCopie os valores de 'media (ms)' e 'FPS equivalente' de cada etapa\n"
    "para a Tabela 1 do artigo (Secao 3 - Resultados e Discussao)."
)