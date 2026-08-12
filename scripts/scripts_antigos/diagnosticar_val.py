"""
diagnosticar_val.py
----------------------
Diagnostico detalhado da avaliacao em dataset/val, para entender POR QUE
letras especificas (ex.: E, M, N, R, S, U) tem desempenho ruim.

O que faz, alem do classification_report:
    1. Mostra a TAXA DE FALHA DE DETECCAO DE MAO por letra -- se uma
       letra falha muito aqui, o problema e a extracao (MediaPipe nao
       consegue achar a mao na imagem), nao o classificador.
    2. Salva ate N imagens ERRADAS por letra, com os landmarks
       desenhados por cima e o rotulo real vs. previsto escritos na
       imagem, em models/diagnostico_val/ -- para voce checar visualmente
       se o sinal esta correto, se a mao esta cortada, se e a mao
       errada (esquerda/direita), etc.

Como usar:
    python diagnosticar_val.py
    python diagnosticar_val.py --letras E,M,N,R,S,U     # so essas letras
    python diagnosticar_val.py --max-exemplos 10          # mais exemplos salvos
"""

import argparse
import os
import warnings

import cv2
import mediapipe as mp
import pickle
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_MODELO = os.path.join(BASE_DIR, "models", "modelo_libras.p")
CAMINHO_VALIDACAO = os.path.join(BASE_DIR, "dataset", "val")
PASTA_DIAGNOSTICO = os.path.join(BASE_DIR, "models", "diagnostico_val")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--letras", type=str, default=None,
                         help="Letras a inspecionar, separadas por virgula (ex.: E,M,N). Padrao: todas.")
    parser.add_argument("--max-exemplos", type=int, default=6,
                         help="Maximo de imagens erradas salvas por letra")
    args = parser.parse_args()
    letras_filtro = set(args.letras.split(",")) if args.letras else None

    with open(CAMINHO_MODELO, "rb") as f:
        model = pickle.load(f)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    if not os.path.exists(CAMINHO_VALIDACAO):
        raise SystemExit(f"Pasta nao encontrada: {CAMINHO_VALIDACAO}")

    os.makedirs(PASTA_DIAGNOSTICO, exist_ok=True)

    y_real, y_pred = [], []

    print(f"{'Letra':6} | {'Total':>6} | {'MaoOK':>6} | {'FalhouDeteccao':>14} | {'%Falha':>7} | {'Erros salvos':>12}")
    print("-" * 70)

    for letra_pasta in sorted(os.listdir(CAMINHO_VALIDACAO)):
        caminho_letra = os.path.join(CAMINHO_VALIDACAO, letra_pasta)
        if not os.path.isdir(caminho_letra):
            continue
        if letras_filtro and letra_pasta not in letras_filtro:
            continue

        total = 0
        mao_ok = 0
        salvos_errados = 0

        for nome_img in sorted(os.listdir(caminho_letra)):
            if not nome_img.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            total += 1

            imagem = cv2.imread(os.path.join(caminho_letra, nome_img))
            if imagem is None:
                continue

            imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            results = hands.process(imagem_rgb)
            if not results.multi_hand_landmarks:
                continue
            mao_ok += 1

            for hand_landmarks in results.multi_hand_landmarks:
                coordenadas = []
                id0_x = hand_landmarks.landmark[0].x
                id0_y = hand_landmarks.landmark[0].y
                for lm in hand_landmarks.landmark:
                    coordenadas.extend([lm.x - id0_x, lm.y - id0_y])

                max_val = max(list(map(abs, coordenadas)))
                if max_val != 0:
                    coordenadas = [c / max_val for c in coordenadas]

                predicao = model.predict([coordenadas])[0]
                y_real.append(letra_pasta)
                y_pred.append(predicao)

                if predicao != letra_pasta and salvos_errados < args.max_exemplos:
                    vis = imagem.copy()
                    mp_drawing.draw_landmarks(vis, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(vis, f"real:{letra_pasta} previsto:{predicao}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    saida = os.path.join(PASTA_DIAGNOSTICO, f"{letra_pasta}_errado_{nome_img}")
                    cv2.imwrite(saida, vis)
                    salvos_errados += 1

        pct_falha = 100 * (total - mao_ok) / total if total else 0
        print(f"{letra_pasta:6} | {total:6d} | {mao_ok:6d} | {total - mao_ok:14d} | {pct_falha:6.1f}% | {salvos_errados:12d}")

    print(f"\nExemplos de erros (com landmarks desenhados) salvos em: {PASTA_DIAGNOSTICO}")
    print("Abra essas imagens e confira: o sinal esta certo? a mao aparece cortada/de lado?")
    print("os landmarks desenhados fazem sentido (nao estao tortos/deslocados)?\n")

    if y_real:
        print(classification_report(y_real, y_pred, zero_division=0))
    else:
        print("Nenhuma mao detectada nas letras filtradas.")


if __name__ == "__main__":
    main()
