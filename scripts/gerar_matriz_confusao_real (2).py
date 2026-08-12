"""
gerar_matriz_confusao_real.py
-------------------------------
Gera a matriz de confusao e o relatorio de classificacao usando dados
NUNCA vistos pelo modelo: as imagens de dataset/val (o mesmo conjunto
que validar_real.py usa para o "Teste Cego Real").

IMPORTANTE (leia antes de colocar no artigo):
    Se dataset/val foi capturado na MESMA sessao/pessoa/fundo/iluminacao
    que dataset/train, a acuracia aqui ainda pode ficar otimista.

Como usar:
    1. Garanta que dataset/val/<LETRA>/*.png|jpg existe.
    2. Execute: python gerar_matriz_confusao_real.py
    3. Use models/matriz_confusao_real.png e
       models/classification_report_real.txt no artigo.
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pickle
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libras_features import extrair_features

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_MODELO = os.path.join(BASE_DIR, "models", "modelo_libras_final.p")
CAMINHO_VALIDACAO = os.path.join(BASE_DIR, "dataset", "val")
SAIDA_FIGURA = os.path.join(BASE_DIR, "models", "matriz_confusao_real.png")
SAIDA_RELATORIO = os.path.join(BASE_DIR, "models", "classification_report_real.txt")

with open(CAMINHO_MODELO, "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

if not os.path.exists(CAMINHO_VALIDACAO):
    raise SystemExit(f"Pasta de validacao nao encontrada em: {CAMINHO_VALIDACAO}")

y_real = []
y_pred = []
sem_mao_detectada = 0

print(f"Extraindo landmarks das imagens INEDITAS em: {CAMINHO_VALIDACAO}\n")

for letra_pasta in sorted(os.listdir(CAMINHO_VALIDACAO)):
    caminho_letra = os.path.join(CAMINHO_VALIDACAO, letra_pasta)
    if not os.path.isdir(caminho_letra):
        continue

    n_letra = 0
    for nome_img in os.listdir(caminho_letra):
        if not nome_img.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        imagem = cv2.imread(os.path.join(caminho_letra, nome_img))
        if imagem is None:
            continue

        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        results = hands.process(imagem_rgb)
        if not results.multi_hand_landmarks:
            sem_mao_detectada += 1
            continue

        for hand_landmarks in results.multi_hand_landmarks:
            coordenadas = extrair_features(hand_landmarks)

            predicao = model.predict([coordenadas])[0]
            y_real.append(letra_pasta)
            y_pred.append(predicao)
            n_letra += 1

    print(f"  {letra_pasta}: {n_letra} imagens avaliadas")

print(
    f"\nTotal avaliado: {len(y_real)} imagens "
    f"({sem_mao_detectada} descartadas por falha na deteccao da mao)\n"
)

if not y_real:
    raise SystemExit("Nenhuma mao detectada em dataset/val. Verifique as imagens.")

relatorio = classification_report(y_real, y_pred, zero_division=0)
print(relatorio)

os.makedirs(os.path.dirname(SAIDA_RELATORIO), exist_ok=True)
with open(SAIDA_RELATORIO, "w", encoding="utf-8") as f:
    f.write(relatorio)
print(f"Relatorio salvo em: {SAIDA_RELATORIO}")

labels = sorted(set(y_real) | set(y_pred))
cm = confusion_matrix(y_real, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(10, 9))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
ax.set_title("Matriz de Confusao - Teste Real (dataset/val, nunca visto no treino)")
plt.tight_layout()
plt.savefig(SAIDA_FIGURA, dpi=200)
print(f"Matriz de confusao salva em: {SAIDA_FIGURA}")
