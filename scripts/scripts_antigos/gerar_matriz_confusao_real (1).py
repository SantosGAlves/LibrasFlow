"""
gerar_matriz_confusao_real.py
-------------------------------
Versao corrigida de gerar_matriz_confusao.py.

O script original refaz o mesmo train_test_split de treinar.py dentro de
data/libras_dados.csv. Se o dataset de treino tem quadros muito parecidos
entre si (varias fotos da mesma sessao/rajada para a mesma letra -- como
os arquivos B1.jpg/B10.jpg/B100.jpg e 1.png/2.png/3.png mostram), o split
aleatorio deixa quadros quase identicos dos dois lados (treino e teste),
inflando a acuracia para ~100% e deixando a matriz de confusao sem
nenhuma confusao real -- ou seja, inutil para analise.

Este script usa dataset/val (a mesma pasta que validar_real.py usa para
o "Teste Cego Real"): imagens que NUNCA passam por coleta.py, nunca
entram no CSV e nunca sao vistas pelo modelo durante o treino. Cada
predicao aqui e sobre uma imagem genuinamente inedita.

IMPORTANTE (leia antes de colocar no artigo):
    Se dataset/val foi capturado na MESMA sessao/pessoa/fundo/iluminacao
    que dataset/train, a acuracia aqui ainda pode ficar otimista --
    apenas nao vai ter duplicatas literais. Para um numero robusto de
    verdade, o ideal e dataset/val vir de uma pessoa, fundo ou dia
    diferente do dataset/train (por isso o README ja separa "acuracia em
    ambiente controlado" de "cenarios reais").

Como usar:
    1. Garanta que dataset/val/<LETRA>/*.png|jpg existe (mesma estrutura
       de dataset/train).
    2. Coloque este arquivo em scripts/ (mesma pasta de treinar.py).
    3. Execute: python gerar_matriz_confusao_real.py
    4. Use models/matriz_confusao_real.png e
       models/classification_report_real.txt no artigo, no lugar dos
       arquivos antigos (que ficam guardados para efeito de comparacao/
       discussao sobre o vazamento de dados).
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import pickle
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_MODELO = os.path.join(BASE_DIR, "models", "modelo_libras.p")
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
