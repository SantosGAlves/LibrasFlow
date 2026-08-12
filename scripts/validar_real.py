import os
import sys
import cv2
import mediapipe as mp
import pickle
import warnings
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libras_features import extrair_features

warnings.filterwarnings('ignore', category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_MODELO = os.path.join(BASE_DIR, 'models', 'modelo_libras_final.p')
CAMINHO_VALIDACAO = os.path.join(BASE_DIR, 'dataset', 'val')

try:
    with open(CAMINHO_MODELO, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Erro: Modelo não encontrado em {CAMINHO_MODELO}. Treine o modelo primeiro.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

y_real = []
y_predito = []

if not os.path.exists(CAMINHO_VALIDACAO):
    print(f"Erro: Pasta de validação inédita não encontrada em: {CAMINHO_VALIDACAO}")
    exit()

print(f"Iniciando Teste Cego Real a partir de: {CAMINHO_VALIDACAO}\n")

for letra_pasta in os.listdir(CAMINHO_VALIDACAO):
    caminho_letra = os.path.join(CAMINHO_VALIDACAO, letra_pasta)

    if os.path.isdir(caminho_letra):
        print(f"Processando a letra: {letra_pasta}...")
        imagens_processadas = 0

        for nome_img in os.listdir(caminho_letra):
            if nome_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                caminho_img = os.path.join(caminho_letra, nome_img)
                imagem = cv2.imread(caminho_img)

                if imagem is None:
                    continue

                imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                results = hands.process(imagem_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        coordenadas = extrair_features(hand_landmarks)

                        predicao = model.predict([coordenadas])[0]

                        y_real.append(letra_pasta)
                        y_predito.append(predicao)
                        imagens_processadas += 1

        print(f"Finalizado {letra_pasta}: {imagens_processadas} imagens analisadas com sucesso.")

if len(y_real) > 0:
    acuracia_real = accuracy_score(y_real, y_predito)
    print("\n" + "-" * 50)
    print(f"ACURÁCIA REAL EM DADOS INÉDITOS: {acuracia_real * 100:.2f}%")
    print("-" * 50)
    print("\nRelatório de Classificação Detalhado:")
    print(classification_report(y_real, y_predito, zero_division=0))
else:
    print("\nNenhuma mão foi detectada nas imagens da pasta de validação externa.")
