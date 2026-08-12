import os
import sys
import cv2
import mediapipe as mp
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libras_features import extrair_features, nomes_colunas, USAR_Z

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_DATASET = os.path.join(BASE_DIR, 'dataset', 'train')
PASTA_SAIDA = os.path.join(BASE_DIR, 'data')
ARQUIVO_SAIDA = os.path.join(PASTA_SAIDA, 'libras_dados.csv')

if not os.path.exists(PASTA_SAIDA):
    os.makedirs(PASTA_SAIDA)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

with open(ARQUIVO_SAIDA, mode='w', newline='') as f:
    writer = csv.writer(f)
    colunas = ['label'] + nomes_colunas()
    writer.writerow(colunas)

print(f"Iniciando processamento do dataset em: {CAMINHO_DATASET}")
print(f"Usando Z (profundidade)? {USAR_Z}")

total_imagens = 0
sucessos = 0

for letra_pasta in os.listdir(CAMINHO_DATASET):
    caminho_letra = os.path.join(CAMINHO_DATASET, letra_pasta)

    if os.path.isdir(caminho_letra):
        print(f"Processando letra: {letra_pasta}...")

        for nome_img in os.listdir(caminho_letra):
            if nome_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_imagens += 1
                caminho_img = os.path.join(caminho_letra, nome_img)
                imagem = cv2.imread(caminho_img)

                if imagem is None:
                    continue

                imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                results = hands.process(imagem_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        coordenadas = extrair_features(hand_landmarks)

                        with open(ARQUIVO_SAIDA, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([letra_pasta] + coordenadas)

                        sucessos += 1

print("-" * 30)
print("PROCESSAMENTO CONCLUÍDO!")
print(f"Total de imagens lidas: {total_imagens}")
print(f"Mãos detectadas e salvas: {sucessos}")
print(f"Dados salvos em: {ARQUIVO_SAIDA}")
