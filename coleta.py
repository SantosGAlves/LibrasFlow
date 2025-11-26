import os
import cv2
import mediapipe as mp
import csv

# --- CONFIGURAÇÃO ---
# Coloque aqui o caminho da pasta onde estão as subpastas (A, B, C...)

CAMINHO_DATASET = r'C:\Projeto IA\train'

ARQUIVO_SAIDA = 'libras_dados.csv'

# --- INICIALIZAÇÃO MEDIAPIPE ---
mp_hands = mp.solutions.hands
# Importante: static_image_mode=True para fotos (é mais preciso/pesado que vídeo)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# --- PREPARAR CSV ---
# Vamos criar o cabeçalho (Label + 21 pontos x,y)
with open(ARQUIVO_SAIDA, mode='w', newline='') as f:
    writer = csv.writer(f)
    colunas = ['label']
    for i in range(21):
        colunas.extend([f'x{i}', f'y{i}'])
    writer.writerow(colunas)

print(f"Iniciando processamento do dataset em: {CAMINHO_DATASET}")

# --- PROCESSAMENTO ---
total_imagens = 0
sucessos = 0

# Varre todas as pastas (A, B, C...)
for letra_pasta in os.listdir(CAMINHO_DATASET):
    caminho_letra = os.path.join(CAMINHO_DATASET, letra_pasta)
    
    # Verifica se é uma pasta mesmo
    if os.path.isdir(caminho_letra):
        print(f"Processando letra: {letra_pasta}...")
        
        # Varre todas as imagens dentro da pasta da letra
        for nome_img in os.listdir(caminho_letra):
            # Filtra apenas imagens (ignora arquivos ocultos)
            if nome_img.lower().endswith(('.png', '.jpg', '.jpeg')):
                total_imagens += 1
                
                caminho_img = os.path.join(caminho_letra, nome_img)
                imagem = cv2.imread(caminho_img)
                
                if imagem is None:
                    continue

                # Converte BGR para RGB (MediaPipe exige RGB)
                imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                
                # Processa a imagem
                results = hands.process(imagem_rgb)
                
                # Se encontrou uma mão na foto
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        coordenadas = []
                        for lm in hand_landmarks.landmark:
                            coordenadas.extend([lm.x, lm.y])
                        
                        # Salva no CSV: A letra (nome da pasta) + as coordenadas
                        with open(ARQUIVO_SAIDA, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([letra_pasta] + coordenadas)
                        
                        sucessos += 1

print("-" * 30)
print("PROCESSAMENTO CONCLUÍDO!")
print(f"Total de imagens lidas: {total_imagens}")
print(f"Mãos detectadas e salvas: {sucessos}")
print(f"Dados salvos em: {ARQUIVO_SAIDA}")