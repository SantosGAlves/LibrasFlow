import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_MODELO = os.path.join(BASE_DIR, 'models', 'modelo_libras.p')

FRAMES_PARA_CONFIRMAR_LETRA = 30  
FRAMES_PARA_ACIONAR_BOTAO = 25    
BTN_LIMPAR = {'x1': 500, 'y1': 50, 'x2': 620, 'y2': 100, 'cor': (0, 0, 255), 'texto': 'LIMPAR'}

try:
    with open(CAMINHO_MODELO, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Erro: Modelo não encontrado em {CAMINHO_MODELO}")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

frase_atual = ""
contador_frames_letra = 0
contador_frames_botao = 0
ultima_predicao = ""

def desenhar_botao(img, btn, contador):
    cor = btn['cor']
    if contador > 0:
        cor = (100, 100, 255) 
    
    cv2.rectangle(img, (btn['x1'], btn['y1']), (btn['x2'], btn['y2']), cor, -1)
    cv2.putText(img, btn['texto'], (btn['x1']+10, btn['y1']+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if contador > 0:
        largura_total = btn['x2'] - btn['x1']
        progresso = int((contador / FRAMES_PARA_ACIONAR_BOTAO) * largura_total)
        cv2.rectangle(img, (btn['x1'], btn['y2'] + 5), (btn['x1'] + progresso, btn['y2'] + 10), (0, 255, 0), -1)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    dedo_no_botao = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_dedo = int(hand_landmarks.landmark[8].x * w)
            y_dedo = int(hand_landmarks.landmark[8].y * h)

            if BTN_LIMPAR['x1'] < x_dedo < BTN_LIMPAR['x2'] and BTN_LIMPAR['y1'] < y_dedo < BTN_LIMPAR['y2']:
                dedo_no_botao = True
                contador_frames_botao += 1
                if contador_frames_botao >= FRAMES_PARA_ACIONAR_BOTAO:
                    frase_atual = ""
                    contador_frames_botao = 0
            else:
                contador_frames_botao = 0

            if not dedo_no_botao:
                coordenadas = []
                id0_x = hand_landmarks.landmark[0].x
                id0_y = hand_landmarks.landmark[0].y

                for lm in hand_landmarks.landmark:
                    coordenadas.extend([lm.x - id0_x, lm.y - id0_y])
                
                predicao = model.predict([coordenadas])[0]
                cv2.putText(frame, f"Letra: {predicao}", (x_dedo+20, y_dedo), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if predicao == ultima_predicao:
                    contador_frames_letra += 1
                    
                    largura_barra = int((contador_frames_letra / FRAMES_PARA_CONFIRMAR_LETRA) * 100)
                    cv2.rectangle(frame, (x_dedo + 20, y_dedo + 10), (x_dedo + 120, y_dedo + 20), (255, 255, 255), 1)
                    cv2.rectangle(frame, (x_dedo + 20, y_dedo + 10), (x_dedo + 20 + largura_barra, y_dedo + 20), (0, 255, 0), -1)

                    if contador_frames_letra == FRAMES_PARA_CONFIRMAR_LETRA:
                        frase_atual += predicao
                        contador_frames_letra = 0
                else:
                    contador_frames_letra = 0
                    ultima_predicao = predicao
    
    desenhar_botao(frame, BTN_LIMPAR, contador_frames_botao)
    
    cv2.rectangle(frame, (0, 0), (w, 50), (50, 50, 50), -1)
    cv2.putText(frame, f"FRASE: {frase_atual}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Libras Vision", frame)
    key = cv2.waitKey(1)
    if key == 27: break 
    elif key == 32: frase_atual += " " 
    elif key == 8: frase_atual = frase_atual[:-1] 

cap.release()
cv2.destroyAllWindows()