import cv2
import mediapipe as mp
import pickle
import numpy as np

# --- CONFIGURAÇÕES ---
FRAMES_PARA_CONFIRMAR_LETRA = 30  # Tempo para escrever a letra
FRAMES_PARA_ACIONAR_BOTAO = 25    # Tempo segurando o botão para ele clicar

# Coordenadas do botão LIMPAR (x1, y1, x2, y2)
BTN_LIMPAR = {'x1': 500, 'y1': 50, 'x2': 620, 'y2': 100, 'cor': (0, 0, 255), 'texto': 'LIMPAR'}

# Carregar modelo
with open('modelo_libras.p', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Variáveis de estado
frase_atual = ""
contador_frames_letra = 0
contador_frames_botao = 0
ultima_predicao = ""

def desenhar_botao(img, btn, contador):
    """Desenha o botão e a barra de carregamento dele"""
    cor = btn['cor']
    # Se estiver "carregando" o clique, clareia a cor
    if contador > 0:
        cor = (100, 100, 255) # Vermelho claro
        
    cv2.rectangle(img, (btn['x1'], btn['y1']), (btn['x2'], btn['y2']), cor, -1)
    cv2.putText(img, btn['texto'], (btn['x1']+10, btn['y1']+35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Barra de progresso do botão (efeito visual)
    if contador > 0:
        largura_barra = int(((btn['x2'] - btn['x1']) * contador) / FRAMES_PARA_ACIONAR_BOTAO)
        cv2.rectangle(img, (btn['x1'], btn['y2']), (btn['x1'] + largura_barra, btn['y2']+5), (0, 255, 0), -1)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) # Espelha
    altura, largura, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Área de visualização da frase
    cv2.rectangle(frame, (0, 0), (largura, 40), (50, 50, 50), -1)
    cv2.putText(frame, f"Frase: {frase_atual}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    dedo_no_botao = False # Flag para saber se estamos interagindo com botão

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- 1. LÓGICA DO BOTÃO VIRTUAL ---
            # Pega a ponta do indicador (Landmark 8)
            ponta_indicador = hand_landmarks.landmark[8]
            x_dedo = int(ponta_indicador.x * largura)
            y_dedo = int(ponta_indicador.y * altura)
            
            # Desenha uma bolinha na ponta do indicador para ajudar a mirar
            cv2.circle(frame, (x_dedo, y_dedo), 10, (255, 0, 255), -1)

            # Verifica se o dedo está dentro do quadrado do botão LIMPAR
            if (BTN_LIMPAR['x1'] < x_dedo < BTN_LIMPAR['x2']) and (BTN_LIMPAR['y1'] < y_dedo < BTN_LIMPAR['y2']):
                dedo_no_botao = True
                contador_frames_botao += 1
                if contador_frames_botao >= FRAMES_PARA_ACIONAR_BOTAO:
                    frase_atual = "" # Ação: Limpar
                    contador_frames_botao = 0 # Reseta
                    # Feedback visual de clique (tela pisca branco rápido)
                    cv2.rectangle(frame, (0,0), (largura, altura), (255, 255, 255), -1)
            else:
                if not dedo_no_botao:
                    contador_frames_botao = 0

            # --- 2. LÓGICA DE PREDIÇÃO DA LETRA (Só roda se NÃO estiver apertando botão) ---
            if not dedo_no_botao:
                coordenadas = []
                for lm in hand_landmarks.landmark:
                    coordenadas.extend([lm.x, lm.y])
                
                predicao = model.predict([coordenadas])[0]
                
                # Texto flutuante perto da mão
                cv2.putText(frame, predicao, (x_dedo+20, y_dedo), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if predicao == ultima_predicao:
                    contador_frames_letra += 1
                    # Barra de progresso da letra
                    if contador_frames_letra > 5: # Só mostra se segurar um pouco
                        cv2.rectangle(frame, (x_dedo+20, y_dedo+10), (x_dedo+20 + contador_frames_letra, y_dedo+20), (0,255,0), -1)
                    
                    if contador_frames_letra == FRAMES_PARA_CONFIRMAR_LETRA:
                        frase_atual += predicao
                        contador_frames_letra = 0
                        # Feedback visual pequeno na área de texto
                        cv2.rectangle(frame, (0, 0), (largura, 40), (0, 100, 0), -1)
                else:
                    contador_frames_letra = 0
                    ultima_predicao = predicao
    
    # Desenha o botão na tela
    desenhar_botao(frame, BTN_LIMPAR, contador_frames_botao)

    cv2.imshow('Tradutor LIBRAS com Botao Virtual', frame)
    
    k = cv2.waitKey(1)
    if k == 27: break # ESC para sair
    if k == 32: frase_atual += " " # ESPAÇO via teclado ainda funciona
    if k == 8: frase_atual = frase_atual[:-1] # BACKSPACE via teclado ainda funciona

cap.release()
cv2.destroyAllWindows()