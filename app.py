from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

app = Flask(__name__)

# Carregando o modelo e o MediaPipe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(BASE_DIR, 'models', 'modelo_libras.p')

with open(CAMINHO_MODELO, 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Variáveis globais para o botão virtual
FRAMES_PARA_CONFIRMAR_LETRA = 30  
FRAMES_PARA_ACIONAR_BOTAO = 25    
BTN_LIMPAR = {'x1': 450, 'y1': 30, 'x2': 590, 'y2': 80, 'cor': (0, 0, 255), 'texto': 'LIMPAR'}

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

def gerar_frames():
    cap = cv2.VideoCapture(0)
    frase_atual = ""
    contador_frames_letra = 0
    contador_frames_botao = 0
    ultima_predicao = ""

    while True:
        success, frame = cap.read()
        if not success:
            break
        
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

                # Lógica do Botão Virtual
                if BTN_LIMPAR['x1'] < x_dedo < BTN_LIMPAR['x2'] and BTN_LIMPAR['y1'] < y_dedo < BTN_LIMPAR['y2']:
                    dedo_no_botao = True
                    contador_frames_botao += 1
                    if contador_frames_botao >= FRAMES_PARA_ACIONAR_BOTAO:
                        frase_atual = ""
                        contador_frames_botao = 0
                else:
                    contador_frames_botao = 0

                # Predição
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
        
        # Barra da frase atual na imagem (opcional na web, mas mantido para segurança)
        cv2.rectangle(frame, (0, h-50), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, f"FRASE: {frase_atual}", (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Codifica o frame para enviar para a web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)