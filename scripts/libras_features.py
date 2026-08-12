"""
libras_features.py
---------------------
Modulo unico de extracao de features a partir dos landmarks do MediaPipe
Hands. TODOS os scripts do projeto (coleta, app, main, benchmark,
validacao, diagnostico) devem importar esta funcao, em vez de duplicar a
logica em cada arquivo -- isso evita o bug classico deste projeto:
esquecer de atualizar a normalizacao em um dos arquivos e a predicao
ficar fora da distribuicao em que o modelo foi treinado.

Historico:
    - Versao original usava so (x, y) relativos ao landmark 0 (pulso),
      normalizados pelo maior valor absoluto.
    - USAR_Z=True adiciona a coordenada Z (profundidade relativa,
      fornecida pelo proprio MediaPipe). Isso ajuda em sinais onde dedos
      se cruzam ou se dobram (ex.: R, S, U, e a ambiguidade real A/F/U
      encontrada na analise por grupos), pois a diferenca entre eles
      muitas vezes esta em qual dedo esta "na frente" ou "atras", que
      x,y sozinhos nao capturam.

IMPORTANTE: se voce mudar USAR_Z aqui, precisa:
    1. Rodar coleta.py de novo (para gerar data/libras_dados.csv com o
       novo numero de colunas).
    2. Retreinar o modelo (treinar_sem_vazamento.py / treinar_final.py).
    Um modelo treinado com um formato de features nao pode receber
    entradas de outro formato -- vai dar erro ou (pior) predicao errada
    silenciosa.
"""

NUM_LANDMARKS = 21
USAR_Z = True  # mude aqui uma unica vez; todos os scripts respeitam este valor


def extrair_features(hand_landmarks, usar_z=USAR_Z):
    """
    Recebe hand_landmarks do MediaPipe (resultado de hands.process(...))
    e retorna uma lista de floats: coordenadas centralizadas no landmark
    0 (pulso) e normalizadas pelo maior valor absoluto -- isso torna o
    vetor invariante a posicao e a escala da mao na imagem.

    Retorna 42 valores (x,y) se usar_z=False, ou 63 valores (x,y,z) se
    usar_z=True.
    """
    id0_x = hand_landmarks.landmark[0].x
    id0_y = hand_landmarks.landmark[0].y
    id0_z = hand_landmarks.landmark[0].z

    coordenadas = []
    for lm in hand_landmarks.landmark:
        coordenadas.append(lm.x - id0_x)
        coordenadas.append(lm.y - id0_y)
        if usar_z:
            coordenadas.append(lm.z - id0_z)

    max_val = max(map(abs, coordenadas))
    if max_val != 0:
        coordenadas = [c / max_val for c in coordenadas]

    return coordenadas


def nomes_colunas(usar_z=USAR_Z):
    """Gera os nomes das colunas do CSV, na MESMA ordem de extrair_features."""
    colunas = []
    for i in range(NUM_LANDMARKS):
        colunas.append(f"x{i}")
        colunas.append(f"y{i}")
        if usar_z:
            colunas.append(f"z{i}")
    return colunas
