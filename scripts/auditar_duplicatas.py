"""
auditar_duplicatas.py
-----------------------
Verifica se data/libras_dados.csv tem amostras QUASE IDENTICAS espalhadas
entre treino e teste -- o que explica uma acuracia "perfeita" (precision/
recall/f1 = 1.00 em quase todas as letras) e torna a matriz de confusao
inutil para analise.

O que o script faz:
    1. Recarrega os dados e refaz O MESMO split de treinar.py
       (test_size=0.2, random_state=42, stratify=y).
    2. Para cada amostra de teste, procura o vizinho mais proximo dentro
       do conjunto de treino DA MESMA CLASSE (distancia euclidiana nas
       42 coordenadas normalizadas).
    3. Reporta quantas amostras de teste tem um "quase-clone" no treino
       (distancia abaixo de um limiar pequeno) e quantas sao duplicatas
       EXATAS (distancia = 0).

Como usar:
    Coloque este arquivo em scripts/ (mesma pasta de treinar.py) e rode:
        python auditar_duplicatas.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_DADOS = os.path.join(BASE_DIR, "data", "libras_dados.csv")

LIMIAR_QUASE_CLONE = 0.02  # distancia euclidiana; diminua/aumente para testar sensibilidade

dados = pd.read_csv(ARQUIVO_DADOS)
dados.dropna(inplace=True)

X = dados.drop("label", axis=1)
y = dados["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras\n")

exatas = 0
quase_clones = 0
distancias_min = []

for classe in sorted(y.unique()):
    treino_classe = X_train[y_train == classe].to_numpy()
    teste_classe = X_test[y_test == classe].to_numpy()
    if len(treino_classe) == 0 or len(teste_classe) == 0:
        continue

    nn = NearestNeighbors(n_neighbors=1).fit(treino_classe)
    dist, _ = nn.kneighbors(teste_classe)
    dist = dist.ravel()
    distancias_min.extend(dist.tolist())

    exatas += int((dist == 0).sum())
    quase_clones += int((dist < LIMIAR_QUASE_CLONE).sum())

distancias_min = np.array(distancias_min)

print("=" * 60)
print("RESULTADO DA AUDITORIA DE DUPLICATAS / QUASE-DUPLICATAS")
print("=" * 60)
print(
    f"Duplicatas EXATAS teste->treino (distancia = 0): {exatas} "
    f"({100 * exatas / len(X_test):.1f}% do teste)"
)
print(
    f"Quase-clones (distancia < {LIMIAR_QUASE_CLONE}): {quase_clones} "
    f"({100 * quase_clones / len(X_test):.1f}% do teste)"
)
print(f"Distancia minima media (teste -> vizinho mais proximo no treino): {distancias_min.mean():.4f}")
print(f"Distancia minima mediana: {np.median(distancias_min):.4f}")
print(f"Percentual de amostras de teste com distancia < 0.10: {100 * (distancias_min < 0.10).mean():.1f}%")

if len(X_test) and quase_clones / len(X_test) > 0.05:
    print(
        "\n>>> DIAGNOSTICO: uma fracao relevante do conjunto de teste tem um\n"
        ">>> 'clone' quase identico no treino. Isso confirma vazamento de\n"
        ">>> dados (data leakage): o modelo nao esta generalizando, esta\n"
        ">>> reconhecendo quadros repetidos da mesma sessao de captura.\n"
        ">>> É por isso que o classification_report sai com 1.00 em quase\n"
        ">>> tudo -- o numero e real, mas nao mede o que voce quer medir."
    )
else:
    print(
        "\n>>> Poucas quase-duplicatas neste limiar. Se a acuracia continua\n"
        ">>> ~100%, vale investigar se o problema (letras estaticas de\n"
        ">>> LIBRAS, so 21 landmarks normalizados) e simplesmente muito\n"
        ">>> facil, ou testar um limiar maior (ex.: 0.05, 0.10)."
    )
