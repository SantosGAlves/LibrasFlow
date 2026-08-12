"""
treinar_final.py
-------------------
Treina o modelo de PRODUCAO (o que app.py e main.py de fato usam),
usando 100% dos dados de data/libras_dados.csv.

Por que um script separado de treinar_sem_vazamento.py:
    treinar_sem_vazamento.py existe para MEDIR a acuracia honesta
    (por isso ele reserva ~20% dos dados, agrupados, como teste). Depois
    que essa medida ja foi feita e reportada, nao ha motivo para o
    modelo que vai pra producao ficar sem ver aquele 20% -- mais dados
    de treino tende a ajudar, e voce ja sabe (pelo outro script) o que
    esperar de acuracia em dados ineditos.

    Resumindo o fluxo do projeto:
        treinar_sem_vazamento.py  -> mede a acuracia real (para o artigo)
        treinar_final.py          -> treina o modelo que vai pro app (usa tudo)
        gerar_matriz_confusao_real.py -> confirma o desempenho em dataset/val

Como usar:
    python treinar_final.py
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_DADOS = os.path.join(BASE_DIR, "data", "libras_dados.csv")
PASTA_MODELO = os.path.join(BASE_DIR, "models")
CAMINHO_MODELO_FINAL = os.path.join(PASTA_MODELO, "modelo_libras_final.p")

os.makedirs(PASTA_MODELO, exist_ok=True)

dados = pd.read_csv(ARQUIVO_DADOS)
dados.dropna(inplace=True)

X = dados.drop("label", axis=1)
y = dados["label"]

print(f"Treinando o modelo final com TODAS as {len(dados)} amostras "
      f"({X.shape[1]} features por amostra)...")

model = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=42
)
model.fit(X, y)

with open(CAMINHO_MODELO_FINAL, "wb") as f:
    pickle.dump(model, f)

print(f"Modelo final salvo em: {CAMINHO_MODELO_FINAL}")
print(
    "\nLembrete: a acuracia esperada deste modelo em dados ineditos NAO e a\n"
    "de um split aleatorio do CSV (isso ainda daria ~100%, veja o motivo em\n"
    "auditar_duplicatas.py). Use os numeros de treinar_sem_vazamento.py e\n"
    "gerar_matriz_confusao_real.py (dataset/val) como referencia real."
)
