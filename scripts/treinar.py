import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_DADOS = os.path.join(BASE_DIR, 'data', 'libras_dados.csv')
PASTA_MODELO = os.path.join(BASE_DIR, 'models')
CAMINHO_MODELO = os.path.join(PASTA_MODELO, 'modelo_libras.p')

if not os.path.exists(PASTA_MODELO):
    os.makedirs(PASTA_MODELO)

try:
    dados = pd.read_csv(ARQUIVO_DADOS)
    print(f"Dados carregados com sucesso de: {ARQUIVO_DADOS}")
except FileNotFoundError:
    print(f"Erro: O arquivo '{ARQUIVO_DADOS}' não foi encontrado. Rode o coleta.py primeiro.")
    exit()

dados.dropna(inplace=True)

X = dados.drop('label', axis=1)
y = dados['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Treinando o modelo (Random Forest) com parâmetros otimizados...")
model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=42) 
model.fit(X_train, y_train)

scores = cross_val_score(model, X, y, cv=5)
print(f"Acurácia média realista (Cross-Validation): {scores.mean() * 100:.2f}%")

y_pred = model.predict(X_test)
acuracia_teste = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste isolado: {acuracia_teste * 100:.2f}%")

with open(CAMINHO_MODELO, 'wb') as f:
    pickle.dump(model, f)

print(f"Modelo salvo em: {CAMINHO_MODELO}")