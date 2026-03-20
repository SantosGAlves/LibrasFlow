import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

print("Treinando o modelo (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1) 
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")


with open(CAMINHO_MODELO, 'wb') as f:
    pickle.dump(model, f)

print(f"Modelo salvo em: {CAMINHO_MODELO}")