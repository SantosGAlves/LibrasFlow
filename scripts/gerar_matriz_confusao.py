"""
gerar_matriz_confusao.py
--------------------------
Gera a matriz de confusao (figura) e o relatorio de classificacao completo
(precision, recall, f1-score por letra) do modelo ja treinado, usando
EXATAMENTE o mesmo conjunto de teste utilizado em treinar.py (mesmo
random_state=42 e mesma estratificacao), para garantir que os numeros
sejam consistentes com a acuracia reportada no artigo (93,88%).

Como usar:
    1. Coloque este arquivo na mesma pasta de treinar.py.
    2. Garanta que data/libras_dados.csv e models/modelo_libras.p
       ja existam (rode coleta.py e treinar.py antes, se necessario).
    3. Execute: python gerar_matriz_confusao.py
    4. Substitua a imagem placeholder da Figura 3 do artigo pelo arquivo
       gerado em models/matriz_confusao.png.
    5. (Opcional) Cole o conteudo de models/classification_report.txt
       em uma tabela suplementar, se quiser detalhar o desempenho por letra.
"""

import os

import matplotlib

matplotlib.use("Agg")  # nao precisa de tela para salvar a figura
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Caminhos (mesma convencao usada em treinar.py / coleta.py)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_DADOS = os.path.join(BASE_DIR, "data", "libras_dados.csv")
CAMINHO_MODELO = os.path.join(BASE_DIR, "models", "modelo_libras.p")
SAIDA_FIGURA = os.path.join(BASE_DIR, "models", "matriz_confusao.png")
SAIDA_RELATORIO = os.path.join(BASE_DIR, "models", "classification_report.txt")

# ----------------------------------------------------------------------
# Carrega o modelo ja treinado
# ----------------------------------------------------------------------
with open(CAMINHO_MODELO, "rb") as f:
    model = pickle.load(f)

# ----------------------------------------------------------------------
# Recarrega os dados e refaz O MESMO split usado em treinar.py
# (test_size=0.2, random_state=42, stratify=y) para obter o mesmo
# conjunto de teste que gerou a acuracia de 93,88% reportada no artigo
# ----------------------------------------------------------------------
dados = pd.read_csv(ARQUIVO_DADOS)
dados.dropna(inplace=True)

X = dados.drop("label", axis=1)
y = dados["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)

# ----------------------------------------------------------------------
# Relatorio de classificacao completo (precision / recall / f1 por letra)
# ----------------------------------------------------------------------
relatorio = classification_report(y_test, y_pred)
print(relatorio)

os.makedirs(os.path.dirname(SAIDA_RELATORIO), exist_ok=True)
with open(SAIDA_RELATORIO, "w", encoding="utf-8") as f:
    f.write(relatorio)
print(f"Relatorio de classificacao salvo em: {SAIDA_RELATORIO}")

# ----------------------------------------------------------------------
# Matriz de confusao
# ----------------------------------------------------------------------
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(10, 9))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
ax.set_title("Matriz de Confusao - Classificador Random Forest (LIBRAS)")
plt.tight_layout()
plt.savefig(SAIDA_FIGURA, dpi=200)
print(f"Matriz de confusao salva em: {SAIDA_FIGURA}")
print("\nSubstitua a imagem placeholder da Figura 3 do artigo por este arquivo.")
