"""
treinar_sem_vazamento.py
--------------------------
Corrige o vazamento de dados (data leakage) encontrado em data/libras_dados.csv.

DIAGNOSTICO (confirmado nos dados reais deste projeto):
    O train_test_split aleatorio de treinar.py/gerar_matriz_confusao.py
    espalha "rajadas" de frames quase identicos entre treino e teste.
    Ao medir a distancia entre cada linha e seu vizinho mais proximo DA
    MESMA CLASSE, o dataset de 34.374 linhas colapsa para so ~6.270
    poses realmente distintas (eps=0.05) -- ou seja, cada pose foi
    capturada em media ~5.5 vezes quase idêntica (em letras como B, W,
    U, V isso passa de 20-100x). Testar com split aleatorio por linha
    e, na pratica, testar o modelo com copias do proprio treino.

O QUE ESTE SCRIPT FAZ:
    1. Agrupa linhas quase-identicas da MESMA CLASSE (DBSCAN, min_samples=1)
       usando um raio pequeno (EPS) -- cada grupo = uma "rajada"/pose.
    2. Faz o split treino/teste no nivel de GRUPO (nao de linha), garantindo
       que nenhuma rajada apareca dos dois lados.
    3. Retreina o RandomForest com os MESMOS hiperparametros de treinar.py.
    4. Gera classification_report + matriz de confusao honestos.
    5. (Opcional) roda uma analise de sensibilidade variando EPS, para
       voce mostrar no artigo o quanto a "acuracia perfeita" original
       dependia de vazamento.

Como usar:
    Coloque em scripts/ (mesma pasta de treinar.py) e execute:
        python treinar_sem_vazamento.py            # split padrao (eps=0.05)
        python treinar_sem_vazamento.py --sensibilidade   # roda a tabela de eps
"""

import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQUIVO_DADOS = os.path.join(BASE_DIR, "data", "libras_dados.csv")
PASTA_MODELO = os.path.join(BASE_DIR, "models")
CAMINHO_MODELO_NOVO = os.path.join(PASTA_MODELO, "modelo_libras_sem_vazamento.p")
SAIDA_FIGURA = os.path.join(PASTA_MODELO, "matriz_confusao_sem_vazamento.png")
SAIDA_RELATORIO = os.path.join(PASTA_MODELO, "classification_report_sem_vazamento.txt")

EPS_PADRAO = 0.10  # abaixo do p90 (~0.057) da distancia intra-classe: agrupa so quase-duplicatas
SEED = 42


def agrupar_quase_duplicatas(X, y, eps):
    """Retorna um array de ids de grupo (uma 'rajada' por grupo), por classe."""
    grupo = np.empty(len(X), dtype=object)
    for classe in sorted(y.unique()):
        idx = np.where(y.values == classe)[0]
        Xc = X.values[idx]
        db = DBSCAN(eps=eps, min_samples=1, n_jobs=-1).fit(Xc)
        for i, lab in zip(idx, db.labels_):
            grupo[i] = f"{classe}_{lab}"
    return grupo


def split_por_grupo(dados, coluna_grupo="grupo", coluna_label="label", test_size=0.2, seed=SEED):
    """80/20 no nivel de GRUPO (nunca de linha), estratificado por classe."""
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for _, sub in dados.groupby(coluna_label):
        grupos = sub[coluna_grupo].unique().tolist()
        rng.shuffle(grupos)
        n_test = max(1, int(round(test_size * len(grupos))))
        grupos_test = set(grupos[:n_test])
        mask_test = sub[coluna_grupo].isin(grupos_test)
        test_idx.extend(sub[mask_test].index.tolist())
        train_idx.extend(sub[~mask_test].index.tolist())
    return np.array(train_idx), np.array(test_idx)


def treinar_e_avaliar(dados_com_grupo, salvar=False):
    X = dados_com_grupo.drop(columns=["label", "grupo"])
    y = dados_com_grupo["label"]
    train_idx, test_idx = split_por_grupo(dados_com_grupo)

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    grupos_treino = set(dados_com_grupo.loc[train_idx, "grupo"])
    grupos_teste = set(dados_com_grupo.loc[test_idx, "grupo"])
    assert not (grupos_treino & grupos_teste), "Vazamento: ha grupos nos dois lados!"

    model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=2, n_jobs=-1, random_state=SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    if salvar:
        relatorio = classification_report(y_test, y_pred, zero_division=0)
        print(f"\nAcuracia (split por grupo, sem vazamento): {acc * 100:.2f}%\n")
        print(relatorio)

        os.makedirs(PASTA_MODELO, exist_ok=True)
        import pickle

        with open(CAMINHO_MODELO_NOVO, "wb") as f:
            pickle.dump(model, f)
        with open(SAIDA_RELATORIO, "w", encoding="utf-8") as f:
            f.write(f"Acuracia geral (split por grupo, eps={EPS_PADRAO}): {acc * 100:.2f}%\n\n")
            f.write(relatorio)

        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(10, 9))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
        ax.set_title(f"Matriz de Confusao - Split por Grupo (sem vazamento) - Acc {acc * 100:.1f}%")
        plt.tight_layout()
        plt.savefig(SAIDA_FIGURA, dpi=200)

        print(f"\nModelo salvo em: {CAMINHO_MODELO_NOVO}")
        print(f"Relatorio salvo em: {SAIDA_RELATORIO}")
        print(f"Figura salva em: {SAIDA_FIGURA}")

    return acc


def main():
    dados = pd.read_csv(ARQUIVO_DADOS)
    dados.dropna(inplace=True)
    X = dados.drop(columns=["label"])
    y = dados["label"]

    if "--sensibilidade" in sys.argv:
        print("Rodando analise de sensibilidade (varios valores de EPS)...\n")
        print(f"{'eps':>6} | {'grupos':>7} | {'acuracia':>9}")
        for eps in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
            dados["grupo"] = agrupar_quase_duplicatas(X, y, eps)
            n_grupos = dados["grupo"].nunique()
            acc = treinar_e_avaliar(dados, salvar=False)
            print(f"{eps:6.2f} | {n_grupos:7d} | {acc * 100:8.2f}%")
        print(
            "\nQuanto maior o eps, mais 'poses distintas' sao forcadas a ficar\n"
            "so de um lado do split -- e mais isso se aproxima de uma prova\n"
            "real de generalizacao (em vez de so remover copias literais).\n"
            "Use isso como evidencia no artigo do quanto a acuracia original\n"
            "dependia de vazamento."
        )
    else:
        print(f"Agrupando quase-duplicatas com eps={EPS_PADRAO} (raio no espaco de landmarks normalizados)...")
        dados["grupo"] = agrupar_quase_duplicatas(X, y, EPS_PADRAO)
        print(f"Linhas: {len(dados)} -> Grupos (poses distintas): {dados['grupo'].nunique()}")
        treinar_e_avaliar(dados, salvar=True)


if __name__ == "__main__":
    main()
