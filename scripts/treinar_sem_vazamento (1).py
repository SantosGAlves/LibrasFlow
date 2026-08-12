"""
treinar_sem_vazamento.py
--------------------------
Corrige o vazamento de dados (data leakage) encontrado em data/libras_dados.csv.

DIAGNOSTICO (confirmado nos dados reais deste projeto):
    O train_test_split aleatorio de treinar.py/gerar_matriz_confusao.py
    espalha "rajadas" de frames quase identicos entre treino e teste.
    Ao medir a distancia entre cada linha e seu vizinho mais proximo DA
    MESMA CLASSE, o dataset de 34.374 linhas colapsa para so ~6.270
    poses realmente distintas (eps=0.05).

CORRECAO 2 (v2 -- fev/2026): a v1 deste script escolhia grupos de teste
    ate atingir 20% do NUMERO DE GRUPOS de cada classe. Isso quebra
    quando os grupos tem tamanhos muito desiguais: em eps=0.10, por
    exemplo, a letra G tinha so 4 grupos, um deles com 900 das 1650
    amostras (54%) -- se esse grupo caia no teste por sorte, a classe
    toda ficava com support=900 e uma proporcao de teste de 54% em vez
    de 20%, produzindo numeros bizarros (precision/recall 0.00 em
    classes que na verdade sao faceis). Esta versao escolhe os grupos
    de teste tentando bater ~20% da QUANTIDADE DE AMOSTRAS de cada
    classe (nao do numero de grupos), testando varias combinacoes
    aleatorias e ficando com a mais proxima do alvo. Tambem avisa
    quando uma classe tem poucos grupos (< 10), caso em que qualquer
    split vai ser um pouco instavel por causa da baixa granularidade
    disponivel -- isso e uma limitacao dos DADOS, nao do metodo.

O QUE ESTE SCRIPT FAZ:
    1. Agrupa linhas quase-identicas da MESMA CLASSE (DBSCAN, min_samples=1).
    2. Faz o split treino/teste no nivel de GRUPO, balanceado por amostras.
    3. Retreina o RandomForest com os MESMOS hiperparametros de treinar.py.
    4. Gera classification_report + matriz de confusao honestos.
    5. (Opcional) roda uma analise de sensibilidade variando EPS.

Como usar:
    python treinar_sem_vazamento.py                    # split padrao (eps=0.05)
    python treinar_sem_vazamento.py --sensibilidade     # tabela de eps

NOTA SOBRE QUAL EPS USAR COMO NUMERO OFICIAL:
    eps=0.05 e o valor recomendado para o numero que voce reporta como
    "acuracia sem vazamento" -- e o unico com grupos suficientes em
    todas as classes para um split estatisticamente estavel (veja os
    avisos impressos). Valores de eps maiores (0.10+) sao uteis como
    ANALISE DE SENSIBILIDADE (--sensibilidade) para mostrar o quanto a
    acuracia cai conforme se exige mais generalizacao -- mas nao devem
    ser tratados como "o" numero final, porque ficam instaveis quando
    uma classe tem poucos grupos.
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

EPS_PADRAO = 0.05  # ver nota no docstring -- nao aumente isso sem ler o aviso de grupos
MIN_GRUPOS_ESTAVEL = 10
SEED = 42


def agrupar_quase_duplicatas(X, y, eps):
    """Retorna um array de ids de grupo (uma 'rajada'/pose por grupo), por classe."""
    grupo = np.empty(len(X), dtype=object)
    for classe in sorted(y.unique()):
        idx = np.where(y.values == classe)[0]
        Xc = X.values[idx]
        db = DBSCAN(eps=eps, min_samples=1, n_jobs=-1).fit(Xc)
        for i, lab in zip(idx, db.labels_):
            grupo[i] = f"{classe}_{lab}"
    return grupo


def split_por_grupo(dados, coluna_grupo="grupo", coluna_label="label", test_size=0.2,
                     seed=SEED, tentativas=50, avisar=True):
    """
    80/20 no nivel de GRUPO (nunca de linha), tentando bater ~test_size
    da QUANTIDADE DE AMOSTRAS de cada classe (nao do numero de grupos).
    Testa `tentativas` combinacoes aleatorias e fica com a mais proxima
    do alvo -- uma heuristica simples para o problema de "subset sum".
    """
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    avisos = []

    for classe, sub in dados.groupby(coluna_label):
        tamanhos = sub.groupby(coluna_grupo).size()
        grupos = tamanhos.index.tolist()
        n_total = len(sub)
        alvo = test_size * n_total

        if len(grupos) < MIN_GRUPOS_ESTAVEL:
            avisos.append(f"{classe} (so {len(grupos)} grupos distintos)")

        melhor_sel, melhor_dist, melhor_acum = None, None, None
        for _ in range(tentativas):
            ordem = grupos.copy()
            rng.shuffle(ordem)
            acumulado, sel = 0, []
            for g in ordem:
                if acumulado >= alvo:
                    break
                sel.append(g)
                acumulado += tamanhos[g]
            dist = abs(acumulado - alvo)
            if melhor_dist is None or dist < melhor_dist:
                melhor_dist, melhor_sel, melhor_acum = dist, sel, acumulado

        mask_test = sub[coluna_grupo].isin(melhor_sel)
        test_idx.extend(sub[mask_test].index.tolist())
        train_idx.extend(sub[~mask_test].index.tolist())

    if avisar and avisos:
        print("AVISO: classes com poucos grupos distintos (split pode nao bater exatamente "
              f"{int(test_size * 100)}% -- limitacao dos dados, nao do metodo):")
        for a in avisos:
            print(f"  - {a}")

    return np.array(train_idx), np.array(test_idx)


def treinar_e_avaliar(dados_com_grupo, salvar=False, avisar=True):
    X = dados_com_grupo.drop(columns=["label", "grupo"])
    y = dados_com_grupo["label"]
    train_idx, test_idx = split_por_grupo(dados_com_grupo, avisar=avisar)

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
        print("Rodando analise de sensibilidade (varios valores de EPS)...")
        print("(numeros com eps >= 0.10 sao so para referencia -- veja o docstring)\n")
        print(f"{'eps':>6} | {'grupos':>7} | {'acuracia':>9}")
        for eps in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
            dados["grupo"] = agrupar_quase_duplicatas(X, y, eps)
            n_grupos = dados["grupo"].nunique()
            acc = treinar_e_avaliar(dados, salvar=False, avisar=False)
            print(f"{eps:6.2f} | {n_grupos:7d} | {acc * 100:8.2f}%")
        print(
            "\nQuanto maior o eps, mais 'poses distintas' sao forcadas a ficar\n"
            "so de um lado do split -- mais proximo de generalizacao real.\n"
            "Use isso como evidencia no artigo do quanto a acuracia original\n"
            "dependia de vazamento, mas reporte o numero de eps=0.05 como o\n"
            "principal (e o mais estatisticamente estavel)."
        )
    else:
        print(f"Agrupando quase-duplicatas com eps={EPS_PADRAO} (raio no espaco de landmarks normalizados)...")
        dados["grupo"] = agrupar_quase_duplicatas(X, y, EPS_PADRAO)
        print(f"Linhas: {len(dados)} -> Grupos (poses distintas): {dados['grupo'].nunique()}")
        treinar_e_avaliar(dados, salvar=True)


if __name__ == "__main__":
    main()
