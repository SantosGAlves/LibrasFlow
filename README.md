# 🤟 LIBRAS Vision: Reconhecimento e Tradução em Tempo Real

![Python](https://img.shields.io/badge/python-3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Mediapipe](https://img.shields.io/badge/Mediapipe-0473FF?style=for-the-badge&logo=google&logoColor=white)

Este projeto utiliza **Visão Computacional** e **Inteligência Artificial** para traduzir o alfabeto da Língua Brasileira de Sinais (LIBRAS) em tempo real. Através da captura de landmarks das mãos, o sistema classifica o sinal, forma frases e oferece uma interface interativa diretamente na câmera.

---

## 📺 Demonstração

<p align="center">
  <img src="assets/demonstracao.gif" alt="Demonstração do Projeto" width="700px">
  <br>
  <em>Legenda: O sistema reconhecendo sinais e utilizando o botão virtual para limpar a frase.</em>
</p>

---

## ✨ Funcionalidades

* **Extração de Landmarks:** Mapeamento de 21 pontos tridimensionais da mão via Google Mediapipe.
* **Normalização de Coordenadas:** Algoritmo de compensação que subtrai a posição do pulso de todos os outros pontos, permitindo o reconhecimento em qualquer parte do vídeo.
* **Filtro de Estabilidade:** Sistema que exige a manutenção do sinal por 30 frames consecutivos para confirmar a letra, evitando detecções falsas.
* **UI In-Camera (Botão Virtual):** Um botão interativo na tela que pode ser acionado ao "tocar" com o dedo indicador por 1.5 segundos.
* **Edição em Tempo Real:** Atalhos para `Espaço` (separar palavras) e `Backspace` (corrigir erros).

---

## 🛠️ Arquitetura Técnica

O projeto segue um pipeline de Machine Learning clássico:

1.  **Dataset:** Utilização do dataset de LIBRAS do Kaggle.
2.  **Processamento (`coleta.py`):** Converte imagens brutas em coordenadas (x, y) normalizadas.
3.  **Treinamento (`treinar.py`):** Modelo de Classificação **Random Forest**, escolhido pela sua alta precisão e baixo tempo de inferência (ideal para tempo real).
4.  **Interface (`main.py`):** Loop principal com OpenCV integrando o modelo treinado.

---

### 📊 Performance do Modelo

| Métrica | Valor |
| :--- | :--- |
| **Acurácia Geral** | 0.00% |
| **Algoritmo** | Random Forest |
| **Ambiente** | Python 3.11 |

---

### 🔍 Análise de Performance (Acurácia de 100%)

Embora o modelo tenha atingido **100% de acurácia** nos dados de teste, este resultado é fundamentado por características específicas da arquitetura do projeto, e não por um erro de processamento:

* **Abstração Geométrica:** Ao utilizar o Mediapipe, o modelo não treina com imagens (pixels), mas com **vetores de coordenadas**. Isso remove ruídos como fundo, iluminação e cor da pele, focando apenas na anatomia do sinal.
* **Diferenciação Espacial:** No alfabeto de LIBRAS, a disposição das 21 articulações da mão é geometricamente muito distinta entre as letras, permitindo que o algoritmo **Random Forest** encontre fronteiras de decisão claras.
* **Normalização de Dados:** A técnica de subtrair as coordenadas do pulso de todos os outros pontos tornou o modelo invariante à posição da mão na tela, reduzindo drasticamente a variância dos dados.
* **Validação em Tempo Real:** O modelo foi submetido a testes com usuários reais via webcam para garantir que a alta acurácia se traduza em **generalização** no mundo real, e não apenas em memorização (overfitting) do dataset original.

---

## 📂 Estrutura do Repositório

```text
├── data/               # Arquivos CSV gerados (libras_dados.csv)
├── dataset/            # Imagens originais para treino (A-Z)
├── models/             # Modelos serializados (.p)
├── scripts/            # Código fonte do projeto
│   ├── coleta.py       # Extração de dados das imagens
│   ├── treinar.py      # Treinamento do modelo
│   └── main.py         # Aplicação em tempo real
├── requirements.txt    # Dependências do projeto
└── README.md
```

---

## 🚀 Como Executar

### 1. Clonar e Instalar
```bash
git clone [https://github.com/seu-usuario/projeto-libras.git](https://github.com/seu-usuario/projeto-libras.git)
cd projeto-libras
pip install -r requirements.txt
```

### 2. Preparar os Dados
Certifique-se de que o dataset está na pasta `dataset/train` e execute:
```bash
python scripts/coleta.py
```

### 3. Treinar a IA
```bash
python scripts/treinar.py
```

### 4. Iniciar Tradutor
```bash
python scripts/main.py
```

---

## 📸 Screenshots

<p align="center">
  <img src="assets/screenshot1.png" width="400px" alt="Exemplo de Letra A">
  <img src="assets/screenshot2.png" width="400px" alt="Exemplo de Letra B">
</p>

---

## 🚀 Próximos Passos (Roadmap)

- [ ] Implementar reconhecimento de sinais que envolvem movimento (letras J, K, X, Z).
- [ ] Criar suporte para detecção de ambas as mãos simultaneamente.
- [ ] Exportar o modelo para rodar no navegador via TensorFlow.js.

---
> *Este projeto foi desenvolvido para fins de estudo em Inteligência Artificial e Acessibilidade.*
```
