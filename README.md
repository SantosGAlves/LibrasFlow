# 🤟 Tradutor de LIBRAS em Tempo Real com Visão Computacional

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-04A8E1?style=for-the-badge&logo=google&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Este projeto utiliza inteligência artificial para reconhecer o alfabeto da Língua Brasileira de Sinais (LIBRAS) através da webcam. O sistema extrai marcos (landmarks) das mãos, utiliza um modelo de Machine Learning para classificação e permite a construção de frases em tempo real, contando inclusive com um **botão virtual interativo**.

---

## 📺 Demonstração

<div align="center">
  <img src="https://via.placeholder.com/800x450.png?text=Coloque+um+GIF+do+projeto+aqui" alt="Demonstração do Tradutor" width="700">
  <p><em>Legenda: O sistema detecta a mão, prevê a letra e permite "clicar" no botão virtual de limpeza.</em></p>
</div>

---

## ✨ Funcionalidades

* **Detecção de Mãos:** Utiliza o Google MediaPipe para identificar 21 pontos-chave da mão.
* **Classificação Inteligente:** Modelo Random Forest treinado para alta precisão.
* **Estabilização de Leitura:** Lógica de frames para confirmar uma letra apenas após permanência na posição, evitando "ruído" na frase.
* **Interface Interativa:** * Botão Virtual: O usuário pode limpar a frase apenas posicionando o dedo sobre o botão na tela.
    * Feedback Visual: Barras de progresso e cores indicam quando uma ação está prestes a ocorrer.
    * Controle via Teclado: `Espaço` para adicionar espaços e `Backspace` para apagar letras.

---

## 🚀 Como funciona?

O projeto é dividido em três etapas principais:

### 1. Coleta de Dados (`coleta.py`)
Varre um dataset de imagens organizado em pastas (A, B, C...). Para cada imagem, o MediaPipe extrai as coordenadas $(x, y)$ dos 21 pontos da mão e as salva em um arquivo `libras_dados.csv`.

### 2. Treinamento (`treinar.py`)
Lê o arquivo CSV, processa os dados com **Pandas** e treina um classificador **Random Forest** usando o **Scikit-Learn**. O modelo final é exportado via **Pickle**.

### 3. Execução em Tempo Real (`main.py`)
Acessa a webcam, processa cada frame, faz a predição da letra atual e gerencia a lógica da frase e do botão virtual.

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.x**
* **OpenCV:** Manipulação de vídeo e interface gráfica.
* **MediaPipe:** Extração de landmarks das mãos.
* **Scikit-Learn:** Criação e treinamento do modelo de Machine Learning.
* **Pandas/Numpy:** Manipulação e estruturação de dados.

---

## 📦 Como Instalar e Rodar

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
   cd seu-repositorio
