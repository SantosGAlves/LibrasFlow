# 🤟 LIBRAS Vision: Tradutor de Sinais em Tempo Real

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Mediapipe](https://img.shields.io/badge/Mediapipe-0473FF?style=for-the-badge&logo=google&logoColor=white)

Este projeto utiliza Visão Computacional e Machine Learning para reconhecer o alfabeto da LIBRAS (Língua Brasileira de Sinais) através da webcam. O sistema conta com um corretor de estabilidade para evitar predições fantasmas e um botão virtual interativo para interface do usuário.

---

## 📺 Demonstração

<p align="center">
  <img src="link-do-seu-gif-aqui.gif" alt="Demonstração do Tradutor" width="600px">
</p>

## ✨ Funcionalidades

* **Detecção de Mãos:** Utiliza o Google Mediapipe para mapear 21 pontos de articulação em tempo real.
* **Reconhecimento de Alfabeto:** Modelo treinado com Random Forest para classificar as letras do alfabeto.
* **Filtro de Estabilidade:** Sistema de contagem de frames que só confirma a letra após uma detecção estável de 30 frames.
* **Interface Interativa (HUD):**
    * Exibição da frase formada no topo da tela.
    * **Botão Virtual:** Um botão "LIMPAR" que é acionado ao posicionar o dedo indicador sobre ele por 1.5 segundos.
* **Atalhos de Teclado:**
    * `Espaço`: Adiciona espaço à frase.
    * `Backspace`: Apaga o último caractere.
    * `Esc`: Encerra a aplicação.

## 🛠️ Tecnologias e Ferramentas

* **Dataset:** [Libras Dataset (Kaggle)](https://www.kaggle.com/) - Imagens processadas para extração de landmarks.
* **Extração de Features:** Mediapipe (Landmarks x, y).
* **Classificador:** Scikit-Learn (Random Forest Classifier).
* **Processamento de Imagem:** OpenCV.

## 🚀 Como Executar o Projeto

### 1. Clonar o repositório
```bash
git clone [https://github.com/seu-usuario/libras-vision.git](https://github.com/seu-usuario/libras-vision.git)
cd libras-vision
