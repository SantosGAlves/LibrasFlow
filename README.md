# 🤟 LibrasFlow

<p align="center">
  <img src="assets/banner_librasflow.png" alt="LibrasFlow Banner" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
  <img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Mediapipe-0473FF?style=for-the-badge&logo=google&logoColor=white" />
</p>

<p align="center">
  <i>Tradutor inteligente de Língua Brasileira de Sinais (LIBRAS) em tempo real utilizando visão computacional e machine learning.</i>
</p>

---

## 📌 Sobre o Projeto

O **LibrasFlow** é uma aplicação que utiliza **Visão Computacional** e **Machine Learning** para reconhecer sinais do alfabeto em LIBRAS em tempo real via webcam.

A solução captura os movimentos da mão, transforma em dados estruturados e realiza a classificação das letras, permitindo a formação de palavras com feedback visual instantâneo.

> 🎯 Objetivo: Tornar a comunicação mais acessível por meio de tecnologia leve e eficiente.

---

## 🎥 Demonstração

<p align="center">
  <img src="assets/demonstracao.gif" alt="Demonstração do Projeto" width="700px">
</p>

> 💡 *Sugestão: adicione aqui um vídeo do projeto rodando (YouTube ou LinkedIn)*

---

## ✨ Funcionalidades

- 🖐️ **Detecção de mãos em tempo real** com MediaPipe  
- 📍 **Extração de 21 landmarks tridimensionais**  
- 🧠 **Classificação com Machine Learning (Random Forest)**  
- 🔄 **Normalização de coordenadas** (independente da posição da mão)  
- ⏱️ **Filtro de estabilidade** para evitar falsos positivos  
- 🖥️ **Interface interativa na câmera (botão virtual)**  

---

## 🧠 Pipeline do Modelo

O sistema segue um fluxo estruturado de processamento:

1. **Captura de imagem (Webcam)**
2. **Detecção da mão (MediaPipe)**
3. **Extração dos landmarks (x, y, z)**
4. **Normalização das coordenadas**
5. **Geração de features**
6. **Classificação com Random Forest**
7. **Pós-processamento (filtro de estabilidade)**
8. **Exibição do resultado na interface**

---

## 🛠️ Arquitetura Técnica

O projeto não utiliza diretamente a imagem bruta. Em vez disso, trabalha com a **representação geométrica da mão**, tornando o sistema:

- ⚡ Mais rápido
- 🎯 Mais preciso
- 🌗 Menos sensível a iluminação e ruído

---

## 📊 Performance

| Métrica | Valor |
|--------|------|
| **Modelo** | Random Forest |
| **Tempo de resposta** | < 30ms (tempo real) |
| **Dataset** | ~34.000 imagens |
| **Acurácia (ambiente controlado)** | ~100% |

> ⚠️ A acurácia foi obtida em ambiente controlado. Em cenários reais, pode variar devido a iluminação, ângulo e execução dos sinais.

---

## 📂 Estrutura do Projeto

```bash
├── data/               # Dados processados (CSV)
├── dataset/            # Imagens para treinamento
├── models/             # Modelos treinados
├── scripts/
│   ├── coleta.py       # Extração de dados
│   ├── treinar.py      # Treinamento do modelo
│   └── main.py         # Execução em tempo real
├── requirements.txt
└── README.md
🚀 Como Executar
1. Clonar o repositório
git clone https://github.com/seu-usuario/projeto-libras.git
cd projeto-libras
2. Instalar dependências
pip install -r requirements.txt
3. Preparar os dados
python scripts/coleta.py
4. Treinar o modelo
python scripts/treinar.py
5. Executar aplicação
python scripts/main.py
💼 Aplicações Reais

Educação inclusiva

Ferramentas de acessibilidade

Interfaces humano-computador

Sistemas assistivos em tempo real

🚀 Roadmap

 Suporte a letras com movimento (J, K, X, Z)

 Reconhecimento com duas mãos

 Deploy em ambiente web (TensorFlow.js)

 API para integração com outras aplicações


