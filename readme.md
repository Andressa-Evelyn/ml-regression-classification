# 📊 Projeto de Aprendizado de Máquina  
## KNN e Naive Bayes com Validação Cruzada

Este projeto implementa, do zero, algoritmos clássicos de aprendizado de máquina para **classificação supervisionada**, com avaliação baseada em **K-Fold Cross Validation**.

---

## 🚀 Objetivo

Comparar o desempenho de:

- K-Nearest Neighbors (KNN)
- Naive Bayes (Univariado e Multivariado)

utilizando diferentes métricas de avaliação e analisando o comportamento dos modelos.

---

## 🧠 Algoritmos Implementados

### 🔹 KNN (K-Nearest Neighbors)

- Classificação baseada em vizinhos mais próximos
- Distâncias implementadas:
  - Euclidiana
  - Manhattan
- Critério de desempate:
  - menor distância média

---

### 🔹 Naive Bayes

#### ✔️ Univariado (Gaussiano)
- Assume independência entre atributos
- Cada atributo modelado com distribuição normal

#### ✔️ Multivariado
- Considera correlação entre atributos
- Utiliza matriz de covariância

---

## ⚙️ Pré-processamento

Antes do treinamento, os dados passam por:

- Codificação de atributos categóricos
- Normalização dos dados
- Amostragem estratificada (para classificação)

---

## 🔁 Validação

Foi utilizada:

👉 **K-Fold Cross Validation (k=3)**

- Os dados são divididos em 3 partes
- Em cada iteração:
  - 2 folds → treino
  - 1 fold → teste

---

## 📏 Métricas Avaliadas

- Accuracy
- Precision
- Recall
- F1-score

---

## ⏱️ Métricas adicionais

Para o Naive Bayes:

- Tempo de treino
- Tempo de teste

---
