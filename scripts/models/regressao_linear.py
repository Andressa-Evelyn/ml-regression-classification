import numpy as np
from models.knn import criar_folds
from metrics import calcular_metricas_regressao
import time
import random
from tqdm import tqdm

def random_oversampling(X, y):
    """
    Equaliza o número de instâncias de todas as classes duplicando 
    amostras da classe minoritária aleatoriamente.
    """
    X = np.array(X)
    y = np.array(y)
    classes = np.unique(y)
    
    # Encontra o tamanho da maior classe
    max_size = max([sum(y == c) for c in classes])
    
    X_resampled, y_resampled = [], []
    
    for c in classes:
        X_c = X[y == c]
        y_c = y[y == c]
        
        # Seleciona índices aleatórios com reposição para igualar ao max_size
        indices_aleatorios = np.random.choice(len(X_c), max_size, replace=True)
        
        X_resampled.extend(X_c[indices_aleatorios])
        y_resampled.extend(y_c[indices_aleatorios])
        
    return np.array(X_resampled).tolist(), np.array(y_resampled).tolist()

def treinar_regressao_linear(X, y):
    X = np.array(X)
    y = np.array(y)

    # adiciona bias (coluna de 1)
    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    # fórmula normal: (X^T X)^-1 X^T y
    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.inv(XtX)
    XtY = np.dot(X.T, y)

    w = np.dot(XtX_inv, XtY)

    return w


def prever_regressao_linear(w, X):
    X = np.array(X)

    ones = np.ones((X.shape[0], 1))
    X = np.hstack((ones, X))

    return np.dot(X, w)

def kfold_regressao_linear(X, y, k):
    folds = criar_folds(X, y, k)

    resultados = []
    tempos_treino = []
    tempos_teste = []

    for X_train, X_test, y_train, y_test in tqdm(folds, desc="Regressão Linear K-Fold"):

        inicio_treino = time.time()
        w = treinar_regressao_linear(X_train, y_train)
        fim_treino = time.time()

        inicio_teste = time.time()
        preds = prever_regressao_linear(w, X_test)
        fim_teste = time.time()

        r2, r2_adj = calcular_metricas_regressao(y_test, preds, len(X[0]))

        resultados.append((r2, r2_adj))
        tempos_treino.append(fim_treino - inicio_treino)
        tempos_teste.append(fim_teste - inicio_teste)

    return resultados, tempos_treino, tempos_teste