import numpy as np
from models.knn import criar_folds
from metrics import calcular_metricas_regressao
import time

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

    for X_train, X_test, y_train, y_test in folds:

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