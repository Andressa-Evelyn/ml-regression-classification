import random
from metrics import calcular_metricas, calcular_metricas_regressao
import time

def criar_folds(X, y, k):
    dados = list(zip(X, y))
    random.shuffle(dados)  # 🔥 MUITO IMPORTANTE

    fold_size = len(dados) // k
    folds = []

    for i in range(k):
        inicio = i * fold_size
        fim = inicio + fold_size

        teste = dados[inicio:fim]
        treino = dados[:inicio] + dados[fim:]

        X_train = [x for x, y in treino]
        y_train = [y for x, y in treino]

        X_test = [x for x, y in teste]
        y_test = [y for x, y in teste]

        folds.append((X_train, X_test, y_train, y_test))

    return folds

def knn_classificacao(X_train, y_train, x_teste, k, distancia_func):
    distancias = []

    for i in range(len(X_train)):
        d = distancia_func(x_teste, X_train[i])
        distancias.append((d, y_train[i]))

    distancias.sort(key=lambda x: x[0])
    vizinhos = distancias[:k]

    contagem = {}

    for dist, classe in vizinhos:
        if classe not in contagem:
            contagem[classe] = 0
        contagem[classe] += 1

    # 🔥 critério de desempate: menor distância média
    max_votos = max(contagem.values())
    candidatos = [c for c in contagem if contagem[c] == max_votos]

    if len(candidatos) == 1:
        return candidatos[0]

    # desempate pela menor distância média
    soma_dist = {c: 0 for c in candidatos}
    for dist, classe in vizinhos:
        if classe in soma_dist:
            soma_dist[classe] += dist

    return min(soma_dist, key=soma_dist.get)

def distancia_euclidiana(x, y):
    soma = 0
    for i in range(len(x)):
        soma += (x[i] - y[i]) ** 2
    return soma  # sem sqrt

def distancia_manhattan(x, y):
    soma = 0
    for i in range(len(x)):
        soma += abs(x[i] - y[i])
    return soma


from tqdm import tqdm

def kfold_knn(X, y, k_folds, k_vizinhos, distancia_func):
    folds = criar_folds(X, y, k_folds)

    resultados = []
    tempos_treino = []
    tempos_teste = []

    for X_train, X_test, y_train, y_test in folds:

        inicio_treino = time.time()
        # KNN não treina
        fim_treino = time.time()

        inicio_teste = time.time()

        preds = []
        for x in X_test:
            pred = knn_classificacao(
                X_train, y_train, x, k_vizinhos, distancia_func
            )
            preds.append(pred)

        fim_teste = time.time()

        acc, p, r, f1 = calcular_metricas(y_test, preds)

        resultados.append((acc, p, r, f1))
        tempos_treino.append(fim_treino - inicio_treino)
        tempos_teste.append(fim_teste - inicio_teste)

    return resultados, tempos_treino, tempos_teste

def knn_regressao(X_train, y_train, x_teste, k, distancia_func):
    distancias = []

    for i in range(len(X_train)):
        d = distancia_func(x_teste, X_train[i])
        distancias.append((d, y_train[i]))

    distancias.sort(key=lambda x: x[0])
    vizinhos = distancias[:k]

    soma = 0
    for dist, valor in vizinhos:
        soma += valor

    return soma / k

def kfold_knn_regressao(X, y, k_folds, k_vizinhos, distancia_func):
    folds = criar_folds(X, y, k_folds)

    resultados = []
    tempos_treino = []
    tempos_teste = []

    for X_train, X_test, y_train, y_test in folds:

        inicio_treino = time.time()
        # KNN não treina, mas vamos considerar preparação
        fim_treino = time.time()

        inicio_teste = time.time()
        preds = []

        for x in X_test:
            pred = knn_regressao(
                X_train, y_train, x, k_vizinhos, distancia_func
            )
            preds.append(pred)
        fim_teste = time.time()

        r2, r2_adj = calcular_metricas_regressao(y_test, preds, len(X[0]))

        resultados.append((r2, r2_adj))
        tempos_treino.append(fim_treino - inicio_treino)
        tempos_teste.append(fim_teste - inicio_teste)

    return resultados, tempos_treino, tempos_teste