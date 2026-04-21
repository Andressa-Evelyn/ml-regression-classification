import random
from metrics import calcular_metricas

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

    for fold_id, (X_train, X_test, y_train, y_test) in enumerate(
        tqdm(folds, desc="K-Fold")
    ):
        preds = []

        for x in X_test:
            pred = knn_classificacao(
                X_train, y_train, x, k_vizinhos, distancia_func
            )
            preds.append(pred)

        acc, p, r, f1 = calcular_metricas(y_test, preds)
        resultados.append((acc, p, r, f1))

    return resultados