import random
from collections import defaultdict

def carregar_arff(caminho):
    X, y = [], []
    lendo_dados = False

    with open(caminho, 'r') as f:
        for linha in f:
            linha = linha.strip()

            if not linha or linha.startswith("%"):
                continue

            if linha.lower() == "@data":
                lendo_dados = True
                continue

            if lendo_dados:
                partes = linha.split(",")

                atributos = partes[:-1]
                try:
                    classe = float(partes[-1])
                except:
                    classe = partes[-1]
                

                nova_linha = []
                for val in atributos:
                    try:
                        nova_linha.append(float(val))
                    except:
                        nova_linha.append(val)

                X.append(nova_linha)
                y.append(classe)

    return X, y

def codificar_categoricos(X):
    X = [linha[:] for linha in X]  # cópia
    
    mapas = [{} for _ in range(len(X[0]))]

    for j in range(len(X[0])):
        valor_id = 0
        for i in range(len(X)):
            val = X[i][j]

            if isinstance(val, str):
                if val not in mapas[j]:
                    mapas[j][val] = valor_id
                    valor_id += 1
                X[i][j] = mapas[j][val]

    return X


def amostra_estratificada(X, y, n_total):
    grupos = defaultdict(list)

    for i in range(len(y)):
        grupos[y[i]].append(i)

    X_out, y_out = [], []

    for classe, indices in grupos.items():
        proporcao = len(indices) / len(y)
        n_classe = max(1, int(proporcao * n_total))  # evita zero

        escolhidos = random.sample(indices, min(n_classe, len(indices)))

        for i in escolhidos:
            X_out.append(X[i])
            y_out.append(y[i])

    return X_out, y_out

def normalizar(X):
    X = [list(map(float, linha)) for linha in X]

    mins = [min(col) for col in zip(*X)]
    maxs = [max(col) for col in zip(*X)]

    X_norm = []
    for linha in X:
        nova = []
        for i in range(len(linha)):
            if maxs[i] == mins[i]:
                nova.append(0.0)
            else:
                nova.append((linha[i] - mins[i]) / (maxs[i] - mins[i]))
        X_norm.append(nova)

    return X_norm