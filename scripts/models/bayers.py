import numpy as np
from metrics import calcular_metricas
import time
import math
from models.knn import criar_folds


def prob_multivariada(x, mu, inv, det):
    d = len(mu)
    x_mu = x - mu
    
    expoente = -0.5 * np.dot(np.dot(x_mu, inv), x_mu.T)
    
    return (1 / (np.sqrt((2 * np.pi) ** d * det))) * np.exp(expoente)


def prever_multivariado(modelo, x):
    probabilidades = {}
    
    for classe, dados in modelo.items():
        prob = prob_multivariada(
            np.array(x),
            dados["mu"],
            dados["inv"],
            dados["det"]
        )
        probabilidades[classe] = prob
    
    return max(probabilidades, key=probabilidades.get)

# separa dados por classe
def separar_por_classe(X, y):
    classes = {}
    for i in range(len(y)):
        classe = y[i]
        if classe not in classes:
            classes[classe] = []
        classes[classe].append(X[i])
    return classes

# calcula média e desvio padrão por atributo
def resumo_dataset(dataset):
    resumo = []
    dataset = np.array(dataset)
    
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        media = np.mean(col)
        desvio = np.std(col)
        resumo.append((media, desvio))
    
    return resumo

# treinar
def treinar_naive_bayes_univariado(X, y):
    separados = separar_por_classe(X, y)
    modelo = {}
    
    for classe, dados in separados.items():
        modelo[classe] = resumo_dataset(dados)
    
    return modelo

# pdf gaussiana
def probabilidade_gaussiana(x, media, desvio):
    if desvio == 0:
        return 1 if x == media else 1e-9
    
    expoente = math.exp(-((x - media) ** 2) / (2 * desvio ** 2))
    return (1 / (math.sqrt(2 * math.pi) * desvio)) * expoente

# prever uma amostra
def prever_univariado(modelo, x):
    probabilidades = {}
    
    for classe, resumo in modelo.items():
        prob = 1
        for i in range(len(resumo)):
            media, desvio = resumo[i]
            prob *= probabilidade_gaussiana(x[i], media, desvio)
        
        probabilidades[classe] = prob
    
    return max(probabilidades, key=probabilidades.get)

def avaliar_univariado(X_train, y_train, X_test, y_test):
    inicio_treino = time.time()
    modelo = treinar_naive_bayes_univariado(X_train, y_train)
    fim_treino = time.time()
    
    inicio_teste = time.time()
    y_pred = [prever_univariado(modelo, x) for x in X_test]
    fim_teste = time.time()
    
    acc, p, r, f1 = calcular_metricas(y_test, y_pred)
    
    return acc, p, r, f1, (fim_treino - inicio_treino), (fim_teste - inicio_teste)

def treinar_bayes_multivariado(X, y):
    separados = separar_por_classe(X, y)
    modelo = {}
    
    for classe, dados in separados.items():
        dados = np.array(dados)
        
        mu = np.mean(dados, axis=0)
        sigma = np.cov(dados, rowvar=False)
        
        # 🔥 correção aqui
        sigma += np.eye(sigma.shape[0]) * 1e-6
        
        modelo[classe] = {
            "mu": mu,
            "sigma": sigma,
            "inv": np.linalg.inv(sigma),
            "det": np.linalg.det(sigma)
        }
    
    return modelo

def avaliar_multivariado(X_train, y_train, X_test, y_test):
    inicio_treino = time.time()
    modelo = treinar_bayes_multivariado(X_train, y_train)
    fim_treino = time.time()
    
    inicio_teste = time.time()
    y_pred = [prever_multivariado(modelo, x) for x in X_test]
    fim_teste = time.time()
    
    acc, p, r, f1 = calcular_metricas(y_test, y_pred)
    
    return acc, p, r, f1, (fim_treino - inicio_treino), (fim_teste - inicio_teste)

def avaliar_kfold(X, y, k=3, tipo="uni"):
    folds = criar_folds(X, y, k)

    accs, precs, f1s = [], [], []
    tempos_treino, tempos_teste = [], []

    for X_train, X_test, y_train, y_test in folds:
        
        if tipo == "uni":
            acc, p, r, f1, t_tr, t_te = avaliar_univariado(
                X_train, y_train, X_test, y_test
            )
        else:
            acc, p, r, f1, t_tr, t_te = avaliar_multivariado(
                X_train, y_train, X_test, y_test
            )

        accs.append(acc)
        precs.append(p)
        f1s.append(f1)
        tempos_treino.append(t_tr)
        tempos_teste.append(t_te)

    return accs, precs, f1s, tempos_treino, tempos_teste


