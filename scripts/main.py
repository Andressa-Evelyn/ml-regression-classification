from datasets import (
    carregar_arff,
    codificar_categoricos,
    amostra_estratificada,
    normalizar
)

from models.knn import (
    kfold_knn,
    distancia_euclidiana,
    distancia_manhattan,
    kfold_knn_regressao
)

from metrics import resumo_metricas, formatar, resumo_metricas_regressao, gerar_linha_tabela

from models.bayers import avaliar_kfold

from models.regressao_linear import kfold_regressao_linear

# carregar dados
Xc, yc = carregar_arff("datasets/BNG(credit-g)_classificacao.arff")
Xr, yr = carregar_arff("datasets/file22f1627e4a960_regressao.arff")

# preprocessamento
Xc = codificar_categoricos(Xc)
Xr = codificar_categoricos(Xr)

# amostra (só classificação)
Xc, yc = amostra_estratificada(Xc, yc, 10000)

# normalização
Xc = normalizar(Xc)
Xr = normalizar(Xr)

print("Classificação:", len(Xc), "amostras")
print("Regressão:", len(Xr), "amostras")

#Classificação: 9999 amostras
#Regressão: 10692 amostras

# Euclidiana
res_euc = kfold_knn(Xc, yc, k_folds=3, k_vizinhos=3, distancia_func=distancia_euclidiana)

print("\nKNN - Euclidiana")
resumo_metricas(res_euc)

# Manhattan
res_man = kfold_knn(Xc, yc, k_folds=3, k_vizinhos=3, distancia_func=distancia_manhattan)

print("\nKNN - Manhattan")
resumo_metricas(res_man)

#uniariado

# classificação (usa Xc, yc)
accs, precs, f1s, t_train, t_test = avaliar_kfold(Xc, yc, k=3, tipo="uni")

print("Naive Bayes Univariado:")
print("Accuracy:", formatar(accs))
print("Precision:", formatar(precs))
print("F1:", formatar(f1s))
print("Tempo treino:", formatar(t_train))
print("Tempo teste:", formatar(t_test))

# multivariado

accs, precs, f1s, t_train, t_test = avaliar_kfold(Xc, yc, k=3, tipo="multi")

print("\nNaive Bayes Multivariado:")
print("Accuracy:", formatar(accs))
print("Precision:", formatar(precs))
print("F1:", formatar(f1s))
print("Tempo treino:", formatar(t_train))
print("Tempo teste:", formatar(t_test))


print("\n================ REGRESSÃO ================")

# KNN regressão
res_knn_reg, t_tr_knn, t_te_knn = kfold_knn_regressao(
    Xr, yr, k_folds=3, k_vizinhos=3, distancia_func=distancia_euclidiana
)

print("\nKNN Regressão:")
resumo_metricas_regressao(res_knn_reg)
print("Tempo treino:", formatar(t_tr_knn))
print("Tempo teste:", formatar(t_te_knn))


# Regressão Linear
res_lin, t_tr_lin, t_te_lin = kfold_regressao_linear(Xr, yr, k=3)

print("\nRegressão Linear:")
resumo_metricas_regressao(res_lin)
print("Tempo treino:", formatar(t_tr_lin))
print("Tempo teste:", formatar(t_te_lin))

print("\n================ TABELA FINAL ================")
print(f"{'Modelo':<30} {'Accuracy':<15} {'Precision':<15} {'F1':<15}")

print(gerar_linha_tabela("KNN Euclidiana", res_euc))
print(gerar_linha_tabela("KNN Manhattan", res_man))
print(gerar_linha_tabela("Naive Bayes Uni", list(zip(accs, precs, [0]*len(accs), f1s))))