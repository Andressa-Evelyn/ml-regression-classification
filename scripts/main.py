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

from metrics import resumo_metricas, formatar, resumo_metricas_regressao, gerar_linha_tabela, gerar_linha_tabela_completa, gerar_linha_tabela_classificacao, gerar_linha_tabela_regressao

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

def extrair_metricas(resultados):
    acc = [r[0] for r in resultados]
    prec = [r[1] for r in resultados]
    rec = [r[2] for r in resultados]
    f1 = [r[3] for r in resultados]
    return acc, prec, rec, f1

#Classificação: 9999 amostras
#Regressão: 10692 amostras

# Euclidiana
res_euc, t_tr_euc, t_te_euc = kfold_knn(Xc, yc, k_folds=3, k_vizinhos=3, distancia_func=distancia_euclidiana)


print("\nKNN - Euclidiana")
resumo_metricas(res_euc)

# Manhattan
res_man, t_tr_man, t_te_man = kfold_knn(Xc, yc, k_folds=3, k_vizinhos=3, distancia_func=distancia_manhattan)

print("\nKNN - Manhattan")
resumo_metricas(res_man)

#uniariado

# classificação (usa Xc, yc)
# UNIVARIADO
accs_uni, precs_uni, rcs_uni, f1s_uni, t_train_uni, t_test_uni = avaliar_kfold(Xc, yc, k=3, tipo="uni")


print("Naive Bayes Univariado:")
print("Accuracy:", formatar(accs_uni))
print("Precision:", formatar(precs_uni))
print("Recall:", formatar(rcs_uni))
print("F1:", formatar(f1s_uni))
print("Tempo treino:", formatar(t_train_uni))
print("Tempo teste:", formatar(t_test_uni))

# multivariado

# MULTIVARIADO
accs_multi, precs_multi, rcs_multi, f1s_multi, t_train_multi, t_test_multi = avaliar_kfold(Xc, yc, k=3, tipo="multi")

print("\nNaive Bayes Multivariado:")
print("Accuracy:", formatar(accs_multi))
print("Precision:", formatar(precs_multi))
print("Recall:", formatar(rcs_multi))
print("F1:", formatar(f1s_multi))
print("Tempo treino:", formatar(t_train_multi))
print("Tempo teste:", formatar(t_test_multi))


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


print("\n================ TABELA FINAL - CLASSIFICAÇÃO ================")
print(f"{'Modelo':<30} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1':<15} {'Treino':<15} {'Teste':<15}")

acc_euc, prec_euc, rec_euc, f1_euc = extrair_metricas(res_euc)
acc_man, prec_man, rec_man, f1_man = extrair_metricas(res_man)

print(gerar_linha_tabela_classificacao("KNN Euclidiana", acc_euc, prec_euc, rec_euc, f1_euc, t_tr_euc, t_te_euc))

print(gerar_linha_tabela_classificacao("KNN Manhattan", acc_man, prec_man, rec_man, f1_man, t_tr_man, t_te_man))

print(gerar_linha_tabela_classificacao("Naive Bayes Uni", accs_uni, precs_uni, rcs_uni, f1s_uni, t_train_uni, t_test_uni))

print(gerar_linha_tabela_classificacao("Naive Bayes Multi", accs_multi, precs_multi, rcs_multi, f1s_multi, t_train_multi, t_test_multi))


print("\n================ TABELA REGRESSÃO ================")
print(f"{'Modelo':<30} {'R2':<15} {'R2 Ajustado':<15} {'Treino':<15} {'Teste':<15}")

r2_knn = [r[0] for r in res_knn_reg]
r2adj_knn = [r[1] for r in res_knn_reg]

r2_lin = [r[0] for r in res_lin]
r2adj_lin = [r[1] for r in res_lin]

print(gerar_linha_tabela_regressao("KNN Regressão", r2_knn, r2adj_knn, t_tr_knn, t_te_knn))

print(gerar_linha_tabela_regressao("Regressão Linear", r2_lin, r2adj_lin, t_tr_lin, t_te_lin))