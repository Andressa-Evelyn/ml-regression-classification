import math

def matriz_confusao(y_real, y_pred, positivo=None):
    if positivo is None:
        positivo = list(set(y_real))[0]  # pega uma classe automaticamente

    TP = FP = TN = FN = 0

    for i in range(len(y_real)):
        if y_real[i] == positivo and y_pred[i] == positivo:
            TP += 1
        elif y_real[i] != positivo and y_pred[i] == positivo:
            FP += 1
        elif y_real[i] != positivo and y_pred[i] != positivo:
            TN += 1
        elif y_real[i] == positivo and y_pred[i] != positivo:
            FN += 1

    return TP, FP, TN, FN

def precision(TP, FP):
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def f1_score(p, r):
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

def media(valores):
    return sum(valores) / len(valores)

def desvio_padrao(valores):
    m = media(valores)
    soma = 0
    for v in valores:
        soma += (v - m) ** 2
    return math.sqrt(soma / len(valores))


def calcular_metricas(y_real, y_pred):
    TP, FP, TN, FN = matriz_confusao(y_real, y_pred)

    p = precision(TP, FP)
    r = recall(TP, FN)
    f1 = f1_score(p, r)

    acc = (TP + TN) / (TP + TN + FP + FN)

    return acc, p, r, f1


def formatar(valores):
    return f"{media(valores):.2f} ± {desvio_padrao(valores):.2f}"

def resumo_metricas(resultados):
    accs = [r[0] for r in resultados]
    ps = [r[1] for r in resultados]
    rs = [r[2] for r in resultados]
    f1s = [r[3] for r in resultados]

    print("Accuracy:", formatar(accs))
    print("Precision:", formatar(ps))
    print("Recall:", formatar(rs))
    print("F1:", formatar(f1s))


def resumo_metricas_regressao(resultados):
    r2s = [r[0] for r in resultados]
    r2_adjs = [r[1] for r in resultados]

    print("R2:", formatar(r2s))
    print("R2 Ajustado:", formatar(r2_adjs))


def gerar_linha_tabela(nome, resultados):
    accs = [r[0] for r in resultados]
    ps = [r[1] for r in resultados]
    f1s = [r[3] for r in resultados]

    return f"{nome:<30} {formatar(accs):<15} {formatar(ps):<15} {formatar(f1s):<15}"      

def gerar_linha_tabela_completa(nome, acc, prec, f1, t_train, t_test):
    return f"{nome:<30} {formatar(acc):<15} {formatar(prec):<15} {formatar(f1):<15} {formatar(t_train):<15} {formatar(t_test):<15}"  

def r2_score(y_real, y_pred):
    media_y = sum(y_real) / len(y_real)

    ss_total = sum((y - media_y) ** 2 for y in y_real)
    ss_res = sum((y_real[i] - y_pred[i]) ** 2 for i in range(len(y_real)))

    return 1 - (ss_res / ss_total)

def gerar_linha_tabela_classificacao(nome, acc, prec, rec, f1, t_train, t_test):
    return f"{nome:<30} {formatar(acc):<15} {formatar(prec):<15} {formatar(rec):<15} {formatar(f1):<15} {formatar(t_train):<15} {formatar(t_test):<15}"


def gerar_linha_tabela_regressao(nome, r2, r2_adj, t_train, t_test):
    return f"{nome:<30} {formatar(r2):<15} {formatar(r2_adj):<15} {formatar(t_train):<15} {formatar(t_test):<15}"

def r2_ajustado(r2, n, p):
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


def calcular_metricas_regressao(y_real, y_pred, p):
    r2 = r2_score(y_real, y_pred)
    r2_adj = r2_ajustado(r2, len(y_real), p)

    return r2, r2_adj

