
"""
Implementa o método iterativo. Começa com o chute inicial zerado e vai atribuindo os valores
a cada iteração até atingir a precisão desejada.
"""
def jacobi(A, b, tol=1e-2, max_iter=100):
    import numpy as np

    n = len(A)
    x = np.zeros(n)       # Chute inicial = 0
    x_novo = np.zeros(n)  # Vetor que guarda os novos valores
    erros = []            # Lista que guarda os erros das iterações
    historico = []        # Lista q armazena o vetor x de cada iteração

    for k in range(max_iter):
        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Divisão por zero detectada na diagonal")

            # Soma todos os termos da linha i, exceto o da diagonal:
            soma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            # Calcula a próxima iteração
            x_novo[i] = (b[i] - soma) / A[i][i]

        # Calcula o erro:
        erro = np.linalg.norm(x_novo - x)
        erros.append(erro)
        historico.append(x_novo.copy())  # Salva o x atual

        # Critério de parada:
        if erro < tol:
            return x_novo, k+1, erros, historico

        x = x_novo.copy()

    raise ValueError("Jacobi não convergiu")