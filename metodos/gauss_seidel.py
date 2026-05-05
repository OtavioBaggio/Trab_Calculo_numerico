import numpy as np

"""
Evolução do Jacobi, com grande diferença aonde os novos valor são usados
na iteração.
"""
def gauss_seidel(A, b, tol=1e-2, max_iter=100):
    
    n = len(A)
    x = np.zeros(n)     # Chute inicial = 0
    erros = []
    historico = []

    for k in range(max_iter):
        x_antigo = x.copy()    # guarda o x do início da iteração

        #Conferindo se há zero na diag principal:
        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Divisão por zero")

            # Aqui usa os valores novos: x[j]
            soma1 = sum(A[i][j] * x[j] for j in range(i))
            # Usa os valor antigos pq ainda nao foram recalculadas:
            soma2 = sum(A[i][j] * x_antigo[j] for j in range(i+1, n))
            # Calculo da proxima aproximação
            x[i] = (b[i] - soma1 - soma2) / A[i][i]

        # Cálculo do erro:
        erro = np.linalg.norm(x - x_antigo)
        erros.append(erro)
        historico.append(x.copy())

        if erro < tol:
            return x, k+1, erros, historico

    raise ValueError("Gauss-Seidel não convergiu")