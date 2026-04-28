import numpy as np

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    
    n = len(A)
    x = np.zeros(n)
    erros = []
    historico = []

    for k in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Divisão por zero")

            soma1 = sum(A[i][j] * x[j] for j in range(i))
            soma2 = sum(A[i][j] * x_old[j] for j in range(i+1, n))
            x[i] = (b[i] - soma1 - soma2) / A[i][i]

        erro = np.linalg.norm(x - x_old)
        erros.append(erro)
        historico.append(x.copy())

        if erro < tol:
            return x, k+1, erros, historico

    raise ValueError("Gauss-Seidel não convergiu")