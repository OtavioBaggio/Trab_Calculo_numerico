def jacobi(A, b, tol=1e-6, max_iter=100):
    import numpy as np

    n = len(A)
    x = np.zeros(n)
    x_new = np.zeros(n)
    erros = []
    historico = []

    for k in range(max_iter):
        for i in range(n):
            if A[i][i] == 0:
                raise ValueError("Divisão por zero detectada na diagonal")

            soma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - soma) / A[i][i]

        erro = np.linalg.norm(x_new - x)
        erros.append(erro)
        historico.append(x_new.copy())

        if erro < tol:
            return x_new, k+1, erros, historico

        x = x_new.copy()

    raise ValueError("Jacobi não convergiu")