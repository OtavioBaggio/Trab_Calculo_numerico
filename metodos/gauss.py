import numpy as np

def eliminacao_gauss(A, b):
    n = len(A)
    A = A.astype(float)
    b = b.astype(float)

    # Eliminação
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError("Divisão por zero - matriz singular")

        for j in range(i+1, n):
            fator = A[j][i] / A[i][i]
            A[j] = A[j] - fator * A[i]
            b[j] = b[j] - fator * b[i]

    # Substituição retroativa
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        soma = sum(A[i][j] * x[j] for j in range(i+1, n))
        x[i] = (b[i] - soma) / A[i][i]

    return x