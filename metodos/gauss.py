import numpy as np

"""
Implementa o método direto, resolvendo o sistema em uma única passagem.
A função transforma um sistema triangular superior, onde só a diagonal e os elementos acima têm
valor. Depois resolve de baixo pra cima.
"""
def eliminacao_gauss(A, b):
    n = len(A)
    # Garante que os números são decimais:
    A = A.astype(float)
    b = b.astype(float)

    # Eliminação
    for i in range(n):  # Para cada linha do pivô
        # Se for zero ele mostra erro, pois não da pra dividir
        if A[i][i] == 0:
            raise ValueError("Divisão por zero - matriz singular")

        for j in range(i+1, n):  # Para cada linha abaixo do pivô
            fator = A[j][i] / A[i][i]
            
            # Subtrai da linha j a linha pivô multiplicada pelo fator
            # Assim, zera A[j][i]
            A[j] = A[j] - fator * A[i]
            b[j] = b[j] - fator * b[i]

    # Substituição retroativa
    x = np.zeros(n)

    for i in range(n-1, -1, -1): # de baixo pra cima
        # soma os que ja foram resolvidos:
        soma = sum(A[i][j] * x[j] for j in range(i+1, n))
        # move a soma para o outro lado e divide pelo coef:
        x[i] = (b[i] - soma) / A[i][i]

    return x    
