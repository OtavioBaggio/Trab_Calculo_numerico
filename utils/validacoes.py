import numpy as np

"""
Calcula o determinante da matriz A
"""
def matriz_singular(A):
    return np.linalg.det(A) == 0


"""
Verifica se a matriz é diagonal dominante.
Percorre a linha, ignora diagonal e soma os demais valores
"""
def diagonal_dominante(A):
    for i in range(len(A)):
        # abs retira retorna o valor positivo
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(len(A)) if j != i):
            return False
    return True