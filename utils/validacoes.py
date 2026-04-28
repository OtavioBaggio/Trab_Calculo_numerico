import numpy as np

def matriz_singular(A):
    return np.linalg.det(A) == 0

def diagonal_dominante(A):
    for i in range(len(A)):
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(len(A)) if j != i):
            return False
    return True