import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from metodos import eliminacao_gauss, jacobi, gauss_seidel
from utils import matriz_singular, diagonal_dominante


# =========================
# Interface
# =========================

def titulo():
    st.title("Solucionador de Sistemas Lineares")


def entrada_tamanho():
    return st.number_input("Tamanho do sistema (n x n)", min_value=2, max_value=10, value=3)


def entrada_matriz(n):
    st.subheader("Matriz A")
    A = []
    for i in range(n):
        cols = st.columns(n)
        linha = [cols[j].number_input(f"A[{i}][{j}]", key=f"A{i}{j}") for j in range(n)]
        A.append(linha)
    return np.array(A, dtype=float)


def entrada_vetor(n):
    st.subheader("Vetor b")
    b = [st.number_input(f"b[{i}]", key=f"b{i}") for i in range(n)]
    return np.array(b, dtype=float)


def entrada_parametros():
    tol = st.number_input("Tolerância", value=1e-2, format="%.4f")
    max_iter = st.number_input("Máx Iterações", value=100)
    return tol, max_iter


def escolher_metodo():
    return st.selectbox("Método", ["Gauss", "Jacobi", "Gauss-Seidel"])


# =========================
# Gráfico
# =========================

def plotar_convergencia(erros):
    fig, ax = plt.subplots()
    ax.plot(erros, marker='o')
    ax.set_xlabel("Iterações")
    ax.set_ylabel("Erro")
    ax.set_title("Convergência do Método")
    ax.grid(True)
    ax.set_yscale('log')
    st.pyplot(fig)


# =========================
# Tabela de Iterações
# =========================

def mostrar_tabela_iteracoes(historico, erros):
    dados = []

    for i, (x, erro) in enumerate(zip(historico, erros)):
        linha = {"Iteração": i + 1, "Erro": erro}
        for j, val in enumerate(x):
            linha[f"x{j+1}"] = val
        dados.append(linha)

    df = pd.DataFrame(dados)
    st.subheader("Tabela de Iterações")
    st.dataframe(df)


# =========================
# Lógica
# =========================

def resolver_sistema(A, b, metodo, tol, max_iter):
    if matriz_singular(A):
        raise ValueError("Matriz singular! Não é possível resolver.")

    if metodo == "Gauss":
        x = eliminacao_gauss(A, b)
        return x, None, None, None

    elif metodo == "Jacobi":
        if not diagonal_dominante(A):
            st.warning("A matriz pode não convergir (não é diagonal dominante)")
        x, it, erros, historico = jacobi(A, b, tol, max_iter)
        return x, it, erros, historico

    elif metodo == "Gauss-Seidel":
        if not diagonal_dominante(A):
            st.warning("A matriz pode não convergir (não é diagonal dominante)")
        x, it, erros, historico = gauss_seidel(A, b, tol, max_iter)
        return x, it, erros, historico


# =========================
# Saída
# =========================

def mostrar_resultado(x, iteracoes):
    x = np.array(x).flatten()

    st.success("Solução:")
    for i, valor in enumerate(x):
        st.write(f"x{i+1} = {valor:.6f}")

    if iteracoes is not None:
        st.info(f"Número de iterações: {iteracoes}")


def mostrar_erro(e):
    st.error(str(e))


# =========================
# Main
# =========================

def main():
    titulo()

    n = entrada_tamanho()
    A = entrada_matriz(n)
    b = entrada_vetor(n)

    tol, max_iter = entrada_parametros()
    metodo = escolher_metodo()

    if st.button("Resolver"):
        try:
            if A.size == 0 or b.size == 0:
                st.error("Preencha todos os valores da matriz e do vetor.")
                return

            x, it, erros, historico = resolver_sistema(A, b, metodo, tol, max_iter)

            mostrar_resultado(x, it)

            if erros is not None:
                plotar_convergencia(erros)

            if historico is not None:
                mostrar_tabela_iteracoes(historico, erros)

        except Exception as e:
            mostrar_erro(e)


if __name__ == "__main__":
    main()