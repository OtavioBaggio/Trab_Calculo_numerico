#Bibliotecas:
import streamlit as st            # Cria a página web do projeto
import numpy as np                # Operações com matrizes e vetores
import matplotlib.pyplot as plt   # Usada no gráfico da convergência
import pandas as pd               # Cria a tabela de iterações

from metodos import eliminacao_gauss, jacobi, gauss_seidel
from utils import matriz_singular, diagonal_dominante


# =========================
# Interface
# =========================


def titulo():
    """
    Função para exibir o título principal.
    st.title cria um elemento <h1> na página com texto grande
    """

    st.title("Solucionador de Sistemas Lineares")



def entrada_tamanho():
    """
    Cria um campo numérico para o usuário escolher o tamanho do sistema,
    que é de mínimo 2, máximo 10 e padrão 3 qnd a página carrega.
    Retorna o valor escolhido para ser usado nas próximas funções
    """
    
    return st.number_input("Tamanho do sistema (n x n)", min_value=2, max_value=10, value=3)



def entrada_matriz(n):
    """
    Recebe o tamanho de n e gera a grade de campos de entradas da matriz
    """
    
    st.subheader("Matriz A")    # Subcabeçalho
    A = []
    for i in range(n):          # Percorre cada linha
        cols = st.columns(n)    # Divide a linha em colunas
        # Campo para cada coluna com uma chave para o streamlit:
        linha = [cols[j].number_input(f"A[{i}][{j}]", key=f"A{i}{j}") for j in range(n)]
        # Adiciona na lista:
        A.append(linha)
    return np.array(A, dtype=float)     # Tive que converter para array numpy



def entrada_vetor(n):
    """
    Cria os campos para o usuário digitar os valores do vetor b
    """
    
    st.subheader("Vetor b")
    # Cria um vetor por linha, com nome de b[0], b[1], etc...
    b = [st.number_input(f"b[{i}]", key=f"b{i}") for i in range(n)]
    return np.array(b, dtype=float)



def entrada_parametros():
    """
    Cria dois campos, com a precisão padrão em 0,0100 e formato com 4 casas decimais e
    o número maximo de iterações. 
    Essa função só é utilizada pelos métodos de Jacobi e Gauss-Seidel

    """
    
    tol = st.number_input("Tolerância", value=1e-2, format="%.4f")
    max_iter = st.number_input("Máx Iterações", value=100)  # nos testes tive que colocar máximo 100 pq gerou loop
    return tol, max_iter



def escolher_metodo():
    """
    Cria uma caixa de seleção/menu para o usuário escolher o método que quiser
    """
    
    return st.selectbox("Método", ["Gauss", "Jacobi", "Gauss-Seidel"])


# =========================
# Gráfico
# =========================


def plotar_convergencia(erros):
    """
    Função que gera o gráfico de erros de cara iteração, utiliza da biblioteca matplotlib.
    O eixo Y usa escala log para visualizar a queda de erro ao longo do processo.
    Essa função também só é utilizada pelos métodos de Jacobi e Gauss-Seidel.
    """
    
    fig, ax = plt.subplots()    # Cria a figura e eixos do matplotlib
    ax.plot(erros, marker='o')  # em cada erro no gráfico marca com o 'o' 
    # Eixos:
    ax.set_xlabel("Iterações")
    ax.set_ylabel("Erro")
    # Título:
    ax.set_title("Convergência do Método")
    # Grade atrás 
    ax.grid(True)
    ax.set_yscale('log')    # Defini como log para aparecer melhor no gráfico
    st.pyplot(fig)


# =========================
# Tabela de Iterações
# =========================


def mostrar_tabela_iteracoes(historico, erros):
    """
    Monta e exibe a tabela com o histórico de iteração, o número da iteração, o
    erro calculado e valores de x naquele momento. 
    Essa função utiliza a biblioteca Pandas.
    Também só é utilizada pelos métodos de Jacobi e Gauss-Seidel.
    """
    
    dados = []
    # Junta o histórico e erros em pares:
    for i, (x, erro) in enumerate(zip(historico, erros)):
        # Cria as linhas da tabela:
        linha = {"Iteração": i + 1, "Erro": erro}
        for j, val in enumerate(x):  
            linha[f"x{j+1}"] = val
        dados.append(linha)

    # Transforma a lista em tabela:
    df = pd.DataFrame(dados)
    st.subheader("Tabela de Iterações")
    st.dataframe(df)


# =========================
# Lógica
# =========================


def resolver_sistema(A, b, metodo, tol, max_iter):
    """
    Função que chama os métodos e parâmetros de suas determinadas classes e trata de 
    chamar as exceções com avisos.
    """
    
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
    """
    Exibe a solução do sistema
    """
    
    x = np.array(x).flatten()

    st.success("Solução:")
    for i, valor in enumerate(x):
        st.write(f"x{i+1} = {valor:.4f}")

    if iteracoes is not None:
        st.info(f"Número de iterações: {iteracoes}")



def mostrar_erro(e):
    """
    Mostra o erro encontrado
    """
    
    st.error(str(e))


# =========================
# Main
# =========================

def main():
    """
    Função principal, define a ordem que os elementos aparecem na tela.
    """
    
    titulo()

    n = entrada_tamanho()
    A = entrada_matriz(n)
    b = entrada_vetor(n)

    tol, max_iter = entrada_parametros()  # Tolerância de iterações
    metodo = escolher_metodo()            # Seleção do método

    if st.button("Resolver"):
        try:
            if A.size == 0 or b.size == 0:
                st.error("Preencha todos os valores da matriz e do vetor.")
                return

            x, it, erros, historico = resolver_sistema(A, b, metodo, tol, max_iter)

            mostrar_resultado(x, it)

            if erros is not None:
                plotar_convergencia(erros) # Gráfico

            if historico is not None:
                mostrar_tabela_iteracoes(historico, erros)

        except Exception as e:
            mostrar_erro(e)   # Mostra qualquer erro inesperado


if __name__ == "__main__":
    main()