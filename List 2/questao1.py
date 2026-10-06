"""
Questão 1 — Gerador aleatório de matrizes positivas definidas.

Ideia (Teorema do enunciado + Burden, Def. 6.20 e Def. 6.22):
    Se A é simétrica, estritamente diagonal dominante e tem diagonal
    não-negativa, então A é positiva definida.

Algoritmo:
    1. Sorteia os elementos abaixo da diagonal a_ij ~ U(-1, 1), i > j.
    2. Espelha: a_ji = a_ij (garante A = A^T).
    3. Para cada linha i, define
          a_ii = sum_{j != i} |a_ij| + delta_i , delta_i ~ U(1, 2) (> 0)
       o que garante |a_ii| > sum_{j != i} |a_ij| (dominância estrita)
       e a_ii > 0 (diagonal positiva).

Só usamos o NumPy como "contêiner" de vetores/matrizes e para sortear números.
Nenhuma rotina de álgebra linear pronta (numpy.linalg, scipy) é usada.
"""

import numpy as np


def gerar_matriz_pd(n, rng=None, faixa=1.0, margem=(1.0, 2.0)):
    """Gera uma matriz n x n simétrica, estritamente diagonal dominante,
    com diagonal positiva (logo, positiva definida)."""
    if rng is None:
        rng = np.random.default_rng()
    A = np.zeros((n, n))
    # 1) parte estritamente inferior aleatória
    for i in range(1, n):
        A[i, :i] = rng.uniform(-faixa, faixa, size=i)
    # 2) simetria: A = L + L^T
    A = A + A.T
    # 3) diagonal: soma dos módulos fora da diagonal + folga positiva
    soma_fora = np.abs(A).sum(axis=1) # diagonal ainda é zero aqui
    delta = rng.uniform(margem[0], margem[1], size=n)
    for i in range(n):
        A[i, i] = soma_fora[i] + delta[i]
    return A


# ---------------------------------------------------------------- verificações
def eh_simetrica(A, tol=0.0):
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j] - A[j, i]) > tol:
                return False
    return True


def eh_estritamente_diagonal_dominante(A):
    n = A.shape[0]
    for i in range(n):
        soma = 0.0
        for j in range(n):
            if j != i:
                soma += abs(A[i, j])
        if not abs(A[i, i]) > soma:
            return False
    return True


def diagonal_nao_negativa(A):
    return all(A[i, i] >= 0 for i in range(A.shape[0]))


def menor_folga(A):
    """min_i ( a_ii - sum_{j!=i} |a_ij| ): quanto maior que zero, mais 'folgada'
    é a dominância diagonal. Pela demonstração, x^T A x >= menor_folga * ||x||^2."""
    n = A.shape[0]
    return min(A[i, i] - (np.abs(A[i]).sum() - abs(A[i, i])) for i in range(n))


def teste_empirico_forma_quadratica(A, amostras=20000, rng=None):
    """Sorteia vetores x != 0 e devolve o menor valor de x^T A x / x^T x."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = A.shape[0]
    menor = np.inf
    for _ in range(amostras):
        x = rng.normal(size=n)
        q = x @ (A @ x)
        menor = min(menor, q / (x @ x))
    return menor


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    for k, n in enumerate([3, 4, 5], start=1):
        A = gerar_matriz_pd(n, rng)
        print(f"===== Exemplo {k}: n = {n} =====")
        print(A)
        print(f" simétrica? {eh_simetrica(A)}")
        print(f" estritamente diag. dominante? {eh_estritamente_diagonal_dominante(A)}")
        print(f" diagonal não-negativa? {diagonal_nao_negativa(A)}")
        print(f" folga mínima min_i(a_ii - R_i) = {menor_folga(A):.4f}")
        print(f" min x^T A x / x^T x (20000 x aleatórios) = "
              f"{teste_empirico_forma_quadratica(A, rng=rng):.4f} (> 0)")
        print()
