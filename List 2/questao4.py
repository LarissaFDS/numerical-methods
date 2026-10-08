"""
Questão 4 (teórica) — Matriz de Gram A = B^T B.

A demonstração está no relatório LaTeX. Este script apenas ILUSTRA numericamente
as duas propriedades para matrizes B aleatórias de vários formatos:
  (a) A = A^T
  (b) x^T A x = ||Bx||_2^2 >= 0  para todo x
Também mostra o caso m < n, em que A é apenas semidefinida (existe x != 0 com
Bx = 0, logo x^T A x = 0), e usa o Cholesky com teste da questão 3 para mostrar
que, quando B tem colunas linearmente independentes, A é de fato definida positiva.
"""

import numpy as np

from questao3 import cholesky_com_teste


def gram(B):
    m, n = B.shape
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = B[:, i] @ B[:, j]          # a_ij = <coluna i, coluna j>
    return A


def vetor_no_nucleo(B):
    """Para B (m x n) com m < n, encontra x != 0 com Bx = 0 por eliminação
    de Gauss (forma escalonada) — sem bibliotecas de álgebra linear."""
    a = np.array(B, dtype=float)
    m, n = a.shape
    pivos, lin = [], 0
    for col in range(n):
        if lin == m:
            break
        p = lin + int(np.argmax(np.abs(a[lin:, col])))
        if abs(a[p, col]) < 1e-12:
            continue
        a[[lin, p]] = a[[p, lin]]
        a[lin] /= a[lin, col]
        for r in range(m):
            if r != lin:
                a[r] -= a[r, col] * a[lin]
        pivos.append(col)
        lin += 1
    livre = next(c for c in range(n) if c not in pivos)
    x = np.zeros(n)
    x[livre] = 1.0
    for r, c in enumerate(pivos):
        x[c] = -a[r, livre]
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    for (m, n) in [(5, 3), (3, 3), (2, 4)]:
        B = rng.normal(size=(m, n))
        A = gram(B)
        sim = np.abs(A - A.T).max()
        menor = min((x @ A @ x) for x in rng.normal(size=(20000, n)))
        print(f"B {m}x{n}:  max|A - A^T| = {sim:.1e}   "
              f"min x^T A x (20000 x) = {menor:.4e}   "
              f"Cholesky diz PD? {cholesky_com_teste(A)[0]}")
        x = rng.normal(size=n)
        print(f"          conferindo x^T A x = ||Bx||^2: {x @ A @ x:.6f} = {(B @ x) @ (B @ x):.6f}")
        if m < n:
            z = vetor_no_nucleo(B)
            print(f"          m < n: z = {np.round(z, 4)} tem Bz = {np.round(B @ z, 12)} "
                  f"e z^T A z = {z @ A @ z:.1e} (zero, a menos de arredondamento) -> A é só semidefinida")
