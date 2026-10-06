"""
Questão 2 — Eliminação Gaussiana com pivoteamento parcial com escala,
Decomposição P^T LU, Decomposição de Cholesky e substituições
(progressiva / regressiva) para resolver Ax = b; experimento de tempos.

Referências (Burden, Faires & Burden, Análise Numérica, 10ª ed.):
  * Algoritmo 6.1 (p. 403) — eliminação de Gauss + substituição regressiva
  * Algoritmo 6.3 (p. 417) — pivoteamento parcial com escala
  * Teorema 6.19 / Alg. 6.4 (pp. 448–450) e p. 452 — fatoração LU e A = P^T L U
  * Corolário 6.28 / Algoritmo 6.6 (pp. 463–464) — fatoração de Cholesky A = L L^T

Restrição da lista: "sem bibliotecas numéricas avançadas". O NumPy é usado
apenas como contêiner de vetores/matrizes e para operações elementares
(somas, produtos escalares, operações de linha). NÃO se usa numpy.linalg
nem scipy: todos os algoritmos estão escritos abaixo.

Uso:
    python questao2.py # roda o experimento completo (itens a, b, c)
"""

import json
import time

import numpy as np

from questao1 import gerar_matriz_pd


class MatrizSingular(Exception):
    pass


class NaoPositivaDefinida(Exception):
    pass


# =============================================================================
# Substituições (as "variações de substituição reversa")
# =============================================================================
def substituicao_progressiva(L, b, diagonal_unitaria=False):
    """Resolve L y = b, L triangular inferior (forward substitution).
    y_1 = b_1 / l_11 ; y_i = (b_i - sum_{j<i} l_ij y_j) / l_ii ."""
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        s = b[i] - L[i, :i] @ y[:i]
        y[i] = s if diagonal_unitaria else s / L[i, i]
    return y


def substituicao_regressiva(U, y):
    """Resolve U x = y, U triangular superior (back substitution, Alg. 6.1 passos 8–9).
    x_n = y_n / u_nn ; x_i = (y_i - sum_{j>i} u_ij x_j) / u_ii ."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0:
            raise MatrizSingular("pivô nulo na substituição regressiva")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def substituicao_regressiva_transposta(L, y):
    """Resolve L^T x = y usando apenas L (sem montar L^T):
    x_i = (y_i - sum_{j>i} l_ji x_j) / l_ii . (passos 9–10 após o Alg. 6.6)"""
    n = L.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L[i + 1:, i] @ x[i + 1:]) / L[i, i]
    return x


# =============================================================================
# 1) Eliminação Gaussiana com pivoteamento parcial com escala (Alg. 6.3)
# =============================================================================
def fatores_de_escala(A):
    """s_i = max_j |a_ij| (Passo 1 do Alg. 6.3)."""
    s = np.abs(A).max(axis=1)
    if np.any(s == 0):
        raise MatrizSingular("linha nula: não existe solução única")
    return s


def eliminacao_gauss_escala(A, b):
    """Resolve Ax = b por eliminação de Gauss com pivoteamento parcial com escala.
    As trocas de linha são 'simuladas' com o vetor NLINHA, como no livro."""
    a = np.array(A, dtype=float) # cópia: não altera A
    bb = np.array(b, dtype=float)
    n = a.shape[0]
    s = fatores_de_escala(a)
    nlinha = np.arange(n) # NLINHA(i) = i

    for i in range(n - 1):
        # Passo 3: menor p >= i que maximiza |a(NLINHA(p), i)| / s(NLINHA(p))
        candidatas = nlinha[i:]
        razoes = np.abs(a[candidatas, i]) / s[candidatas]
        p = i + int(np.argmax(razoes))
        if a[nlinha[p], i] == 0:
            raise MatrizSingular("não existe solução única")
        # Passo 5: troca simulada
        if p != i:
            nlinha[i], nlinha[p] = nlinha[p], nlinha[i]
        piv = nlinha[i]
        abaixo = nlinha[i + 1:]
        # Passos 6–7: m_ji = a_ji / a_ii ; E_j <- E_j - m_ji E_i
        m = a[abaixo, i] / a[piv, i]
        a[abaixo, i:] -= np.outer(m, a[piv, i:])
        bb[abaixo] -= m * bb[piv]

    if a[nlinha[n - 1], n - 1] == 0:
        raise MatrizSingular("não existe solução única")
    # Substituição regressiva na ordem NLINHA (sistema triangular superior)
    return substituicao_regressiva(a[nlinha], bb[nlinha])


# =============================================================================
# 2) Decomposição P^T L U (PA = LU <=> A = P^T L U)
# =============================================================================
def fatoracao_ptlu(A, com_escala=True):
    """Fatoração de Doolittle (Alg. 6.4, l_ii = 1) com trocas de linhas.

    Usa as fórmulas do Alg. 6.4 em forma de produto interno:
        u_ij = a_ij - sum_{k<i} l_ik u_kj (linha i de U)
        l_ji = (a_ji - sum_{k<i} l_jk u_ki) / u_ii (coluna i de L)
    Antes de dividir por u_ii, escolhe-se o pivô na coluna i pelo critério do
    pivoteamento parcial com escala (Alg. 6.3) e trocam-se as linhas.

    Devolve (LU, perm): L (diagonal unitária, abaixo da diagonal) e U (diagonal
    e acima) guardadas na mesma matriz, e o vetor de permutação perm tal que
    P[i, perm[i]] = 1, ou seja, (PA)[i] = A[perm[i]] e PA = LU."""
    a = np.array(A, dtype=float)
    n = a.shape[0]
    perm = np.arange(n)
    s = fatores_de_escala(a) if com_escala else np.ones(n)

    for i in range(n):
        # candidatos a pivô: a_ji - sum_{k<i} l_jk u_ki , j = i..n-1
        a[i:, i] -= a[i:, :i] @ a[:i, i]
        p = i + int(np.argmax(np.abs(a[i:, i]) / s[i:]))
        if a[p, i] == 0:
            raise MatrizSingular("fatoração impossível: matriz singular")
        if p != i: # troca de linhas (inclui parte já calculada de L)
            a[[i, p]] = a[[p, i]]
            perm[[i, p]] = perm[[p, i]]
            s[[i, p]] = s[[p, i]]
        # linha i de U (colunas i+1..n-1)
        a[i, i + 1:] -= a[i, :i] @ a[:i, i + 1:]
        # coluna i de L (multiplicadores)
        a[i + 1:, i] /= a[i, i]
    return a, perm


def resolver_ptlu(LU, perm, b):
    """Ax = b com A = P^T L U: L y = P b (progressiva), U x = y (regressiva)."""
    pb = np.asarray(b, dtype=float)[perm]
    y = substituicao_progressiva(LU, pb, diagonal_unitaria=True)
    return substituicao_regressiva(LU, y)


def extrair_P_L_U(LU, perm):
    n = LU.shape[0]
    L = np.tril(LU, -1) + np.eye(n)
    U = np.triu(LU)
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0
    return P, L, U


# =============================================================================
# 3) Decomposição de Cholesky A = L L^T (Alg. 6.6)
# =============================================================================
def fatoracao_cholesky(A):
    """l_ii = sqrt(a_ii - sum_{k<i} l_ik^2)
       l_ji = (a_ji - sum_{k<i} l_jk l_ik) / l_ii , j > i."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        d = A[i, i] - L[i, :i] @ L[i, :i]
        if d <= 0:
            raise NaoPositivaDefinida(f"pivô {d:.3e} <= 0 na etapa {i + 1}")
        L[i, i] = np.sqrt(d)
        L[i + 1:, i] = (A[i + 1:, i] - L[i + 1:, :i] @ L[i, :i]) / L[i, i]
    return L


def resolver_cholesky(L, b):
    """Ax = b com A = L L^T: L y = b (progressiva), L^T x = y (regressiva)."""
    y = substituicao_progressiva(L, b)
    return substituicao_regressiva_transposta(L, y)


# =============================================================================
# Experimento
# =============================================================================
def vetor_b_nao_nulo(n, rng):
    while True:
        b = rng.uniform(-10, 10, size=n)
        if np.any(b != 0):
            return b


def residuo(A, x, b):
    return float(np.max(np.abs(A @ x - b)))


def _mediana(v):
    v = sorted(v)
    return v[len(v) // 2]


def item_a(ns, reps, rng, repeticoes=5):
    """10 matrizes PD distintas e 10 vetores b: um sistema A_k x = b_k por par.
    Cada A_k é diferente, então os três métodos têm de processar A_k do zero.
    O tempo total (10 sistemas) é medido `repeticoes` vezes e guarda-se a mediana,
    para reduzir o ruído do sistema operacional."""
    tempos = {"Gauss": [], "PtLU": [], "Cholesky": []}
    for n in ns:
        As = [gerar_matriz_pd(n, rng) for _ in range(reps)]
        bs = [vetor_b_nao_nulo(n, rng) for _ in range(reps)]
        medidas = {"Gauss": [], "PtLU": [], "Cholesky": []}
        res_max = 0.0
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            xs1 = [eliminacao_gauss_escala(A, b) for A, b in zip(As, bs)]
            t1 = time.perf_counter()
            xs2 = []
            for A, b in zip(As, bs):
                LU, perm = fatoracao_ptlu(A) # cada A é nova: precisa fatorar
                xs2.append(resolver_ptlu(LU, perm, b))
            t2 = time.perf_counter()
            xs3 = []
            for A, b in zip(As, bs):
                L = fatoracao_cholesky(A)
                xs3.append(resolver_cholesky(L, b))
            t3 = time.perf_counter()
            medidas["Gauss"].append(t1 - t0)
            medidas["PtLU"].append(t2 - t1)
            medidas["Cholesky"].append(t3 - t2)
        for A, b, x1, x2, x3 in zip(As, bs, xs1, xs2, xs3):
            res_max = max(res_max, residuo(A, x1, b), residuo(A, x2, b), residuo(A, x3, b))
        t = {k: _mediana(v) for k, v in medidas.items()}
        for k in tempos:
            tempos[k].append(t[k])
        print(f"(a) n={n:4d} Gauss={t['Gauss']:.4f}s PtLU={t['PtLU']:.4f}s "
              f"Cholesky={t['Cholesky']:.4f}s max||Ax-b||_inf={res_max:.2e}")
    return tempos


def item_b(ns, reps, rng, repeticoes=5):
    """1 matriz PD e 10 vetores b. Gauss refaz a eliminação a cada b;
    P^T LU e Cholesky fatoram UMA vez e reaproveitam a fatoração
    (só as substituições, O(n^2), são repetidas para cada b)."""
    tempos = {"Gauss": [], "PtLU": [], "Cholesky": []}
    for n in ns:
        A = gerar_matriz_pd(n, rng)
        bs = [vetor_b_nao_nulo(n, rng) for _ in range(reps)]
        medidas = {"Gauss": [], "PtLU": [], "Cholesky": []}
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            xs1 = [eliminacao_gauss_escala(A, b) for b in bs]
            t1 = time.perf_counter()

            LU, perm = fatoracao_ptlu(A) # fatora uma única vez
            xs2 = [resolver_ptlu(LU, perm, b) for b in bs]
            t2 = time.perf_counter()

            L = fatoracao_cholesky(A) # fatora uma única vez
            xs3 = [resolver_cholesky(L, b) for b in bs]
            t3 = time.perf_counter()
            medidas["Gauss"].append(t1 - t0)
            medidas["PtLU"].append(t2 - t1)
            medidas["Cholesky"].append(t3 - t2)

        res_max = 0.0
        for b, x1, x2, x3 in zip(bs, xs1, xs2, xs3):
            res_max = max(res_max, residuo(A, x1, b), residuo(A, x2, b), residuo(A, x3, b))
        t = {k: _mediana(v) for k, v in medidas.items()}
        for k in tempos:
            tempos[k].append(t[k])
        print(f"(b) n={n:4d} Gauss={t['Gauss']:.4f}s PtLU={t['PtLU']:.4f}s "
              f"Cholesky={t['Cholesky']:.4f}s max||Ax-b||_inf={res_max:.2e}")
    return tempos


def plotar(ns, ta, tb, arquivo="q2_tempos.png", log=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = {"Gauss": "tab:red", "PtLU": "tab:blue", "Cholesky": "tab:green"}
    rotulo = {"Gauss": "Gauss c/ pivot. parcial c/ escala",
              "PtLU": r"$P^TLU$", "Cholesky": "Cholesky"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k in ["Gauss", "PtLU", "Cholesky"]:
        ax.plot(ns, ta[k], "-o", color=cores[k], label=f"(a) {rotulo[k]}")
        ax.plot(ns, tb[k], "--s", color=cores[k], label=f"(b) {rotulo[k]}")
    ax.set_xlabel("tamanho da matriz $n$")
    ax.set_ylabel("tempo total de execução (s)")
    ax.set_title("Tempo total para resolver 10 sistemas $Ax=b$")
    ax.set_xticks(ns)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    plt.close(fig)


def demonstracao_pequena():
    """Mostra os três métodos num sistema 4x4 e confere que A = P^T L U e A = L L^T."""
    rng = np.random.default_rng(1)
    A = gerar_matriz_pd(4, rng)
    b = vetor_b_nao_nulo(4, rng)
    np.set_printoptions(precision=5, suppress=True)
    print("A =\n", A, "\nb =", b)
    print("Gauss (escala):", eliminacao_gauss_escala(A, b))
    LU, perm = fatoracao_ptlu(A)
    P, L, U = extrair_P_L_U(LU, perm)
    print("P^T LU :", resolver_ptlu(LU, perm, b),
          " ||P^T L U - A||_max =", np.abs(P.T @ L @ U - A).max())
    Lc = fatoracao_cholesky(A)
    print("Cholesky :", resolver_cholesky(Lc, b),
          " ||L L^T - A||_max =", np.abs(Lc @ Lc.T - A).max())
    print()


if __name__ == "__main__":
    demonstracao_pequena()

    ns = [100, 200, 300, 400, 500]
    reps = 10
    rng = np.random.default_rng(12345)

    # aquecimento (evita que a 1ª medição inclua custos de inicialização)
    _A = gerar_matriz_pd(50, rng); _b = vetor_b_nao_nulo(50, rng)
    eliminacao_gauss_escala(_A, _b); resolver_ptlu(*fatoracao_ptlu(_A), _b)
    resolver_cholesky(fatoracao_cholesky(_A), _b)

    ta = item_a(ns, reps, rng)
    print()
    tb = item_b(ns, reps, rng)

    with open("q2_resultados.json", "w") as f:
        json.dump({"n": ns, "a": ta, "b": tb}, f, indent=2)

    plotar(ns, ta, tb, "q2_tempos.png")
    plotar(ns, ta, tb, "q2_tempos_log.png", log=True)

    for item, t in (("a", ta), ("b", tb)):
        totais = {k: sum(v) for k, v in t.items()}
        venc = [min(t, key=lambda k: t[k][i]) for i in range(len(ns))]
        print(f"\nItem ({item}) — método mais rápido em cada n: "
              + ", ".join(f"n={n}: {m}" for n, m in zip(ns, venc)))
        print(f" tempos somados: " + ", ".join(f"{k}={v:.3f}s" for k, v in totais.items()))
