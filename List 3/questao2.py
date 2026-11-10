"""
Questão 2 — Calibração de reservatório: mínimos quadrados, Lagrange e Neville.

Referências (Burden, Faires & Burden, Análise Numérica, 10ª ed.):
  * Seção 8.1, equações normais (8.3), p. 561 — mínimos quadrados discretos
  * Teorema 3.2, Eq. (3.1)–(3.2), p. 120 — polinômio interpolador de Lagrange
  * Teorema 3.3, Eq. (3.3), p. 122 — erro da interpolação
  * Teorema 3.5 (p. 130) e Algoritmo 3.1 (p. 134) — método de Neville

Sem bibliotecas numéricas: aritmética exata com fractions.Fraction (os dados
são múltiplos de 0,5, então tudo é representável exatamente) e eliminação de
Gauss implementada abaixo. O matplotlib é usado só para o gráfico.
"""

from fractions import Fraction as Fr

H = [Fr(0), Fr(1, 2), Fr(1), Fr(3, 2), Fr(2)]          # altura (m)
V = [Fr(0), Fr(620), Fr(1480), Fr(2570), Fr(4020)]     # volume (L)
H_ALVO = Fr(5, 4)                                      # 1,25 m


def fmt(q, casas=6):
    return f"{float(q):.{casas}f}".rstrip("0").rstrip(".")


# ----------------------------------------------------------------------------- (a)
def equacoes_normais(x, y, n):
    """Sistema das equações normais (8.3) para p_n(x) = a0 + a1 x + ... + an x^n:
         sum_k a_k * (sum_i x_i^(j+k)) = sum_i y_i x_i^j ,   j = 0..n."""
    S = [sum(xi ** p for xi in x) for p in range(2 * n + 1)]
    A = [[S[j + k] for k in range(n + 1)] for j in range(n + 1)]
    b = [sum(yi * xi ** j for xi, yi in zip(x, y)) for j in range(n + 1)]
    return A, b


def eliminacao_gauss(A, b):
    """Eliminação de Gauss com pivoteamento parcial + substituição regressiva
    (Algoritmos 6.1/6.2). Funciona com Fraction (exato) ou float."""
    n = len(A)
    M = [list(linha) + [bi] for linha, bi in zip(A, b)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        if M[p][i] == 0:
            raise ValueError("sistema singular")
        M[i], M[p] = M[p], M[i]
        for r in range(i + 1, n):
            m = M[r][i] / M[i][i]
            M[r] = [a - m * c for a, c in zip(M[r], M[i])]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def avaliar_poli(coef, t):
    """Horner: coef = [a0, a1, ..., an]."""
    r = 0
    for a in reversed(coef):
        r = r * t + a
    return r


# ----------------------------------------------------------------------------- (b)
def base_lagrange(x, k, t):
    """L_{n,k}(t) = prod_{i != k} (t - x_i) / (x_k - x_i)   (Eq. 3.2)."""
    L = Fr(1)
    for i, xi in enumerate(x):
        if i != k:
            L *= (t - xi) / (x[k] - xi)
    return L


def lagrange(x, y, t):
    """P(t) = sum_k y_k L_{n,k}(t)   (Eq. 3.1)."""
    return sum(yk * base_lagrange(x, k, t) for k, yk in enumerate(y))


def mult_poli(p, q):
    r = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def coeficientes_lagrange(x, y):
    """Expande P(x) = sum_k y_k L_k(x) na base de monômios (só para exibição)."""
    n = len(x)
    total = [Fr(0)] * n
    for k in range(n):
        Lk = [Fr(1)]
        for i in range(n):
            if i != k:
                Lk = mult_poli(Lk, [-x[i] / (x[k] - x[i]), 1 / (x[k] - x[i])])
        for g in range(n):
            total[g] += y[k] * Lk[g]
    return total


# ----------------------------------------------------------------------------- (c)
def neville(x, y, t):
    """Algoritmo 3.1. Q[i][0] = y_i e
        Q[i][j] = ((t - x_{i-j}) Q[i][j-1] - (t - x_i) Q[i-1][j-1]) / (x_i - x_{i-j}),
    Q[i][j] = valor em t do polinômio que interpola x_{i-j}, ..., x_i (Teorema 3.5)."""
    n = len(x)
    Q = [[None] * n for _ in range(n)]
    for i in range(n):
        Q[i][0] = y[i]
    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i][j] = ((t - x[i - j]) * Q[i][j - 1] - (t - x[i]) * Q[i - 1][j - 1]) / (x[i] - x[i - j])
    return Q[n - 1][n - 1], Q


def termo_poli(coef, var="h"):
    partes = []
    for g, a in enumerate(coef):
        if a == 0:
            continue
        mon = "" if g == 0 else (var if g == 1 else f"{var}^{g}")
        partes.append(f"{'+' if a > 0 else '-'} {fmt(abs(a))}{mon}")
    return " ".join(partes).lstrip("+ ")


def plotar(c2, c4, v2, v4, arquivo="q2_ajuste.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = [i * 2 / 400 for i in range(401)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(hs, [float(avaliar_poli(c2, Fr(h).limit_denominator(10**6))) for h in hs],
            color="tab:blue", label=r"mínimos quadrados $p_2(h)$")
    ax.plot(hs, [float(avaliar_poli(c4, Fr(h).limit_denominator(10**6))) for h in hs],
            color="tab:orange", ls="--", label=r"Lagrange $p_4(h)$")
    ax.scatter([float(h) for h in H], [float(v) for v in V], color="k", zorder=3, label="dados medidos")
    ax.scatter([1.25], [float(v2)], color="tab:blue", marker="D", s=60, zorder=4,
               label=rf"$p_2(1{{,}}25)={float(v2):.2f}$ L")
    ax.scatter([1.25], [float(v4)], color="tab:orange", marker="s", s=60, zorder=4,
               label=rf"$p_4(1{{,}}25)={float(v4):.2f}$ L")
    ax.axvline(1.25, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("altura $h$ (m)")
    ax.set_ylabel("volume $V$ (L)")
    ax.set_title("Calibração do reservatório")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print("Dados:", [(fmt(h), fmt(v)) for h, v in zip(H, V)])

    # (a) -------------------------------------------------------------------
    A, b = equacoes_normais(H, V, 2)
    print("\n(a) Equações normais (8.3) para p2(h) = a0 + a1 h + a2 h^2:")
    for linha, bi in zip(A, b):
        print("    [" + "  ".join(f"{fmt(a):>7}" for a in linha) + f" ] [a]  =  {fmt(bi)}")
    a = eliminacao_gauss(A, b)
    print(f"    a0 = {fmt(a[0])},  a1 = {fmt(a[1])},  a2 = {fmt(a[2])}")
    print(f"    p2(h) = {termo_poli(a)}")
    res = [v - avaliar_poli(a, h) for h, v in zip(H, V)]
    print(f"    resíduos V_i - p2(h_i) = {[fmt(r) for r in res]};  E = soma dos quadrados = {fmt(sum(r*r for r in res))}")
    v2 = avaliar_poli(a, H_ALVO)
    print(f"    V(1,25) ~ p2(1,25) = {fmt(v2)} L")

    # (b) -------------------------------------------------------------------
    print("\n(b) Lagrange com os 5 pontos:")
    for k in range(5):
        print(f"    L_{k}(1,25) = {base_lagrange(H, k, H_ALVO)}")
    v4 = lagrange(H, V, H_ALVO)
    c4 = coeficientes_lagrange(H, V)
    print(f"    p4(h) = {termo_poli(c4)}")
    print(f"    conferência p4(h_i) = V_i: {[fmt(avaliar_poli(c4, h)) for h in H]}")
    print(f"    V(1,25) ~ p4(1,25) = {fmt(v4)} L  (= {v4})")

    # (c) -------------------------------------------------------------------
    vn, Q = neville(H, V, H_ALVO)
    print("\n(c) Tabela de Neville em h = 1,25 (linha i, coluna j = Q_{i,j}):")
    print("    " + f"{'x_i':>6}" + "".join(f"{'Q_i,' + str(j):>14}" for j in range(5)))
    for i in range(5):
        print("    " + f"{fmt(H[i]):>6}" + "".join(f"{fmt(Q[i][j], 6):>14}" for j in range(i + 1)))
    print(f"    Valor final Q_4,4 = {fmt(vn)} L  (igual a p4(1,25)? {vn == v4})")

    print(f"\nDiferença entre as estimativas: p2 - p4 = {fmt(v2 - v4)} L")
    plotar(a, c4, v2, v4)
    print("Figura salva em q2_ajuste.png")
