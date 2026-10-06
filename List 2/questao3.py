"""
Questão 3 — Temperatura de equilíbrio numa placa circular (Anton & Rorres, §10.11).

Malha: 4 pontos interiores t1 (sup. esq.), t2 (sup. dir.), t3 (inf. esq.),
t4 (inf. dir.) e 8 pontos de contorno: temperatura 0 na metade esquerda da
circunferência e 1 na metade direita.

Propriedade discreta do valor médio: a temperatura num ponto interior é a média
das temperaturas nos 4 vizinhos (cima, baixo, esquerda, direita).
Isso dá t = M t + b -> (I - M) t = b -> A t = b.

Item (b): Cholesky "adaptado" — além de fatorar, decide se A é positiva definida
(Burden, Teorema 6.26 / Corolário 6.28: A simétrica é PD <=> a fatoração
A = L L^T existe com l_ii > 0, i.e., todos os radicandos são > 0).
"""

import numpy as np

from questao2 import (resolver_ptlu, fatoracao_ptlu, substituicao_progressiva,
                      substituicao_regressiva_transposta)


# ----------------------------------------------------------------------------- (a)
def montar_sistema():
    """Monta M e b a partir da vizinhança de cada ponto interior.
    Vizinhos do tipo ('t', k) são incógnitas; ('c', valor) são pontos de contorno."""
    vizinhos = {
        # ponto: cima baixo esquerda direita
        0: [("c", 0.0), ("t", 2), ("c", 0.0), ("t", 1)], # t1
        1: [("c", 1.0), ("t", 3), ("t", 0), ("c", 1.0)], # t2
        2: [("t", 0), ("c", 0.0), ("c", 0.0), ("t", 3)], # t3
        3: [("t", 1), ("c", 1.0), ("t", 2), ("c", 1.0)], # t4
    }
    M = np.zeros((4, 4))
    b = np.zeros(4)
    for i, viz in vizinhos.items():
        for tipo, v in viz:
            if tipo == "t":
                M[i, v] += 0.25
            else:
                b[i] += 0.25 * v
    return M, b


# ----------------------------------------------------------------------------- (b)
def cholesky_com_teste(A, tol_sim=1e-12):
    """Algoritmo 6.6 (Cholesky) adaptado.
    Devolve (eh_pd, L). eh_pd = False se A não for simétrica ou se algum
    radicando a_ii - sum_k l_ik^2 for <= 0 (nesse caso L = None)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    # 1) PD (no sentido do Burden, Def. 6.22) exige simetria
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j] - A[j, i]) > tol_sim:
                return False, None
    # 2) fatoração; falha <=> A não é PD
    L = np.zeros((n, n))
    for i in range(n):
        d = A[i, i] - L[i, :i] @ L[i, :i]
        if d <= 0:
            return False, None
        L[i, i] = np.sqrt(d)
        L[i + 1:, i] = (A[i + 1:, i] - L[i + 1:, :i] @ L[i, :i]) / L[i, i]
    return True, L


def resolver(A, b):
    eh_pd, L = cholesky_com_teste(A)
    if eh_pd:
        y = substituicao_progressiva(L, b)
        t = substituicao_regressiva_transposta(L, y)
        return t, "Cholesky", L
    LU, perm = fatoracao_ptlu(A)
    return resolver_ptlu(LU, perm, b), "P^T LU", None


# ----------------------------------------------------------------------------- (c)
def desenhar(t, arquivo="q3_placa.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    h = 1.0 # espaçamento da malha
    c = h / 2 # pontos interiores em (±h/2, ±h/2)
    yb = c + h # contorno a uma distância h dos interiores
    R = np.hypot(c, yb) # raio do disco que passa pelos 8 pontos

    interiores = {"$t_1$": (-c, c), "$t_2$": (c, c), "$t_3$": (-c, -c), "$t_4$": (c, -c)}
    contorno = [(-c, yb, 0), (c, yb, 1), (-yb, c, 0), (yb, c, 1),
                (-yb, -c, 0), (yb, -c, 1), (-c, -yb, 0), (c, -yb, 1)]

    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.add_patch(Circle((0, 0), R, facecolor="#ececec", edgecolor="0.25", lw=1.5, zorder=0))
    # metades do contorno com a cor da temperatura
    th = np.linspace(np.pi / 2, 3 * np.pi / 2, 200)
    ax.plot(R * np.cos(th), R * np.sin(th), color=cmap(0.0), lw=4, zorder=1)
    ax.plot(-R * np.cos(th), R * np.sin(th), color=cmap(1.0), lw=4, zorder=1)
    # linhas da malha
    for v in (-c, c):
        ax.plot([v, v], [-yb, yb], color="0.45", lw=1, zorder=1)
        ax.plot([-yb, yb], [v, v], color="0.45", lw=1, zorder=1)

    for x, y, T in contorno:
        ax.scatter(x, y, s=170, c=[T], cmap=cmap, vmin=0, vmax=1,
                   edgecolors="k", zorder=3)
        dx, dy = 0.22 * np.sign(x) * (abs(x) > c), 0.22 * np.sign(y) * (abs(y) > c)
        ax.text(x + dx, y + dy, f"{T:g}", ha="center", va="center", fontsize=12)

    for (nome, (x, y)), T in zip(interiores.items(), t):
        sc = ax.scatter(x, y, s=420, c=[T], cmap=cmap, vmin=0, vmax=1,
                        edgecolors="k", zorder=3)
        ax.text(x, y + 0.28, f"{nome} = {T:.4f}", ha="center", fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("temperatura")
    ax.set_aspect("equal")
    ax.set_xlim(-R - 0.4, R + 0.4)
    ax.set_ylim(-R - 0.4, R + 0.4)
    ax.axis("off")
    ax.set_title("Placa circular: temperaturas na malha")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    M, b = montar_sistema()
    A = np.eye(4) - M
    print("M =\n", M, "\nb =", b, "\nA = I - M =\n", A)

    eh_pd, L = cholesky_com_teste(A)
    print("\nA é positiva definida?", eh_pd)
    t, metodo, L = resolver(A, b)
    if L is not None:
        print("L (Cholesky) =\n", L)
        print("||L L^T - A||_max =", np.abs(L @ L.T - A).max())
    print(f"\nTemperaturas (resolvido por {metodo}):")
    for k, v in enumerate(t, start=1):
        print(f" t{k} = {v:.6f}")
    print("Verificação: ||t - (M t + b)||_inf =", np.abs(t - (M @ t + b)).max())

    # contra-exemplo: o teste detecta uma matriz simétrica NÃO PD
    B = np.array([[1.0, 2.0], [2.0, 1.0]])
    print("\nTeste do detector com B = [[1,2],[2,1]] (autovalores 3 e -1):",
          cholesky_com_teste(B)[0])

    desenhar(t)
    print("\nFigura salva em q3_placa.png")
