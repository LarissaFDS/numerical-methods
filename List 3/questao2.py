"""
Questão 2 — Calibração de reservatório: mínimos quadrados, Lagrange e Neville.

Modelagem e métodos numéricos aplicados:
  * Ajuste de curvas por Mínimos Quadrados Discretos:
      Ajuste do polinômio quadrático p2(h) = a0 + a1*h + a2*h^2 aos dados experimentais.
      Montagem das equações normais (sistema linear 3x3) e resolução exata via eliminação de Gauss.
  * Interpolação polinomial de Lagrange:
      Construção do polinômio p4(h) interpolador de grau 4 utilizando os 5 pontos da tabela.
      Avaliação direta de p4(1.25) através das funções base L_{4,k}(1.25).
  * Método de Neville:
      Algoritmo recursivo para interpolação sem construção explícita dos coeficientes do polinômio.
      Geração e exibição da tabela triangular de diferenças divididas/aproximações Q_{i,j}.
      Comprovação de equivalência: Q_{4,4} coincide exatamente com p4(1.25).
  * Comparação entre o ajuste por mínimos quadrados (que atenua o ruído experimental)
    e a interpolação exata (que passa obrigatoriamente por todos os pontos com ruído).

Referências (Burden, Faires & Burden, Análise Numérica, 10ª ed.):
  * Seção 8.1, equações normais (8.3), p. 561 — Mínimos quadrados discretos
  * Teorema 3.2, Eq. (3.1)–(3.2), p. 120 — Polinômio interpolador de Lagrange
  * Teorema 3.3, Eq. (3.3), p. 122 — Erro da interpolação polinomial
  * Teorema 3.5 (p. 130) e Algoritmo 3.1 (p. 134) — Método de Neville

Implementação sem bibliotecas numéricas avançadas (Python puro com fractions.Fraction).
"""

from fractions import Fraction as Fr
import os

# Dados experimentais fornecidos na tabela (altura em metros, volume em litros)
H = [Fr(0), Fr(1, 2), Fr(1), Fr(3, 2), Fr(2)]          # h (m)
V = [Fr(0), Fr(620), Fr(1480), Fr(2570), Fr(4020)]     # V (L)
H_ALVO = Fr(5, 4)                                      # h = 1.25 m (ponto de consulta)


def fmt(q, casas=6):
    """Formata um número (Fraction ou float) para representação decimal limpa.

    Args:
        q (Fraction or float or int): Valor a ser formatado.
        casas (int): Número máximo de casas decimais. Default: 6.

    Returns:
        str: String formatada sem zeros à direita desnecessários.
    """
    return f"{float(q):.{casas}f}".rstrip("0").rstrip(".")


# ----------------------------------------------------------------------------- (a)
def equacoes_normais(x, y, n):
    """Monta o sistema das equações normais para ajuste polinomial por mínimos quadrados.

    Para um polinômio de grau n:
        p_n(x) = a0 + a1*x + ... + an*x^n,
    o critério de mínimos quadrados minimiza E(a0, ..., an) = sum_{i=1}^m (y_i - p_n(x_i))^2.
    As equações normais (Burden, Eq. 8.3) formam o sistema linear (n+1) x (n+1):
        sum_{k=0}^n a_k * (sum_{i=1}^m x_i^(j+k)) = sum_{i=1}^m y_i * x_i^j,   para j = 0, ..., n.

    Args:
        x (list[Fraction]): Abscissas dos pontos amostrados (alturas).
        y (list[Fraction]): Ordenadas dos pontos amostrados (volumes).
        n (int): Grau do polinômio de ajuste (n = 2 para p2(h)).

    Returns:
        tuple[list[list[Fraction]], list[Fraction]]:
            - A: Matriz simétrica dos coeficientes (n+1) x (n+1).
            - b: Vetor dos termos independentes de tamanho (n+1).
    """
    S = [sum(xi ** p for xi in x) for p in range(2 * n + 1)]
    A = [[S[j + k] for k in range(n + 1)] for j in range(n + 1)]
    b = [sum(yi * xi ** j for xi, yi in zip(x, y)) for j in range(n + 1)]
    return A, b


def eliminacao_gauss(A, b):
    """Resolve o sistema linear A x = b via Eliminação de Gauss com pivoteamento parcial.

    Aplica os Algoritmos 6.1 e 6.2 de Burden & Faires com substituição regressiva.
    Opera de forma exata quando alimentado com fractions.Fraction.

    Args:
        A (list[list[Fraction]]): Matriz dos coeficientes n x n.
        b (list[Fraction]): Vetor dos termos independentes de tamanho n.

    Returns:
        list[Fraction]: Vetor solução x de tamanho n.

    Raises:
        ValueError: Se o sistema for singular (pivô nulo após busca de pivô).
    """
    n = len(A)
    M = [list(linha) + [bi] for linha, bi in zip(A, b)]
    for i in range(n):
        # Pivoteamento parcial: busca linha com maior pivô em módulo
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        if M[p][i] == 0:
            raise ValueError("Sistema linear é singular (não possui solução única).")
        if p != i:
            M[i], M[p] = M[p], M[i]
        # Eliminação para frente
        for r in range(i + 1, n):
            m = M[r][i] / M[i][i]
            M[r] = [a - m * c for a, c in zip(M[r], M[i])]
    # Substituição regressiva
    x = [Fr(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def avaliar_poli(coef, t):
    """Avalia um polinômio p(t) = a0 + a1*t + ... + an*t^n pelo Método de Horner.

    Algoritmo de Horner (Burden, Algoritmo 2.6):
        b_n = a_n
        b_k = a_k + b_{k+1} * t,   para k = n-1, ..., 0.
        p(t) = b_0.

    Args:
        coef (list[Fraction]): Coeficientes [a0, a1, ..., an] em ordem crescente de grau.
        t (Fraction or float): Ponto onde o polinômio será avaliado.

    Returns:
        Fraction or float: Valor numérico p(t).
    """
    r = Fr(0)
    for a in reversed(coef):
        r = r * t + a
    return r


# ----------------------------------------------------------------------------- (b)
def base_lagrange(x, k, t):
    """Calcula o k-ésimo polinômio base de Lagrange L_{n,k}(t).

    Fórmula (Burden, Eq. 3.2):
        L_{n,k}(t) = prod_{i=0, i != k}^n ((t - x_i) / (x_k - x_i)).

    Propriedade fundamental: L_{n,k}(x_j) = delta_{kj} (1 se j == k, 0 se j != k).

    Args:
        x (list[Fraction]): Nós de interpolação x_0, ..., x_n.
        k (int): Índice da base a ser avaliada (0 <= k <= n).
        t (Fraction): Ponto onde a base será avaliada.

    Returns:
        Fraction: Valor numérico de L_{n,k}(t).
    """
    L = Fr(1)
    for i, xi in enumerate(x):
        if i != k:
            L *= (t - xi) / (x[k] - xi)
    return L


def lagrange(x, y, t):
    """Avalia o polinômio interpolador de Lagrange P(t) no ponto t.

    Fórmula (Burden, Eq. 3.1):
        P(t) = sum_{k=0}^n y_k * L_{n,k}(t).

    Garante que P(x_i) = y_i para todos os pontos amostrados.

    Args:
        x (list[Fraction]): Nós de interpolação x_0, ..., x_n.
        y (list[Fraction]): Valores da função nos nós y_0, ..., y_n.
        t (Fraction): Ponto de interpolação.

    Returns:
        Fraction: Valor interpolado P(t).
    """
    return sum(yk * base_lagrange(x, k, t) for k, yk in enumerate(y))


def mult_poli(p, q):
    """Multiplica dois polinômios representados pelas listas de coeficientes.

    Args:
        p (list[Fraction]): Coeficientes do primeiro polinômio [p0, p1, ...].
        q (list[Fraction]): Coeficientes do segundo polinômio [q0, q1, ...].

    Returns:
        list[Fraction]: Coeficientes do polinômio produto p * q.
    """
    r = [Fr(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def coeficientes_lagrange(x, y):
    """Obtém a forma canônica (na base de monômios) do polinômio interpolador de Lagrange.

    Expande P(x) = sum_{k=0}^n y_k * L_{n,k}(x) em termos de potências 1, x, x^2, ..., x^n.

    Args:
        x (list[Fraction]): Nós de interpolação.
        y (list[Fraction]): Valores nos nós.

    Returns:
        list[Fraction]: Coeficientes [c0, c1, ..., cn] tais que P(x) = c0 + c1*x + ... + cn*x^n.
    """
    n = len(x)
    total = [Fr(0)] * n
    for k in range(n):
        Lk = [Fr(1)]
        for i in range(n):
            if i != k:
                # Multiplica pelo fator linear (t - x_i) / (x_k - x_i)
                Lk = mult_poli(Lk, [-x[i] / (x[k] - x[i]), Fr(1) / (x[k] - x[i])])
        for g in range(n):
            total[g] += y[k] * Lk[g]
    return total


# ----------------------------------------------------------------------------- (c)
def neville(x, y, t):
    """Calcula o valor interpolado no ponto t utilizando o Método de Neville (Algoritmo 3.1).

    O algoritmo constrói uma tabela triangular de aproximações polinomiais Q_{i,j},
    onde Q_{i,j} interpola a função nos pontos x_{i-j}, x_{i-j+1}, ..., x_i:
        Q_{i,0} = y_i,   para i = 0, ..., n.
        Q_{i,j} = ((t - x_{i-j}) * Q_{i,j-1} - (t - x_i) * Q_{i-1,j-1}) / (x_i - x_{i-j}),
                  para 1 <= j <= i <= n.
    O resultado final de maior grau interpolando todos os pontos é Q_{n,n}.

    Args:
        x (list[Fraction]): Nós de interpolação x_0, ..., x_n.
        y (list[Fraction]): Valores nos nós y_0, ..., y_n.
        t (Fraction): Ponto onde se deseja estimar o valor.

    Returns:
        tuple[Fraction, list[list[Fraction or None]]]:
            - Q[n-1][n-1]: O valor final da interpolação em t.
            - Q: Matriz triangular n x n com toda a tabela de Neville.
    """
    n = len(x)
    Q = [[None] * n for _ in range(n)]
    for i in range(n):
        Q[i][0] = y[i]
    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i][j] = ((t - x[i - j]) * Q[i][j - 1] - (t - x[i]) * Q[i - 1][j - 1]) / (x[i] - x[i - j])
    return Q[n - 1][n - 1], Q


def termo_poli(coef, var="h"):
    """Formata os coeficientes de um polinômio como expressão matemática em string.

    Args:
        coef (list[Fraction]): Coeficientes [a0, a1, a2, ...] em ordem crescente de grau.
        var (str): Nome da variável independente. Default: 'h'.

    Returns:
        str: String formatada representando o polinômio (ex.: '10 + 918h + 540h^2').
    """
    partes = []
    for g, a in enumerate(coef):
        if a == 0:
            continue
        mon = "" if g == 0 else (var if g == 1 else f"{var}^{g}")
        partes.append(f"{'+' if a > 0 else '-'} {fmt(abs(a))}{mon}")
    return " ".join(partes).lstrip("+ ")


def plotar(c2, c4, v2, v4, arquivo=None):
    """Gera o gráfico comparativo entre o ajuste por mínimos quadrados e a interpolação de Lagrange.

    Mostra os dados amostrados, a curva suave p2(h), a curva oscilante p4(h) e as estimativas em h = 1.25 m.

    Args:
        c2 (list[Fraction]): Coeficientes do polinômio p2(h) (mínimos quadrados).
        c4 (list[Fraction]): Coeficientes do polinômio p4(h) (Lagrange).
        v2 (Fraction): Volume estimado por p2(1.25).
        v4 (Fraction): Volume estimado por p4(1.25).
        arquivo (str, optional): Caminho do arquivo de imagem a ser salvo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if arquivo is None:
        diretorio = os.path.dirname(os.path.abspath(__file__))
        arquivo = os.path.join(diretorio, "q2_ajuste.png")

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
    print("=====================================================================")
    print("QUESTÃO 2 — Calibração de Reservatório: Mínimos Quadrados, Lagrange e Neville")
    print("=====================================================================")
    print("Dados experimentais:")
    for h, v in zip(H, V):
        print(f"    h = {fmt(h):>3} m  ->  V = {fmt(v):>5} L")

    # (a) -------------------------------------------------------------------
    A, b = equacoes_normais(H, V, 2)
    print("\n(a) Equações normais para p2(h) = a0 + a1*h + a2*h^2:")
    for linha, bi in zip(A, b):
        print("    [" + "  ".join(f"{fmt(a):>7}" for a in linha) + f" ] [a]  =  {fmt(bi)}")
    a = eliminacao_gauss(A, b)
    print(f"    Coeficientes obtidos: a0 = {fmt(a[0])}, a1 = {fmt(a[1])}, a2 = {fmt(a[2])}")
    print(f"    Polinômio de ajuste: p2(h) = {termo_poli(a)}")
    res = [v - avaliar_poli(a, h) for h, v in zip(H, V)]
    sq_res = sum(r * r for r in res)
    print(f"    Resíduos V_i - p2(h_i) = {[fmt(r) for r in res]}")
    print(f"    Soma dos quadrados dos resíduos (erro quadrático mínimo) E = {fmt(sq_res)}")
    v2 = avaliar_poli(a, H_ALVO)
    print(f"    Estimativa do volume para h = 1,25 m: V(1,25) ~ p2(1,25) = {fmt(v2)} L")

    # (b) -------------------------------------------------------------------
    print("\n(b) Interpolação de Lagrange com os 5 pontos:")
    for k in range(5):
        print(f"    L_{k}(1,25) = {base_lagrange(H, k, H_ALVO)}")
    v4 = lagrange(H, V, H_ALVO)
    c4 = coeficientes_lagrange(H, V)
    print(f"    Polinômio interpolador: p4(h) = {termo_poli(c4)}")
    print(f"    Verificação p4(h_i) = V_i nos pontos amostrados: {[fmt(avaliar_poli(c4, h)) for h in H]}")
    print(f"    Estimativa do volume para h = 1,25 m: V(1,25) ~ p4(1,25) = {fmt(v4)} L  (= {v4})")

    # (c) -------------------------------------------------------------------
    vn, Q = neville(H, V, H_ALVO)
    print("\n(c) Tabela triangular de Neville em h = 1,25 m (linha i, coluna j = Q_{i,j}):")
    print("    " + f"{'x_i':>6}" + "".join(f"{'Q_i,' + str(j):>14}" for j in range(5)))
    for i in range(5):
        print("    " + f"{fmt(H[i]):>6}" + "".join(f"{fmt(Q[i][j], 6):>14}" for j in range(i + 1)))
    print(f"    Valor final obtido Q_4,4 = {fmt(vn)} L")
    print(f"    Conferência com Lagrange p4(1,25): valores idênticos? {vn == v4}")

    dif = v2 - v4
    print(f"\nComparativo: Diferença entre mínimos quadrados e interpolação: p2 - p4 = {fmt(dif)} L")
    print("Explicação física/experimental: como os dados experimentais possuem pequenos erros,")
    print("o polinômio de mínimos quadrados p2(h) filtra o ruído, enquanto o interpolador")
    print("p4(h) força a passagem exata sobre os pontos ruidosos, introduzindo pequenas oscilações.")

    diretorio = os.path.dirname(os.path.abspath(__file__))
    arq_fig = os.path.join(diretorio, "q2_ajuste.png")
    plotar(a, c4, v2, v4, arquivo=arq_fig)
    print(f"Figura salva em: {arq_fig}")
