"""
Questão 1 — Sistema massa-mola em equilíbrio resolvido por Gauss-Seidel.

Modelagem física e métodos numéricos aplicados:
  * Balanço de forças pela Lei de Hooke para 3 blocos acoplados por 4 molas
    entre duas paredes fixas:
      Bloco 1: -k1*x1 + k2*(x2 - x1) + F1 = 0
      Bloco 2: -k2*(x2 - x1) + k3*(x3 - x2) + F2 = 0
      Bloco 3: -k3*(x3 - x2) - k4*x3 + F3 = 0
  * Forma matricial: A x = b.
  * Análise teórica de convergência:
      - Dominância diagonal estrita (verificação mostra que não se aplica na linha 2).
      - Simetria e Positividade Definida (Critério de Sylvester / Teorema de Ostrowski-Reich):
        garante convergência para qualquer vetor inicial.
      - Forma iterativa estacionária x^(k+1) = T x^(k) + c:
        cálculo explícito de T = (D - L)^(-1) U e c = (D - L)^(-1) b.
      - Raio espectral rho(T) = 1/3 < 1 e norma ||T||_inf = 2/3 < 1 (critérios de convergência).
  * Resolução numérica por Gauss-Seidel com tolerância 1e-5 na norma infinito.

Referências (Burden, Faires & Burden, Análise Numérica, 10ª ed.):
  * Seção 7.3, Eq. (7.8)–(7.10) e Algoritmo 7.2 (pp. 506–507) — Gauss-Seidel
  * Teorema 7.11 (p. 489) — Norma infinito de matriz
  * Definição 7.14 (p. 496) — Raio espectral
  * Teorema 7.19 e Corolário 7.20 (pp. 508–509) — Convergência de x = Tx + c
  * Teorema 7.21 (p. 510) — Matriz estritamente diagonal dominante
  * Teorema 7.25 (Ostrowski–Reich) e 7.26 (p. 518) — Matriz positiva definida
  * Teorema 6.25 (p. 462) — Critério dos menores principais de Sylvester

Implementação sem bibliotecas numéricas avançadas (Python puro com fractions.Fraction).
"""

from fractions import Fraction as Fr
import os


# ----------------------------------------------------------------------------- (a)
def montar_sistema(k, f):
    """Monta o sistema linear Ax = b para uma cadeia de blocos e molas em equilíbrio.

    Considera uma cadeia horizontal:
        parede — k1 — bloco 1 — k2 — bloco 2 — ... — bloco n — k_{n+1} — parede.
    
    Balanço de forças no bloco i (Hooke, sentido positivo para a direita):
        -k_i * (x_i - x_{i-1}) + k_{i+1} * (x_{i+1} - x_i) + f_i = 0,
    com condições de contorno x_0 = x_{n+1} = 0 (paredes fixas).
    Rearranjando em termos dos deslocamentos:
        -k_i * x_{i-1} + (k_i + k_{i+1}) * x_i - k_{i+1} * x_{i+1} = f_i.

    Args:
        k (list of float or int): Lista com as n+1 constantes elásticas das molas (N/m).
        f (list of float or int): Lista com as n forças externas aplicadas a cada bloco (N).

    Returns:
        tuple[list[list[Fraction]], list[Fraction]]:
            - A: Matriz dos coeficientes n x n, com elementos em Fraction.
            - b: Vetor dos termos independentes de tamanho n, em Fraction.
    """
    n = len(f)
    A = [[Fr(0)] * n for _ in range(n)]
    for i in range(n):
        A[i][i] = Fr(k[i] + k[i + 1])
        if i > 0:
            A[i][i - 1] = Fr(-k[i])
        if i < n - 1:
            A[i][i + 1] = Fr(-k[i + 1])
    b = [Fr(v) for v in f]
    return A, b


# ----------------------------------------------------------------------------- (b)
def estritamente_diagonal_dominante(A):
    """Verifica se a matriz A é estritamente diagonal dominante por linhas.

    Uma matriz é estritamente diagonal dominante se para toda linha i:
        |a_ii| > sum_{j != i} |a_ij|.

    Args:
        A (list[list[Fraction]]): Matriz quadrada n x n.

    Returns:
        tuple[bool, list[tuple[Fraction, Fraction, bool]]]:
            - Booleano indicando se A é estritamente diagonal dominante.
            - Lista de tuplas (diag_i, soma_fora_i, estritamente_maior) para cada linha.
    """
    linhas = []
    for i, linha in enumerate(A):
        diag = abs(linha[i])
        soma_fora = sum(abs(a) for j, a in enumerate(linha) if j != i)
        linhas.append((diag, soma_fora, diag > soma_fora))
    return all(ok for _, _, ok in linhas), linhas


def eh_simetrica(A):
    """Verifica se a matriz quadrada A é simétrica (A = A^T).

    Args:
        A (list[list[Fraction]]): Matriz quadrada n x n.

    Returns:
        bool: True se A[i][j] == A[j][i] para todos os índices i, j; False caso contrário.
    """
    n = len(A)
    return all(A[i][j] == A[j][i] for i in range(n) for j in range(n))


def determinante(M):
    """Calcula o determinante exato de uma matriz quadrada via eliminação de Gauss.

    Utiliza aritmética exata com fractions.Fraction e troca de linhas quando necessário.

    Args:
        M (list[list[Fraction]]): Matriz quadrada n x n.

    Returns:
        Fraction: Valor do determinante de M.
    """
    M = [linha[:] for linha in M]
    n, det = len(M), Fr(1)
    for i in range(n):
        p = next((r for r in range(i, n) if M[r][i] != 0), None)
        if p is None:
            return Fr(0)
        if p != i:
            M[i], M[p] = M[p], M[i]
            det = -det
        det *= M[i][i]
        for r in range(i + 1, n):
            m = M[r][i] / M[i][i]
            M[r] = [a - m * c for a, c in zip(M[r], M[i])]
    return det


def menores_principais(A):
    """Calcula os determinantes dos menores principais líderes det(A_k), k = 1..n.

    Pelo Critério de Sylvester (Burden, Teorema 6.25), uma matriz simétrica real
    é positiva definida se e somente se todos os seus menores principais líderes
    forem estritamente positivos.

    Args:
        A (list[list[Fraction]]): Matriz quadrada n x n.

    Returns:
        list[Fraction]: Lista com os n determinantes [det(A_1), det(A_2), ..., det(A_n)].
    """
    return [determinante([linha[:k] for linha in A[:k]]) for k in range(1, len(A) + 1)]


# ----------------------------------------------------------------------------- (c)
def matriz_iteracao_gauss_seidel(A, b):
    """Determina a matriz de iteração T e o vetor c do método de Gauss-Seidel.

    Dada a decomposição A = D - L - U, onde D é a diagonal de A, -L é a parte
    triangular estritamente inferior e -U é a parte triangular estritamente superior:
        (D - L) x^(k+1) = U x^(k) + b
        x^(k+1) = (D - L)^(-1) U x^(k) + (D - L)^(-1) b.
    Logo:
        T = (D - L)^(-1) U
        c = (D - L)^(-1) b.

    O cálculo é feito por substituição progressiva coluna a coluna, evitando inversão direta.

    Args:
        A (list[list[Fraction]]): Matriz quadrada dos coeficientes n x n.
        b (list[Fraction]): Vetor dos termos independentes de tamanho n.

    Returns:
        tuple[list[list[Fraction]], list[Fraction]]:
            - T: Matriz de iteração de Gauss-Seidel n x n em frações exatas.
            - c: Vetor constante de iteração de tamanho n em frações exatas.
    """
    n = len(A)
    DL = [[A[i][j] if j <= i else Fr(0) for j in range(n)] for i in range(n)]
    U = [[-A[i][j] if j > i else Fr(0) for j in range(n)] for i in range(n)]

    def progressiva(Lm, y):
        x = [Fr(0)] * n
        for i in range(n):
            x[i] = (y[i] - sum(Lm[i][j] * x[j] for j in range(i))) / Lm[i][i]
        return x

    colunas = [progressiva(DL, [U[i][j] for i in range(n)]) for j in range(n)]
    T = [[colunas[j][i] for j in range(n)] for i in range(n)]
    c = progressiva(DL, b)
    return T, c


def norma_inf_matriz(M):
    """Calcula a norma infinito de uma matriz M (máxima soma das linhas em módulo).

    Fórmula (Burden, Teorema 7.11):
        ||M||_inf = max_{1 <= i <= n} sum_{j=1}^n |m_{ij}|.

    Args:
        M (list[list[Fraction]]): Matriz quadrada n x n.

    Returns:
        Fraction: Valor exato de ||M||_inf.
    """
    return max(sum(abs(a) for a in linha) for linha in M)


def raio_espectral_potencia(M, iters=200, tol=1e-14):
    """Estima o raio espectral rho(M) pelo Método das Potências com norma infinito.

    Args:
        M (list[list[Fraction]]): Matriz quadrada n x n.
        iters (int): Número máximo de iterações do método das potências. Default: 200.
        tol (float): Tolerância para convergência do autovalor dominante. Default: 1e-14.

    Returns:
        float: Estimativa de rho(M) = max |lambda_i|.
    """
    n = len(M)
    x = [1.0] * n
    mu = 0.0
    for _ in range(iters):
        y = [sum(float(M[i][j]) * x[j] for j in range(n)) for i in range(n)]
        norma = max(abs(v) for v in y)
        if norma == 0:
            return 0.0
        mu_novo = norma / max(abs(v) for v in x)
        x = [v / norma for v in y]
        if abs(mu_novo - mu) < tol:
            return mu_novo
        mu = mu_novo
    return mu


def polinomio_caracteristico(M):
    """Calcula os coeficientes do polinômio característico det(lambda*I - M).

    Utiliza o Algoritmo de Faddeev–LeVerrier com aritmética exata de frações:
        p(lambda) = lambda^n + c_1 lambda^{n-1} + ... + c_n.

    Args:
        M (list[list[Fraction]]): Matriz quadrada n x n.

    Returns:
        list[Fraction]: Lista com os coeficientes [1, c_1, c_2, ..., c_n].
    """
    n = len(M)
    I = [[Fr(int(i == j)) for j in range(n)] for i in range(n)]
    mult = lambda X, Y: [[sum(X[i][t] * Y[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
    coef = [Fr(1)]
    Mk = [[Fr(0)] * n for _ in range(n)]
    ck = Fr(1)
    for k in range(1, n + 1):
        Mk = mult(M, [[Mk[i][j] + ck * I[i][j] for j in range(n)] for i in range(n)])
        ck = -sum(Mk[i][i] for i in range(n)) / k
        coef.append(ck)
    return coef


def texto_polinomio(coef):
    """Converte os coeficientes de um polinômio em string formatada em potências de lambda.

    Args:
        coef (list[Fraction]): Coeficientes do polinômio do grau mais alto ao termo independente.

    Returns:
        str: Representação textual do polinômio.
    """
    n = len(coef) - 1
    termos = []
    for k, c in enumerate(coef):
        if c == 0:
            continue
        g = n - k
        var = "" if g == 0 else ("l" if g == 1 else f"l^{g}")
        if abs(c) == 1 and g > 0:
            num = ""
        elif g > 0 and abs(c).denominator != 1:
            num = f"({fr(abs(c))})"
        else:
            num = fr(abs(c))
        termos.append(f"{'+' if c > 0 else '-'} {num}{var}")
    return " ".join(termos).lstrip("+ ")


# ----------------------------------------------------------------------------- (d)
def gauss_seidel(A, b, x0, tol=1e-5, max_iter=1000):
    """Executa o método iterativo de Gauss-Seidel para resolver Ax = b.

    Atualiza as variáveis na ordem x_1, x_2, ..., x_n utilizando imediatamente
    os valores mais recentes calculados na iteração corrente:
        x_i^(k+1) = (b_i - sum_{j < i} a_ij x_j^(k+1) - sum_{j > i} a_ij x_j^(k)) / a_ii.

    Critério de parada:
        ||x^(k+1) - x^(k)||_inf < tol.

    Args:
        A (list[list[float or Fraction]]): Matriz dos coeficientes n x n.
        b (list[float or Fraction]): Vetor dos termos independentes de tamanho n.
        x0 (list[float or Fraction]): Vetor de estimativa inicial de tamanho n.
        tol (float): Tolerância para o critério de parada na norma infinito. Default: 1e-5.
        max_iter (int): Número máximo de iterações permitidas. Default: 1000.

    Returns:
        tuple[list[float], int, bool, list[tuple[int, list[float], float or None]]]:
            - x: Vetor solução aproximada.
            - it: Número de iterações executadas.
            - conv: Booleano indicando se convergiu dentro de max_iter.
            - historico: Lista de tuplas (k, x_k, dif_k) com a trajetória iterativa.
    """
    n = len(A)
    A_float = [[float(a) for a in linha] for linha in A]
    b_float = [float(v) for v in b]
    x = [float(v) for v in x0]
    historico = [(0, x[:], None)]
    for k in range(1, max_iter + 1):
        x_ant = x[:]
        for i in range(n):
            s1 = sum(A_float[i][j] * x[j] for j in range(i))            # valores já atualizados
            s2 = sum(A_float[i][j] * x_ant[j] for j in range(i + 1, n))  # valores da iteração anterior
            x[i] = (b_float[i] - s1 - s2) / A_float[i][i]
        dif = max(abs(x[i] - x_ant[i]) for i in range(n))
        historico.append((k, x[:], dif))
        if dif < tol:
            return x, k, True, historico
    return x, max_iter, False, historico


def fr(v):
    """Formata uma fração como string reduzida (ex.: '1/3' ou '5').

    Args:
        v (Fraction): Valor fracionário.

    Returns:
        str: Representação simplificada.
    """
    return str(v) if v.denominator != 1 else str(v.numerator)


def plotar_convergencia(historico, x_exato=(0.1, 0.2, 0.1), arquivo=None):
    """Gera gráficos da convergência das variáveis e da norma do passo por iteração.

    Args:
        historico (list[tuple]): Histórico de iterações gerado por gauss_seidel.
        x_exato (tuple[float]): Solução exata analítica do sistema.
        arquivo (str, optional): Caminho do arquivo para salvar a figura.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if arquivo is None:
        diretorio = os.path.dirname(os.path.abspath(__file__))
        arquivo = os.path.join(diretorio, "q1_convergencia.png")

    ks = [item[0] for item in historico]
    x1 = [item[1][0] for item in historico]
    x2 = [item[1][1] for item in historico]
    x3 = [item[1][2] for item in historico]
    difs = [item[2] for item in historico[1:]]
    kdifs = ks[1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Gráfico 1: Trajetória dos deslocamentos
    ax1.plot(ks, x1, "o-", color="tab:blue", label="$x_1^{(k)}$", ms=5)
    ax1.plot(ks, x2, "s-", color="tab:orange", label="$x_2^{(k)}$", ms=5)
    ax1.plot(ks, x3, "^-", color="tab:green", label="$x_3^{(k)}$", ms=5)
    ax1.axhline(x_exato[0], color="tab:blue", ls="--", alpha=0.5, label="exato $x_1=0.1$ m")
    ax1.axhline(x_exato[1], color="tab:orange", ls="--", alpha=0.5, label="exato $x_2=0.2$ m")
    ax1.set_xlabel("Iteração $k$")
    ax1.set_ylabel("Deslocamento (m)")
    ax1.set_title("Convergência dos deslocamentos")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    # Gráfico 2: Norma infinito do passo (escala logarítmica)
    ax2.semilogy(kdifs, difs, "o-", color="tab:red", ms=5, label=r"$\|x^{(k)} - x^{(k-1)}\|_\infty$")
    ax2.axhline(1e-5, color="k", ls=":", label="tolerância $10^{-5}$ m")
    ax2.set_xlabel("Iteração $k$")
    ax2.set_ylabel(r"$\|x^{(k)} - x^{(k-1)}\|_\infty$ (m)")
    ax2.set_title("Critério de parada (escala log)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    k = [200, 100, 100, 200]          # Constantes elásticas das molas (N/m)
    f = [10, 20, 10]                   # Forças externas aplicadas aos blocos (N)
    A, b = montar_sistema(k, f)

    print("=====================================================================")
    print("QUESTÃO 1 — Sistema Massa-Mola e Método de Gauss-Seidel")
    print("=====================================================================")

    # (a) ---------------------------------------------------------------------
    print("\n(a) Sistema linear Ax = b:")
    for linha, bi in zip(A, b):
        print("   ", " ".join(f"{fr(a):>6}" for a in linha), "|", fr(bi))

    # (b) ---------------------------------------------------------------------
    print("\n(b) Propriedades da matriz A:")
    sdd, linhas = estritamente_diagonal_dominante(A)
    for i, (d, R, ok) in enumerate(linhas, 1):
        print(f"    Linha {i}: |a_{i}{i}| = {fr(d)}, soma fora da diagonal = {fr(R)} -> {'>' if ok else 'NÃO >'}")
    print(f"    Estritamente diagonal dominante? {sdd}")
    print(f"    Simétrica? {eh_simetrica(A)}")
    dets = menores_principais(A)
    print(f"    Menores principais líderes det(A_1), det(A_2), det(A_3) = {[fr(d) for d in dets]}")
    eh_pd = all(d > 0 for d in dets)
    print(f"    Positiva definida (Critério de Sylvester, Teorema 6.25)? {eh_pd}")
    print("    Teorema de Ostrowski-Reich (Teorema 7.25):")
    print("    Se A é simétrica, com a_ii > 0 e positiva definida, então o método")
    print("    de Gauss-Seidel converge para qualquer vetor inicial x^(0).")

    # (c) ---------------------------------------------------------------------
    print("\n(c) Forma iterativa x^(k+1) = T x^(k) + c:")
    T, c = matriz_iteracao_gauss_seidel(A, b)
    for linha, ci in zip(T, c):
        print("    T:", " ".join(f"{fr(a):>6}" for a in linha), "   c:", fr(ci))
    nT = norma_inf_matriz(T)
    print(f"    ||T||_inf = {fr(nT)} = {float(nT):.6f}  (< 1? {nT < 1})")
    print(f"    Polinômio característico det(lambda*I - T) = {texto_polinomio(polinomio_caracteristico(T))}"
          "  -> autovalores 0, 0, 1/3")
    rho = raio_espectral_potencia(T)
    print(f"    Raio espectral rho(T) = {rho:.10f}  (valor analítico: 1/3 < 1)")
    print("    Critério de Convergência (Teorema 7.19): rho(T) < 1 garante convergência para todo x^(0).")
    print("    Adicionalmente, ||T||_inf = 2/3 < 1 é condição suficiente (Corolário 7.20).")

    # (d) ---------------------------------------------------------------------
    print("\n(d) Execução de Gauss-Seidel:")
    print("    x^(0) = (0, 0, 0)^T, tol = 10^-5 m na norma infinito, max_iter = 1000")
    x, it, conv, hist = gauss_seidel(A, b, [0, 0, 0])
    print(f"    {'k':>3} {'x1 (m)':>12} {'x2 (m)':>12} {'x3 (m)':>12} {'||x(k)-x(k-1)||inf':>20}")
    for kk, xk, d in hist:
        ds = "" if d is None else f"{d:.3e}"
        print(f"    {kk:>3} {xk[0]:12.8f} {xk[1]:12.8f} {xk[2]:12.8f} {ds:>20}")
    if conv:
        print(f"\n    Convergência atingida com sucesso em {it} iterações.")
    else:
        print(f"\n    ATENÇÃO: Critério de parada não atingido em {it} iterações.")
    print(f"    Deslocamentos de equilíbrio: x1 = {x[0]:.6f} m, x2 = {x[1]:.6f} m, x3 = {x[2]:.6f} m")
    res = max(abs(sum(float(A[i][j]) * x[j] for j in range(3)) - float(b[i])) for i in range(3))
    print(f"    Resíduo final ||Ax - b||_inf = {res:.2e} N")
    erro = max(abs(x[0] - 0.1), abs(x[1] - 0.2), abs(x[2] - 0.1))
    print(f"    Erro absoluto em relação à solução analítica (0.1, 0.2, 0.1): {erro:.2e} m")

    diretorio = os.path.dirname(os.path.abspath(__file__))
    arq_fig = os.path.join(diretorio, "q1_convergencia.png")
    plotar_convergencia(hist, arquivo=arq_fig)
    print(f"    Gráfico de convergência salvo em: {arq_fig}")
