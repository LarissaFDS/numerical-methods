"""
Questão 3 — Perfil de corrente I(t) = exp(-t^2) * sin(t^3):
amostragem, diferenciação numérica de 3 pontos e integração por Simpson composta.

Modelagem física e métodos numéricos aplicados:
  * Modelagem da corrente elétrica transitória em processadores de alto desempenho:
      I(t) = e^(-t^2) * sin(t^3), para t em [0, 3] segundos.
  * Amostragem e cálculo simbólico/analítico:
      Interpretação da expressão analítica via SymPy (sympify e subs) com fallback para math.
  * Diferenciação numérica por fórmulas de 3 pontos (erro de truncamento O(h^2)):
      - Pontos interiores (fórmula do ponto médio / diferenças centradas, Burden Eq. 4.5):
          f'(x_i) ≈ (f(x_{i+1}) - f(x_{i-1})) / (2h) - (h^2 / 6) * f'''(xi)
      - Extremidades (fórmulas de 3 pontos no bordo, Burden Eq. 4.4):
          f'(x_0)     ≈ (-3*f(x_0) + 4*f(x_1) - f(x_2)) / (2h) + (h^2 / 3) * f'''(xi)
          f'(x_{n-1}) ≈ ( 3*f(x_{n-1}) - 4*f(x_{n-2}) + f(x_{n-3})) / (2h) + (h^2 / 3) * f'''(xi)
      - Determinação do instante t_max onde a taxa de variação dI/dt é máxima.
  * Integração numérica (cálculo da carga total transferida Q = integral_0^3 I(t) dt):
      - Regra Composta de Simpson (Burden Algoritmo 4.1 / Teorema 4.4) para n par entre 2 e 50:
          Q ≈ (h / 3) * [ f(x_0) + 2*sum_{j=1}^{n/2-1} f(x_{2j}) + 4*sum_{j=1}^{n/2} f(x_{2j-1}) + f(x_n) ]
      - Análise de convergência assintótica O(h^4) e estimativa de erro por extrapolação de Richardson.

Referências (Burden, Faires & Burden, Análise Numérica, 10ª ed.):
  * Seção 4.1, Eq. (4.4) e (4.5), p. 194 — Fórmulas de derivação numérica de 3 pontos
  * Seção 4.3, p. 216 — Regra elementar de Simpson
  * Seção 4.4, Teorema 4.4 (p. 226) e Algoritmo 4.1 (p. 227) — Regra Composta de Simpson
  * Seção 4.2 — Extrapolação de Richardson e taxa de convergência
"""

import math
import os

try:
    import sympy as sp
    HAS_SYMPY = True
    t_sym = sp.Symbol("t")
except ImportError:
    sp = None
    HAS_SYMPY = False
    t_sym = None

FUNCAO_STR = "exp(-t**2)*sin(t**3)"


# ----------------------------------------------------------------------------- (a)
def amostrar(expr_str, a, b, n):
    """Gera vetores uniformemente espaçados de tempo t e corrente I(t).

    Utiliza SymPy (sympify e subs) para avaliar a expressão matemática dada como string,
    conforme sugerido no enunciado do problema. Caso SymPy não esteja disponível,
    utiliza avaliação matemática segura com a biblioteca math padrão.

    Args:
        expr_str (str): Expressão matemática da função I(t) em termos da variável 't'.
        a (float): Limite inferior do intervalo de tempo (segundos).
        b (float): Limite superior do intervalo de tempo (segundos).
        n (int): Quantidade total de amostras (n >= 2).

    Returns:
        tuple[list[float], list[float]]:
            - t: Lista com os n instantes de tempo amostrados, de a até b inclusive.
            - I: Lista com os n valores calculados da corrente I(t) correspondentes.

    Raises:
        ValueError: Se n < 2 ou se b <= a.
    """
    if n < 2:
        raise ValueError("A quantidade de amostras n deve ser pelo menos 2.")
    if b <= a:
        raise ValueError("O limite superior b deve ser estritamente maior que a.")

    h = (b - a) / (n - 1)
    t = [a + i * h for i in range(n)]

    if HAS_SYMPY:
        expr = sp.sympify(expr_str)
        # Para velocidade e precisão, compila para função matemática nativa
        func = sp.lambdify(t_sym, expr, modules=["math"])
        I = [float(func(ti)) for ti in t]
    else:
        # Fallback analítico direto para exp(-t^2)*sin(t^3)
        func = lambda v: math.exp(-(v ** 2)) * math.sin(v ** 3)
        I = [func(ti) for ti in t]

    return t, I


# ----------------------------------------------------------------------------- (b)
def derivada_3_pontos(t, y):
    """Aproxima numericamente a primeira derivada y'(t_i) para cada ponto amostrado.

    Utiliza fórmulas de 3 pontos com erro de truncamento de ordem O(h^2) (Burden, Seção 4.1):
      * Pontos interiores (1 <= i <= n-2) — Fórmula de diferença centrada de 3 pontos (Eq. 4.5):
            y'_i ≈ (y_{i+1} - y_{i-1}) / (2h)
      * Primeiro ponto (i = 0) — Fórmula de extremidade de 3 pontos para a frente (Eq. 4.4):
            y'_0 ≈ (-3*y_0 + 4*y_1 - y_2) / (2h)
      * Último ponto (i = n-1) — Fórmula de extremidade de 3 pontos para trás (Eq. 4.4 com passo -h):
            y'_{n-1} ≈ (3*y_{n-1} - 4*y_{n-2} + y_{n-3}) / (2h)

    Args:
        t (list[float]): Lista de instantes de tempo igualmente espaçados (passo constante h).
        y (list[float]): Lista de valores amostrados correspondentes a cada instante em t.

    Returns:
        list[float]: Lista de aproximações da primeira derivada dy/dt para cada ponto amostrado.

    Raises:
        ValueError: Se a quantidade de pontos for menor do que 3.
    """
    n = len(t)
    if n < 3:
        raise ValueError("São necessários pelo menos 3 pontos para a fórmula de 3 pontos.")

    h = t[1] - t[0]
    d = [0.0] * n

    # Extremidade esquerda (bordo inicial i = 0)
    d[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h)

    # Pontos interiores (diferenças centradas)
    for i in range(1, n - 1):
        d[i] = (y[i + 1] - y[i - 1]) / (2.0 * h)

    # Extremidade direita (bordo final i = n - 1)
    d[n - 1] = (3.0 * y[n - 1] - 4.0 * y[n - 2] + y[n - 3]) / (2.0 * h)

    return d


# ----------------------------------------------------------------------------- (c)
def simpson_composta(f, a, b, n):
    """Calcula a aproximação numérica da integral definida de f(t) em [a, b] pela Regra de Simpson Composta.

    Fórmula (Burden, Teorema 4.4 / Algoritmo 4.1):
        h = (b - a) / n   (n deve ser par)
        x_i = a + i*h,  para i = 0, ..., n
        Integral ≈ (h / 3) * [ f(x_0) + 2 * sum_{j=1}^{n/2-1} f(x_{2j})
                                      + 4 * sum_{j=1}^{n/2} f(x_{2j-1})
                                      + f(x_n) ]
    Erro de truncamento teórico: E = - ((b - a) / 180) * h^4 * f^(4)(mu), para algum mu em (a, b).

    Args:
        f (callable): Função contínua a ser integrada, que recebe um float e retorna float.
        a (float): Limite inferior de integração.
        b (float): Limite superior de integração.
        n (int): Número de subintervalos (deve ser um inteiro par estritamente positivo).

    Returns:
        float: Valor aproximado da integral definida no intervalo [a, b].

    Raises:
        ValueError: Se n não for um número inteiro par positivo.
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError(f"O número de subintervalos n deve ser um inteiro par positivo. Recebido: {n}")

    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    fx = [f(xi) for xi in x]

    soma_pares = sum(fx[2 * j] for j in range(1, n // 2))
    soma_impares = sum(fx[2 * j - 1] for j in range(1, n // 2 + 1))

    return (h / 3.0) * (fx[0] + 2.0 * soma_pares + 4.0 * soma_impares + fx[n])


def funcao_corrente(expr_str):
    """Constrói uma função executável em Python para avaliar I(t).

    Args:
        expr_str (str): Expressão matemática textual.

    Returns:
        callable: Função Python que recebe float t e retorna float I(t).
    """
    if HAS_SYMPY:
        expr = sp.sympify(expr_str)
        func = sp.lambdify(t_sym, expr, modules=["math"])
        return lambda v: float(func(v))
    return lambda v: math.exp(-(v ** 2)) * math.sin(v ** 3)


def plot_a(t, I, arquivo=None):
    """Gera e salva o gráfico do sinal de corrente I(t) para n = 200 amostras em [0, 3].

    Args:
        t (list[float]): Vetor de tempo.
        I (list[float]): Vetor de corrente amostrada.
        arquivo (str, optional): Caminho do arquivo de saída. Default: 'q3a_sinal.png' na pasta do script.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if arquivo is None:
        diretorio = os.path.dirname(os.path.abspath(__file__))
        arquivo = os.path.join(diretorio, "q3a_sinal.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, I, color="tab:blue", lw=1.8, label=r"$I(t) = e^{-t^2}\sin(t^3)$")
    ax.axhline(0, color="0.5", lw=0.7, ls="--")
    ax.set_xlabel("Tempo $t$ (s)")
    ax.set_ylabel("Corrente $I(t)$ (A)")
    ax.set_title(r"Sinal de Corrente no Processador ($t \in [0, 3]$, $n = 200$ amostras)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    plt.close(fig)


def plot_b(t, I, d, tmax, dmax, tf, If, df, arquivo=None):
    """Gera e salva o gráfico comparativo entre I(t) e dI/dt com destaque para o instante t_max.

    Args:
        t (list[float]): Instantes de tempo amostrados (n = 15).
        I (list[float]): Corrente amostrada nos 15 pontos.
        d (list[float]): Derivadas numéricas calculadas pelas fórmulas de 3 pontos.
        tmax (float): Instante amostrado de máxima taxa de crescimento da corrente.
        dmax (float): Valor máximo de dI/dt observado.
        tf (list[float]): Malha fina de tempo para curva analítica contínua.
        If (list[float]): Corrente exata na malha fina.
        df (list[float]): Derivada exata na malha fina.
        arquivo (str, optional): Caminho do arquivo de saída.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if arquivo is None:
        diretorio = os.path.dirname(os.path.abspath(__file__))
        arquivo = os.path.join(diretorio, "q3b_derivada.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    # Curvas contínuas de referência (traço suave)
    ax.plot(tf, If, color="tab:blue", lw=1.0, alpha=0.35, label=r"$I(t)$ exato (referência)")
    ax.plot(tf, df, color="tab:red", lw=1.0, alpha=0.35, label=r"$dI/dt$ exato (referência)")
    # Pontos amostrados discretos (n = 15)
    ax.plot(t, I, "o-", color="tab:blue", lw=1.4, ms=5, label=r"$I(t)$ (15 amostras)")
    ax.plot(t, d, "s--", color="tab:red", lw=1.4, ms=5, label=r"$dI/dt$ (diferenças de 3 pontos)")
    # Destaque para o ponto de máxima taxa de crescimento
    ax.scatter([tmax], [dmax], s=160, facecolors="none", edgecolors="black", lw=2.0, zorder=5,
               label=rf"Máx. numérico: $t_{{max}} \approx {tmax:.4f}$ s ($dI/dt \approx {dmax:.4f}$ A/s)")
    ax.axhline(0, color="0.5", lw=0.7, ls=":")
    ax.set_xlabel("Tempo $t$ (s)")
    ax.set_ylabel(r"Corrente $I$ (A) e Taxa de Variação $dI/dt$ (A/s)")
    ax.set_title(r"Corrente e Derivada Numérica ($n = 15$ amostras)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    plt.close(fig)


def plot_c(ns, Qs, Qref, arquivo=None):
    """Gera e salva os gráficos de convergência da integral Q em função do número de subintervalos n.

    Apresenta a convergência do valor de Q e o decaimento do erro absoluto em escala logarítmica.

    Args:
        ns (list[int]): Lista de números pares de subintervalos (2, 4, ..., 50).
        Qs (list[float]): Valores calculados de Q para cada n.
        Qref (float): Valor de referência altamente preciso obtido com malha refinada.
        arquivo (str, optional): Caminho do arquivo de saída.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if arquivo is None:
        diretorio = os.path.dirname(os.path.abspath(__file__))
        arquivo = os.path.join(diretorio, "q3c_simpson.png")

    erros = [abs(q - Qref) for q in Qs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Gráfico 1: Valor de Q em função de n
    ax1.plot(ns, Qs, "o-", color="tab:green", ms=4, lw=1.4, label=r"$Q_n$ (Simpson Composta)")
    ax1.axhline(Qref, color="black", ls="--", lw=1.0, label=rf"Referência $Q \approx {Qref:.8f}$ C")
    ax1.set_xlabel("Número de subintervalos $n$ (par)")
    ax1.set_ylabel("Carga elétrica transferida $Q$ (Coulombs)")
    ax1.set_title("Simpson Composta: $Q(n)$")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    # Gráfico 2: Erro absoluto em escala logarítmica (convergência de ordem 4)
    ax2.semilogy(ns, erros, "o-", color="tab:purple", ms=4, lw=1.4, label=r"$|Q_n - Q_{ref}|$")
    ax2.set_xlabel("Número de subintervalos $n$ (par)")
    ax2.set_ylabel(r"Erro absoluto $|Q_n - Q_{ref}|$ (C)")
    ax2.set_title(r"Erro de integração (escala $\log_{10}$)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(arquivo, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    print("=====================================================================")
    print("QUESTÃO 3 — Análise de Corrente Transitória em Processador")
    print("=====================================================================")

    # (a) ---------------------------------------------------------------------
    print("\n(a) Amostragem do sinal de corrente I(t) = exp(-t^2)*sin(t^3) em [0, 3] com n = 200:")
    t200, I200 = amostrar(FUNCAO_STR, 0.0, 3.0, 200)
    diretorio = os.path.dirname(os.path.abspath(__file__))
    arq_a = os.path.join(diretorio, "q3a_sinal.png")
    plot_a(t200, I200, arquivo=arq_a)
    print(f"    Total de amostras geradas: {len(t200)}")
    print(f"    I(t=0.0 s) = {I200[0]:.6f} A")
    print(f"    I(t=3.0 s) = {I200[-1]:.6e} A")
    kmax_I = max(range(len(I200)), key=lambda i: I200[i])
    print(f"    Pico de corrente registrado: {I200[kmax_I]:.4f} A no instante t = {t200[kmax_I]:.4f} s")
    print(f"    Gráfico salvo em: {arq_a}")

    # (b) ---------------------------------------------------------------------
    print("\n(b) Aproximação da derivada dI/dt com fórmulas de 3 pontos (n = 15):")
    t15, I15 = amostrar(FUNCAO_STR, 0.0, 3.0, 15)
    d15 = derivada_3_pontos(t15, I15)
    h15 = t15[1] - t15[0]
    print(f"    Número de amostras n = 15, passo temporal h = 3/14 ~ {h15:.6f} s")

    # Derivada analítica exata via SymPy para conferência do erro numérico
    if HAS_SYMPY:
        dI_simb = sp.diff(sp.sympify(FUNCAO_STR), t_sym)
        func_deriv = sp.lambdify(t_sym, dI_simb, modules=["math"])
    else:
        func_deriv = lambda v: math.exp(-(v ** 2)) * (3.0 * (v ** 2) * math.cos(v ** 3) - 2.0 * v * math.sin(v ** 3))

    print(f"\n    {'i':>2} {'t_i (s)':>9} {'I(t_i) (A)':>12} {'dI/dt aprox':>13} {'dI/dt exata':>13} {'Erro':>11}")
    for i, (ti, Ii, di) in enumerate(zip(t15, I15, d15)):
        ex = func_deriv(ti)
        err = di - ex
        print(f"    {i:>2} {ti:9.4f} {Ii:12.6f} {di:13.6f} {ex:13.6f} {err:11.2e}")

    k_max = max(range(len(d15)), key=lambda i: d15[i])
    t_max_num = t15[k_max]
    d_max_num = d15[k_max]
    print(f"\n    Máxima taxa de crescimento da corrente entre os pontos amostrados:")
    print(f"    t_max = {t_max_num:.6f} s (índice i = {k_max}), com dI/dt ~ {d_max_num:.6f} A/s")

    # Malha fina contínua para comparação gráfica
    tf, If = amostrar(FUNCAO_STR, 0.0, 3.0, 3001)
    df = [func_deriv(v) for v in tf]
    kf_max = max(range(len(df)), key=lambda i: df[i])
    print(f"    Referência contínua refinada (3001 pontos): t_max ~ {tf[kf_max]:.4f} s, dI/dt ~ {df[kf_max]:.6f} A/s")

    arq_b = os.path.join(diretorio, "q3b_derivada.png")
    plot_b(t15, I15, d15, t_max_num, d_max_num, tf, If, df, arquivo=arq_b)
    print(f"    Gráfico comparativo salvo em: {arq_b}")

    # (c) ---------------------------------------------------------------------
    print("\n(c) Cálculo da carga elétrica transferida Q = integral_0^3 I(t) dt por Simpson Composta:")
    f_integrando = funcao_corrente(FUNCAO_STR)
    ns = list(range(2, 51, 2))
    Qs = [simpson_composta(f_integrando, 0.0, 3.0, n) for n in ns]
    # Referência de alta precisão com n = 4000
    Q_ref = simpson_composta(f_integrando, 0.0, 3.0, 4000)

    print(f"    {'n (subintervalos)':>18} {'Passo h (s)':>14} {'Q_n (Coulombs)':>18} {'|Q_n - Q_ref| (C)':>18}")
    for n, q in zip(ns, Qs):
        h_sub = 3.0 / n
        print(f"    {n:>18} {h_sub:14.6f} {q:18.10f} {abs(q - Q_ref):18.3e}")

    print(f"\n    Carga transferida final (n = 50): Q_50 ~ {Qs[-1]:.10f} C")
    print(f"    Valor de referência refinado (n = 4000): Q_ref ~ {Q_ref:.10f} C")

    # Estimativa de erro assintótico por Extrapolação de Richardson:
    # Como o erro de Simpson é O(h^4), Q - Q_50 ≈ (16/15) * (Q_100 - Q_50)
    q100 = simpson_composta(f_integrando, 0.0, 3.0, 100)
    erro_richardson = (16.0 / 15.0) * abs(q100 - Qs[-1])
    print(f"    Erro estimado em Q_50 por Extrapolação de Richardson: {erro_richardson:.3e} C")

    arq_c = os.path.join(diretorio, "q3c_simpson.png")
    plot_c(ns, Qs, Q_ref, arquivo=arq_c)
    print(f"    Gráficos de convergência salvos em: {arq_c}")
