"""
Questao 6 - Metodos Numericos (Prof. Thales Vieira)
Equacao: x^3 - 2x - 5 = 0

Implementacoes SEM uso de bibliotecas numericas (apenas Python puro).
"""


def f(x):
    """f(x) = x^3 - 2x - 5"""
    return x**3 - 2*x - 5


def df(x):
    """f'(x) = 3x^2 - 2 (derivada de f)"""
    return 3*x**2 - 2


# ---------------------------------------------------------------------------
# a) Metodo de Newton
# ---------------------------------------------------------------------------
def metodo_newton(f, df, x0, tol, n_max=100):
    """
    Metodo de Newton-Raphson para encontrar raiz de f(x) = 0.

    Parametros:
        f     -- funcao cuja raiz se deseja encontrar
        df    -- derivada de f
        x0    -- aproximacao inicial
        tol   -- tolerancia (criterio de parada: |x_{n+1} - x_n| < tol)
        n_max -- numero maximo de iteracoes

    Retorna:
        (raiz, numero_de_iteracoes) ou (None, numero_de_iteracoes) se nao convergir
    """
    x_atual = x0
    for i in range(1, n_max + 1):
        fx = f(x_atual)
        dfx = df(x_atual)

        if dfx == 0:
            print("Derivada nula: metodo de Newton falhou.")
            return None, i

        x_prox = x_atual - fx / dfx

        if abs(x_prox - x_atual) < tol:
            return x_prox, i

        x_atual = x_prox

    return None, n_max


# ---------------------------------------------------------------------------
# c) Metodo da Bissecao
# ---------------------------------------------------------------------------
def metodo_bissecao(f, a, b, tol, n_max=1000):
    """
    Metodo da Bissecao para encontrar raiz de f(x) = 0 em [a, b].

    Parametros:
        f     -- funcao cuja raiz se deseja encontrar
        a, b  -- extremos do intervalo (f(a) e f(b) devem ter sinais opostos)
        tol   -- tolerancia (criterio de parada: |p_n - p_{n-1}| < tol)
        n_max -- numero maximo de iteracoes

    Retorna:
        (raiz, numero_de_iteracoes) ou (None, numero_de_iteracoes) se nao for
        possivel encontrar a raiz
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        print("f(a) e f(b) tem o mesmo sinal: nao ha garantia de raiz em [a, b].")
        return None, 0

    p_anterior = a
    for i in range(1, n_max + 1):
        p = (a + b) / 2
        fp = f(p)

        if fp == 0 or abs(p - p_anterior) < tol:
            return p, i

        if fa * fp < 0:
            b = p
            fb = fp
        else:
            a = p
            fa = fp

        p_anterior = p

    return None, n_max


# ---------------------------------------------------------------------------
# a) Implementacao do Metodo de Newton
#    A funcao metodo_newton() acima implementa o algoritmo solicitado.
#
# b) Aplicacao do Metodo de Newton:
#    x0 = 2 e tolerancia = 10^-6
#
# c) Aplicacao do Metodo da Bissecao:
#    intervalo [2, 3] e tolerancia = 10^-6
#
# d) Comparacao entre as raizes obtidas e o numero de iteracoes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tol = 1e-6

    # b) Newton com x0 = 2
    raiz_newton, iter_newton = metodo_newton(f, df, x0=2, tol=tol)

    # c) Bissecao no intervalo [2, 3]
    raiz_bissecao, iter_bissecao = metodo_bissecao(f, a=2, b=3, tol=tol)

print("=" * 55)
print("Questao 6 - x^3 - 2x - 5 = 0")
print("=" * 55)

print("a) Metodo de Newton implementado em metodo_newton().")

print("-" * 55)
print("b) Metodo de Newton (x0 = 2, tol = 1e-6):")
print(f"  raiz         = {raiz_newton:.10f}")
print(f"  f(raiz)      = {f(raiz_newton):.2e}")
print(f"  iteracoes    = {iter_newton}")

print("-" * 55)
print("c) Metodo da Bissecao ([2, 3], tol = 1e-6):")
print(f"  raiz         = {raiz_bissecao:.10f}")
print(f"  f(raiz)      = {f(raiz_bissecao):.2e}")
print(f"  iteracoes    = {iter_bissecao}")

print("-" * 55)
print("d) Comparacao:")
print(f"  Diferenca entre as raizes = "
      f"{abs(raiz_newton - raiz_bissecao):.2e}")
print(f"  Newton:   {iter_newton} iteracoes")
print(f"  Bissecao: {iter_bissecao} iteracoes")
print(f"  A Bissecao utilizou "
      f"{iter_bissecao - iter_newton} iteracoes a mais.")
print("=" * 55)

# ---------------------------------------------------------------------------
# Resultados numericos obtidos (saida da execucao acima):
#
# =======================================================
# Equacao: x^3 - 2x - 5 = 0
# =======================================================
# Metodo de Newton (x0 = 2):
#   raiz         = 2.0945514815
#   f(raiz)      = -8.88e-16
#   iteracoes    = 4
# -------------------------------------------------------
# Metodo da Bissecao ([2, 3]):
#   raiz         = 2.0945520401
#   f(raiz)      = 6.23e-06
#   iteracoes    = 20
# -------------------------------------------------------
# Diferenca entre as raizes: 5.59e-07
# Newton precisou de 4 iteracoes.
# Bissecao precisou de 20 iteracoes (16 a mais que Newton).
# =======================================================
