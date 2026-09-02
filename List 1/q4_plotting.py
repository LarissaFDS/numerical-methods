
import math
import matplotlib.pyplot as plt


# a) Funcao que amostra e plota y = f(x) em [a, b] usando segmentos de reta

def plotar_funcao(f, a, b, n, ax=None, titulo="Grafico de f(x)"):
    """
    Amostra f uniformemente em [a, b] com n pontos e plota y = f(x)
    usando segmentos de reta ligando os pontos amostrados.

    Parametros:
        f      -- funcao a ser plotada
        a, b   -- extremos do intervalo
        n      -- quantidade de pontos amostrados uniformemente
        ax     -- eixo matplotlib onde desenhar (cria um novo se None)
        titulo -- titulo do grafico

    Retorna:
        (xs, ys) -- listas com os pontos amostrados (uteis para reuso)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    passo = (b - a) / (n - 1)
    xs = [a + i * passo for i in range(n)]
    ys = [f(x) for x in xs]

    # desenha manualmente os segmentos de reta ligando (x_i, y_i) a (x_{i+1}, y_{i+1})
    for i in range(n - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], color="tab:blue", linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)

    return xs, ys


def funcao_teste(x):
    """Funcao qualquer usada apenas para validar plotar_funcao (letra a)."""
    return math.sin(x) + 0.3 * x



# b) Bissecao que retorna a sequencia (p_n) de aproximacoes, nao so a raiz

def bissecao_sequencia(f, a, b, tol, n_max=1000):
    """
    Igual ao Metodo da Bissecao da questao 3b, mas retorna a LISTA completa
    de aproximacoes p_n geradas ao longo das iteracoes, em vez de apenas a
    raiz final.

    Retorna:
        lista_p -- lista [p_1, p_2, ..., p_k] com todas as aproximacoes
                   (lista vazia se f(a) e f(b) tiverem o mesmo sinal)
    """
    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        print("f(a) e f(b) tem o mesmo sinal: nao ha garantia de raiz em [a, b].")
        return []

    lista_p = []
    p_anterior = a
    for _ in range(n_max):
        p = (a + b) / 2
        fp = f(p)
        lista_p.append(p)

        if fp == 0 or abs(p - p_anterior) < tol:
            break

        if fa * fp < 0:
            b = p
            fb = fp
        else:
            a = p
            fa = fp

        p_anterior = p

    return lista_p


def plotar_convergencia(f, a, b, n, lista_p, titulo="f(x) e sequencia de aproximacoes"):
    """
    Plota y = f(x) (usando plotar_funcao) junto com os pontos
    (p_n, f(p_n)) gerados pela bissecao, na ordem em que foram calculados.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    plotar_funcao(f, a, b, n, ax=ax, titulo=titulo)

    xs_p = lista_p
    ys_p = [f(p) for p in lista_p]

    ax.plot(xs_p, ys_p, "o--", color="tab:red", markersize=5,
             linewidth=1, label="p_n (bissecao)")

    for i, (x, y) in enumerate(zip(xs_p, ys_p)):
        ax.annotate(str(i + 1), (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="tab:red")

    ax.legend()
    return fig

# Execucao

if __name__ == "__main__":
    # a) validacao com uma funcao qualquer
    fig_a, ax_a = plt.subplots(figsize=(7, 5))
    plotar_funcao(funcao_teste, a=0, b=10, n=200, ax=ax_a,
                  titulo="Validacao da letra a): f(x) = sin(x) + 0.3x")
    fig_a.savefig("questao4a_validacao.png", dpi=150, bbox_inches="tight")
    print("Grafico da letra a) salvo em questao4a_validacao.png")

    # b) aplicacao em f(x) = ln(x) - 2^x + x^2 - 1, [3, 5]
    def f_b(x):
        return math.log(x) - 2 ** x + x ** 2 - 1

    tol = 1e-6
    lista_p = bissecao_sequencia(f_b, a=3, b=5, tol=tol)

    print("=" * 55)
    print("f(x) = ln(x) - 2^x + x^2 - 1,  intervalo [3, 5]")
    print("=" * 55)
    for i, p in enumerate(lista_p, start=1):
        print(f"  p{i:<3d} = {p:.10f}   f(p{i}) = {f_b(p): .6e}")
    print("-" * 55)
    print(f"Raiz aproximada : {lista_p[-1]:.10f}")
    print(f"Numero de iteracoes: {len(lista_p)}")
    print("=" * 55)

    fig_b = plotar_convergencia(f_b, a=3, b=5, n=200, lista_p=lista_p,
                                 titulo="f(x) = ln(x) - 2^x + x^2 - 1  e  sequencia p_n")
    fig_b.savefig("questao4b_convergencia.png", dpi=150, bbox_inches="tight")
    print("Grafico da letra b) salvo em questao4b_convergencia.png")

# ---------------------------------------------------------------------------
# Resultados numericos obtidos (saida da execucao acima):
#
# f(x) = ln(x) - 2^x + x^2 - 1,  intervalo [3, 5]
#
#   p1   = 4.0000000000   f(p1)  =  3.862944e-01
#   p2   = 4.5000000000   f(p2)  = -1.873340e+00
#   p3   = 4.2500000000   f(p3)  = -5.178949e-01
#   p4   = 4.1250000000   f(p4)  = -1.543270e-02
#   p5   = 4.0625000000   f(p5)  =  1.973243e-01
#   p6   = 4.0937500000   f(p6)  =  9.400407e-02
#   p7   = 4.1093750000   f(p7)  =  4.006108e-02
#   p8   = 4.1171875000   f(p8)  =  1.250940e-02
#   p9   = 4.1210937500   f(p9)  = -1.412677e-03
#   p10  = 4.1191406250   f(p10) =  5.560583e-03
#   p11  = 4.1201171875   f(p11) =  2.077011e-03
#   p12  = 4.1206054688   f(p12) =  3.329321e-04
#   p13  = 4.1208496094   f(p13) = -5.396811e-04
#   p14  = 4.1207275391   f(p14) = -1.033267e-04
#   p15  = 4.1206665039   f(p15) =  1.148147e-04
#   p16  = 4.1206970215   f(p16) =  5.746984e-06
#   p17  = 4.1207122803   f(p17) = -4.878910e-05
#   p18  = 4.1207046509   f(p18) = -2.152087e-05
#   p19  = 4.1207008362   f(p19) = -7.886898e-06
#   p20  = 4.1206989288   f(p20) = -1.069945e-06
#   p21  = 4.1206979752   f(p21) =  2.338523e-06
#
# Raiz aproximada: 4.1206979752
# Numero de iteracoes: 21
# (Graficos salvos em questao4a_validacao.png e questao4b_convergencia.png)