"""
Testes automatizados para a Lista 3 de Métodos Numéricos.

Cobre a verificação matemática, numérica e comportamental de:
  * Questão 1 (sistema linear massa-mola, propriedades de A, matriz T, Gauss-Seidel)
  * Questão 2 (equações normais, mínimos quadrados, Lagrange, método de Neville)
  * Questão 3 (amostragem simbólica/numérica, diferenças de 3 pontos, regra de Simpson)
"""

from fractions import Fraction as Fr
import math
import os
import sys
import pytest

# Adiciona o diretório da Lista 3 ao path para importar as questões
DIRETORIO_LISTA3 = os.path.dirname(os.path.abspath(__file__))
if DIRETORIO_LISTA3 not in sys.path:
    sys.path.insert(0, DIRETORIO_LISTA3)

from questao1 import (
    montar_sistema,
    estritamente_diagonal_dominante,
    eh_simetrica,
    menores_principais,
    matriz_iteracao_gauss_seidel,
    norma_inf_matriz,
    polinomio_caracteristico,
    gauss_seidel,
)
from questao2 import (
    H,
    V,
    H_ALVO,
    equacoes_normais,
    eliminacao_gauss,
    avaliar_poli,
    base_lagrange,
    lagrange,
    neville,
)
from questao3 import (
    FUNCAO_STR,
    amostrar,
    derivada_3_pontos,
    simpson_composta,
    funcao_corrente,
)


# =============================================================================
# TESTES DA QUESTÃO 1
# =============================================================================
def test_q1_montagem_sistema():
    """Verifica se a montagem do sistema Ax = b corresponde ao balanço de forças físico."""
    k = [200, 100, 100, 200]
    f = [10, 20, 10]
    A, b = montar_sistema(k, f)

    assert len(A) == 3 and len(A[0]) == 3
    assert len(b) == 3

    # Linha 1: 300 x1 - 100 x2 = 10
    assert A[0] == [Fr(300), Fr(-100), Fr(0)]
    assert b[0] == Fr(10)

    # Linha 2: -100 x1 + 200 x2 - 100 x3 = 20
    assert A[1] == [Fr(-100), Fr(200), Fr(-100)]
    assert b[1] == Fr(20)

    # Linha 3: -100 x2 + 300 x3 = 10
    assert A[2] == [Fr(0), Fr(-100), Fr(300)]
    assert b[2] == Fr(10)


def test_q1_propriedades_matriz_A():
    """Verifica dominância estrita (falsa), simetria e positividade definida."""
    k = [200, 100, 100, 200]
    f = [10, 20, 10]
    A, _ = montar_sistema(k, f)

    # A não é estritamente diagonal dominante pois linha 2 tem |200| = |-100| + |-100|
    sdd, linhas = estritamente_diagonal_dominante(A)
    assert sdd is False
    assert linhas[1][2] is False

    # A é simétrica
    assert eh_simetrica(A) is True

    # Menores principais líderes det(A_1)=300, det(A_2)=50000, det(A_3)=12000000 > 0
    dets = menores_principais(A)
    assert dets == [Fr(300), Fr(50000), Fr(12000000)]
    assert all(d > 0 for d in dets)


def test_q1_matriz_iteracao_T_e_c():
    """Verifica a matriz de iteração T, vetor c, ||T||_inf e autovalores analíticos."""
    k = [200, 100, 100, 200]
    f = [10, 20, 10]
    A, b = montar_sistema(k, f)

    T, c = matriz_iteracao_gauss_seidel(A, b)

    # T esperada
    assert T[0] == [Fr(0), Fr(1, 3), Fr(0)]
    assert T[1] == [Fr(0), Fr(1, 6), Fr(1, 2)]
    assert T[2] == [Fr(0), Fr(1, 18), Fr(1, 6)]

    # c esperado
    assert c == [Fr(1, 30), Fr(7, 60), Fr(13, 180)]

    # ||T||_inf = 2/3 < 1
    assert norma_inf_matriz(T) == Fr(2, 3)

    # Polinômio característico: lambda^3 - (1/3) lambda^2 -> autovalores 0 (duplo) e 1/3
    coef = polinomio_caracteristico(T)
    assert coef[0] == Fr(1)
    assert coef[1] == Fr(-1, 3)
    assert coef[2] == Fr(0)
    assert coef[3] == Fr(0)


def test_q1_gauss_seidel_convergencia():
    """Verifica a convergência do método de Gauss-Seidel para a solução exata."""
    k = [200, 100, 100, 200]
    f = [10, 20, 10]
    A, b = montar_sistema(k, f)

    x, it, conv, hist = gauss_seidel(A, b, [0, 0, 0], tol=1e-5, max_iter=1000)

    assert conv is True
    assert it == 10
    # Solução analítica exata é (0.1, 0.2, 0.1)
    assert abs(x[0] - 0.1) < 1e-5
    assert abs(x[1] - 0.2) < 1e-5
    assert abs(x[2] - 0.1) < 1e-5


# =============================================================================
# TESTES DA QUESTÃO 2
# =============================================================================
def test_q2_equacoes_normais_e_ajuste():
    """Verifica o sistema das equações normais e coeficientes do polinômio p2(h)."""
    A, b = equacoes_normais(H, V, 2)

    # Sistema 3x3 com frações
    assert len(A) == 3 and len(b) == 3
    assert A[0][0] == Fr(5)
    assert A[0][1] == Fr(5)
    assert A[0][2] == Fr(15, 2)

    coef = eliminacao_gauss(A, b)
    # Coeficientes analíticos exatos: a0 = 10, a1 = 918, a2 = 540
    assert coef == [Fr(10), Fr(918), Fr(540)]

    # Estimativa em h = 1.25 m: p2(1.25) = 2001.25 L
    v_est = avaliar_poli(coef, H_ALVO)
    assert v_est == Fr(8005, 4)
    assert float(v_est) == 2001.25


def test_q2_lagrange_interpolacao():
    """Verifica a propriedade interpoladora e a estimativa de Lagrange p4(1.25)."""
    # p4(h_i) deve ser exatamente igual a V_i para todos os pontos da tabela
    for hi, vi in zip(H, V):
        assert lagrange(H, V, hi) == vi

    # Estimativa de Lagrange em h = 1.25 m
    v_lagrange = lagrange(H, V, H_ALVO)
    assert v_lagrange == Fr(63725, 32)
    assert float(v_lagrange) == 1991.40625


def test_q2_metodo_neville():
    """Verifica se o método de Neville reproduz exatamente a interpolação de Lagrange."""
    vn, Q = neville(H, V, H_ALVO)
    assert vn == Fr(63725, 32)
    assert vn == lagrange(H, V, H_ALVO)

    # Verifica a primeira coluna da tabela triangular de Neville (Q_{i,0} == V_i)
    for i in range(5):
        assert Q[i][0] == V[i]


# =============================================================================
# TESTES DA QUESTÃO 3
# =============================================================================
def test_q3_amostragem():
    """Verifica a geração das amostras de tempo e corrente."""
    t200, I200 = amostrar(FUNCAO_STR, 0.0, 3.0, 200)

    assert len(t200) == 200
    assert len(I200) == 200
    assert t200[0] == 0.0
    assert pytest.approx(t200[-1]) == 3.0
    assert pytest.approx(I200[0]) == 0.0

    # Amostragem com n inválido deve lançar ValueError
    with pytest.raises(ValueError):
        amostrar(FUNCAO_STR, 0.0, 3.0, 1)


def test_q3_diferencas_3_pontos():
    """Verifica a derivada numérica de 3 pontos em n = 15."""
    t15, I15 = amostrar(FUNCAO_STR, 0.0, 3.0, 15)
    d15 = derivada_3_pontos(t15, I15)

    assert len(d15) == 15

    # Determinação do t_max numérico entre as 15 amostras
    k_max = max(range(len(d15)), key=lambda i: d15[i])
    assert k_max == 3
    assert pytest.approx(t15[k_max]) == 9.0 / 14.0
    assert d15[k_max] > 0.50

    # Menos de 3 pontos deve lançar ValueError
    with pytest.raises(ValueError):
        derivada_3_pontos([0.0, 1.0], [0.0, 1.0])


def test_q3_simpson_composta():
    """Verifica a regra de Simpson Composta para cálculo da carga Q."""
    f = funcao_corrente(FUNCAO_STR)

    # n ímpar deve lançar ValueError
    with pytest.raises(ValueError):
        simpson_composta(f, 0.0, 3.0, 3)

    # Teste de precisão com polinômio de grau 3 (Simpson composta é exata para grau <= 3)
    p3 = lambda x: 4.0 * (x ** 3) - 3.0 * (x ** 2) + 2.0 * x + 1.0
    # Integral de 0 a 2: [x^4 - x^3 + x^2 + x]_0^2 = 16 - 8 + 4 + 2 = 14.0
    assert pytest.approx(simpson_composta(p3, 0.0, 2.0, 4)) == 14.0

    # Valor da integral de I(t) em [0, 3] para n = 50
    q50 = simpson_composta(f, 0.0, 3.0, 50)
    assert 0.200 < q50 < 0.201
