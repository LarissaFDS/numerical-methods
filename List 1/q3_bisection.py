def exponencial_manual(x, termos=50):
    """
    Calcula e^x utilizando a expansão em série de Maclaurin:
    e^x = 1 + x + x^2/2! + x^3/3! + ...
    """
    soma = 1.0
    termo_atual = 1.0
    for n in range(1, termos):
        termo_atual = termo_atual * x / n
        soma += termo_atual
    return soma

def f(t):
    """
    Função baseada na equação: 22 + (85 - 22)*e^(-0.25*t) = 35
    Isolando tudo em um lado: f(t) = 63 * e^(-0.25 * t) - 13
    """
    return 63 * exponencial_manual(-0.25 * t) - 13

def bissecao(f, a, b, tol):
    """
    Método da bisseção.
    Retorna a raiz aproximada ou None caso f(a) * f(b) > 0.
    Critério de parada: |p_n - p_{n-1}| < tol
    """
    if f(a) * f(b) > 0:
        return None
        
    #Casos triviais
    if f(a) == 0.0: return a
    if f(b) == 0.0: return b
    
    p_prev = a  #Valor inicial para o cálculo de |p_n - p_{n-1}|
    
    while True:
        p_n = (a + b) / 2.0
        
        #Critério de parada: diferença entre aproximações consecutivas
        if abs(p_n - p_prev) < tol:
            return p_n
            
        #Caso encontre a raiz exata
        if f(p_n) == 0.0:
            return p_n
            
        #Atualização do intervalo
        if f(a) * f(p_n) < 0:
            b = p_n
        else:
            a = p_n
            
        p_prev = p_n

def busca_intervalo_e_raiz(f, a_geral, b_geral, C, tol):
    """
    Divide o intervalo [a_geral, b_geral] em subintervalos de comprimento C.
    Encontra o primeiro onde ocorre mudança de sinal e aplica a Bisseção.
    """
    inicio = a_geral
    
    while inicio < b_geral:
        fim = min(inicio + C, b_geral)
        
        #Verifica se há mudança de sinal no subintervalo atual
        if f(inicio) * f(fim) <= 0:
            raiz = bissecao(f, inicio, fim, tol)
            return (inicio, fim), raiz
            
        inicio += C
        
    return None, None


if __name__ == '__main__':
    a = 0
    b = 30
    C = 3
    tol = 1e-6
    
    print("=" * 70)
    print("RESOLUÇÃO QUESTÃO 3 - MÉTODO DA BISSEÇÃO (EXPONENCIAL MANUAL)")
    print("=" * 70)
    print("Equação a ser resolvida: f(t) = 63 * e^(-0.25*t) - 13 = 0")
    print(f"Parâmetros da busca: intervalo geral [{a}, {b}], passo C={C}, tol={tol}")
    print("-" * 70)
    
    intervalo, raiz = busca_intervalo_e_raiz(f, a, b, C, tol)
    
    if raiz is not None:
        print(f"Mudança de sinal encontrada no subintervalo: [{intervalo[0]}, {intervalo[1]}]")
        print(f"Tempo calculado (raiz t): {raiz:.6f}")
        print(f"Verificação: f({raiz:.6f}) = {f(raiz):.6e}")
    else:
        print("Nenhuma raiz encontrada no intervalo especificado.")
    print("=" * 70)
