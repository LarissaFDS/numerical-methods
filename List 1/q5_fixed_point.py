def g1(x):
    """
    Função de iteração g1(x) = 1 / (1 + x)
    """
    return 1.0 / (1.0 + x)

def g2(x):
    """
    Função de iteração g2(x) = 1 - (1 / x)
    """
    return 1.0 - (1.0 / x)

def ponto_fixo(g, x0, tol, Nmax):
    """
    Método de iteração de ponto fixo.
    Critério de parada: |x_new - x_old| < tol
    """
    x_old = x0
    
    for i in range(1, Nmax + 1):
        try:
            x_new = g(x_old)
        except ZeroDivisionError:
            #Tratamento de exceção para divisão por zero (especialmente para g2)
            return None, i, "Erro de convergência: divisão por zero."
            
        #Critério de parada
        if abs(x_new - x_old) < tol:
            return x_new, i, "Sucesso (Convergiu)"
            
        x_old = x_new
        
    return None, Nmax, "Falha"


if __name__ == '__main__':
    x0 = 0.5
    tol = 1e-6
    Nmax = 100
    
    print("=" * 70)
    print("RESOLUÇÃO QUESTÃO 5 - ITERAÇÃO DE PONTO FIXO")
    print("=" * 70)
    print(f"Equação original: x^2 + x - 1 = 0")
    print(f"Parâmetros: x0 = {x0}, tol = {tol}, Nmax = {Nmax}")
    print("-" * 70)
    
    print("-> Testando g1(x) = 1 / (1 + x)")
    raiz_g1, iter_g1, status_g1 = ponto_fixo(g1, x0, tol, Nmax)
    
    print(f"Status: {status_g1}")
    if raiz_g1 is not None:
        print(f"Raiz encontrada: {raiz_g1:.6f}")
    print(f"Iterações realizadas: {iter_g1}")
    
    print("-" * 70)
    
    print("-> Testando g2(x) = 1 - (1 / x)")
    raiz_g2, iter_g2, status_g2 = ponto_fixo(g2, x0, tol, Nmax)
    
    print(f"Status: {status_g2}")
    if raiz_g2 is not None:
        print(f"Raiz encontrada: {raiz_g2:.6f}")
    print(f"Iterações realizadas: {iter_g2}")
    
    print("=" * 70)
