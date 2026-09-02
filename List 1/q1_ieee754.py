def resolver_ieee754(entrada_binaria):
    #Limpar espaços/quebras de linha e preencher com zeros à direita
    str_limpa = entrada_binaria.replace(" ", "").replace("\n", "")
    str_64bits = str_limpa.ljust(64, '0')
    
    #Extrair sinal, expoente e mantissa
    bit_sinal = str_64bits[0]
    bits_expoente = str_64bits[1:12]
    bits_mantissa = str_64bits[12:64]
    
    #Converter sinal
    s = int(bit_sinal)
    
    #Calcular expoente (e) iterativamente
    e = 0
    for i, bit in enumerate(reversed(bits_expoente)):
        if bit == '1':
            e += 2**i
            
    #Calcular o valor fracionário da mantissa (f)
    f = 0.0
    for i, bit in enumerate(bits_mantissa):
        if bit == '1':
            f += 2**(-(i + 1))
            
    forma_matematica = f"(-1)^{s} * (1.{bits_mantissa})_2 * 2^({e}-1023)"
    
    #Calcular valor decimal
    #(-1)^s * (1 + f) * 2^(e - 1023)
    valor_mantissa = 1.0 + f
    multiplicador = 2**(e - 1023)
    
    sinal_mult = -1 if s == 1 else 1
    valor_decimal = sinal_mult * valor_mantissa * multiplicador
    
    print(f"Sequência original: {entrada_binaria}")
    print(f"String 64 bits:   {str_64bits}")
    print(f"Sinal (s):        {bit_sinal} -> {s}")
    print(f"Expoente (e):     {bits_expoente} -> {e}")
    print(f"Fração/Mantissa:  {bits_mantissa}")
    print(f"Forma matemática: {forma_matematica}")
    print(f"Valor Decimal:    {valor_decimal}")
    print("-" * 70)


if __name__ == '__main__':
    entradas = [
        "0 10000000101 011010010000",
        "1 10000000101 01101001000000000000000 000000 000000000",
        "0 10000000000 0011010100000000000000000000000000000000000000000000",
        "0 10000000000 001101010000 000001"
    ]
    
    print("=" * 70)
    print("RESOLUÇÃO QUESTÃO 1 - PADRÃO IEEE-754 (64 BITS)")
    print("=" * 70)
    for letra, sequencia in zip(['a', 'b', 'c', 'd'], entradas):
        print(f"\nItem {letra})")
        resolver_ieee754(sequencia)
