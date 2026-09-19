from decimal import Decimal, localcontext, ROUND_HALF_EVEN

K4 = 4
ALTA = 40

class ContaDecimal:
    contador = 0

    def __init__(self, valor):
        self.valor = valor if isinstance(valor, Decimal) else Decimal(valor)

    def _bin_op(self, outro, func):
        ContaDecimal.contador += 1
        outro_val = outro.valor if isinstance(outro, ContaDecimal) else Decimal(outro)
        return ContaDecimal(func(self.valor, outro_val))

    def __mul__(self, outro):
        return self._bin_op(outro, lambda a, b: a * b)
    __rmul__ = __mul__

    def __add__(self, outro):
        return self._bin_op(outro, lambda a, b: a + b)
    __radd__ = __add__

    def __sub__(self, outro):
        return self._bin_op(outro, lambda a, b: a - b)

    def __rsub__(self, outro):
        ContaDecimal.contador += 1
        outro_val = outro.valor if isinstance(outro, ContaDecimal) else Decimal(outro)
        return ContaDecimal(outro_val - self.valor)

    def __repr__(self):
        return repr(self.valor)


def arctan(x):
    with localcontext() as ctx:
        ctx.prec = ALTA
        s = Decimal("0")
        pot = x
        x2 = x * x
        sinal = 1
        n = 1
        for _ in range(200):
            termo = pot / Decimal(n)
            if sinal > 0:
                s += termo
            else:
                s -= termo
            if abs(termo) < Decimal("1e-35"):
                break
            pot *= x2
            n += 2
            sinal *= -1
        return s


def calcular_pi():
    with localcontext() as ctx:
        ctx.prec = ALTA
        a = arctan(Decimal("1") / Decimal("5"))
        b = arctan(Decimal("1") / Decimal("239"))
        return Decimal("16") * a - Decimal("4") * b


def referencia_cosseno(x):
    with localcontext() as ctx:
        ctx.prec = ALTA
        x = Decimal(x)
        termo = Decimal("1")
        soma = Decimal("1")
        n = 1
        for _ in range(50):
            # Série de Taylor para cosseno: 1 - x^2/2! + x^4/4! - ...
            termo = -termo * x * x / Decimal((2*n-1)*(2*n))
            soma += termo
            if abs(termo) < Decimal("1e-35"):
                break
            n += 1
        return soma


def coeficientes():
    with localcontext() as ctx:
        ctx.prec = K4
        ctx.rounding = ROUND_HALF_EVEN
        return (
            Decimal("1") / Decimal("2"),
            Decimal("1") / Decimal("24"),
            Decimal("1") / Decimal("720"),
            Decimal("1") / Decimal("40320")
        )


A, B, C, D = coeficientes()


def nao_aninhado(z):
    with localcontext() as ctx:
        ctx.prec = K4
        ctx.rounding = ROUND_HALF_EVEN

        ContaDecimal.contador = 0
        zc = ContaDecimal(z)
        Ac, Bc, Cc, Dc = ContaDecimal(A), ContaDecimal(B), ContaDecimal(C), ContaDecimal(D)

        z2 = zc * zc
        z4 = z2 * z2
        z6 = z4 * z2
        z8 = z4 * z4

        resultado = ContaDecimal(Decimal("1")) - z2 * Ac + z4 * Bc - z6 * Cc + z8 * Dc

        return resultado.valor, ContaDecimal.contador


def aninhado(z):
    with localcontext() as ctx:
        ctx.prec = K4
        ctx.rounding = ROUND_HALF_EVEN

        ContaDecimal.contador = 0
        y = ContaDecimal(z) * ContaDecimal(z)

        # Coeficientes na ordem do maior grau (em y) até o termo constante
        coefs_horner = [D, -C, B, -A, Decimal("1")]

        resultado = ContaDecimal(coefs_horner[0])
        for c in coefs_horner[1:]:
            resultado = resultado * y + ContaDecimal(c)

        return resultado.valor, ContaDecimal.contador


if __name__ == "__main__":

    print("QUESTAO 2")

    print("\na)")
    print("P8(x) = 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320")

    with localcontext() as ctx:
        ctx.prec = K4
        ctx.rounding = ROUND_HALF_EVEN
        z = Decimal("0.5")

    print("\nArgumento usado:")
    print("cos(0,5)")
    print(f"z = {z}")

    print("\nb)")
    r_nao, op_nao = nao_aninhado(z)
    print(f"Resultado nao aninhado = {r_nao}")
    print(f"Operacoes = {op_nao}")

    print("\nc)")
    print("Com y = z^2:")
    print("P8(z) = 1 + y(-1/2 + y(1/24 + y(-1/720 + y/40320)))")

    print("\nd)")
    r_ani, op_ani = aninhado(z)
    print(f"Resultado aninhado = {r_ani}")
    print(f"Operacoes = {op_ani}")

    print("\ne)")
    ref = referencia_cosseno("0.5")

    erro_nao = abs(Decimal(str(r_nao)) - ref)
    erro_ani = abs(Decimal(str(r_ani)) - ref)

    print(f"Referencia cos(0,5) = {ref}")
    print(f"Nao aninhado: {r_nao} | erro = {erro_nao:.4e} | {op_nao} ops")
    print(f"Aninhado:     {r_ani} | erro = {erro_ani:.4e} | {op_ani} ops")