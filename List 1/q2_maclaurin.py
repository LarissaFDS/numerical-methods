from decimal import Decimal, localcontext, ROUND_HALF_EVEN

K4 = 4
ALTA = 40


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


def referencia_seno(x):
    with localcontext() as ctx:
        ctx.prec = ALTA

        x = Decimal(x)
        termo = x
        soma = x
        n = 1

        for _ in range(50):
            termo = -termo * x * x / Decimal((2*n)*(2*n+1))
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

        z2 = z * z
        z4 = z2 * z2
        z6 = z4 * z2
        z8 = z4 * z4

        resultado = (
            Decimal("1")
            - z2 * A
            + z4 * B
            - z6 * C
            + z8 * D
        )

        return resultado, 12


def aninhado(z):
    with localcontext() as ctx:
        ctx.prec = K4
        ctx.rounding = ROUND_HALF_EVEN

        y = z * z
        resultado = (
            Decimal("1")
            + y * (
                -A
                + y * (
                    B
                    + y * (
                        -C
                        + y * D
                    )
                )
            )
        )

        return resultado, 9


if __name__ == "__main__":

    print("QUESTAO 2")

    print("\na)")
    print("P8(x) = 1 - x^2/2 + x^4/24 - x^6/720 + x^8/40320")

    # sen(0,5) = cos(pi/2 - 0,5)
    pi = calcular_pi()

    with localcontext() as ctx:
        ctx.prec = K4
        ctx.rounding = ROUND_HALF_EVEN

        pi_k4 = +pi
        metade_pi = pi_k4 / Decimal("2")
        z = metade_pi - Decimal("0.5")

    print("\nArgumento usado:")
    print("sen(0,5) = cos(pi/2 - 0,5)")
    print(f"pi com k=4 = {pi_k4}")
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
    ref = referencia_seno("0.5")

    erro_nao = abs(Decimal(str(r_nao)) - ref)
    erro_ani = abs(Decimal(str(r_ani)) - ref)

    print(f"Referencia sen(0,5) = {ref}")
    print(f"Nao aninhado: {r_nao} | erro = {erro_nao:.4e} | {op_nao} ops")
    print(f"Aninhado:     {r_ani} | erro = {erro_ani:.4e} | {op_ani} ops")