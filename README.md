# Métodos Numéricos

Repositório dedicado às resoluções de listas de exercícios, trabalhos e implementações computacionais da disciplina de **Métodos Numéricos**.

## 👥 Alunos
- Otávio Joshua Costa Brandão Menezes
- Larissa Ferreira Dias de Souza

## 👨‍🏫 Professor
- Thales Miranda de Almeida Vieira

## 📂 Estrutura do Repositório

Este repositório está organizado em diretórios correspondentes às listas de exercícios e atividades passadas ao longo do curso:

* **[List 1](./List%201/)**
  * Implementações de algoritmos numéricos básicos, incluindo representação IEEE 754, série de Maclaurin, busca de raízes (Método da Bisseção, Ponto Fixo, Newton) e rotinas para plotagem de funções e convergência.
* **[List 2](./List%202/)**
  * Gerador de matrizes positivas definidas, fatoração PTLU, eliminação gaussiana e método de Cholesky adaptado aplicado a problemas térmicos de equilíbrio em placa circular.
* **[List 3](./List%203/)**
  * **Questão 1**: Sistema massa-mola em equilíbrio resolvido por Gauss-Seidel, análise de convergência via raio espectral $\rho(T) = 1/3 < 1$, norma $\|T\|_\infty = 2/3 < 1$ e Critério de Sylvester (Ostrowski-Reich).
  * **Questão 2**: Calibração de reservatório via ajuste por Mínimos Quadrados $p_2(h)$, Interpolação de Lagrange $p_4(h)$ e Método de Neville com tabela triangular para $h = 1{,}25$ m.
  * **Questão 3**: Análise de sinal transitório de corrente $I(t) = e^{-t^2}\sin(t^3)$, diferenciação numérica por diferenças finitas de 3 pontos ($t_{\max}$ para taxa máxima) e cálculo de carga transferida $Q$ via Regra Composta de Simpson.

## 💻 Execução e Testes

Os códigos foram desenvolvidos em **Python 3** sem o uso de bibliotecas prontas de álgebra linear ou diferenciação/integração numérica (apenas rotinas nativas, `fractions.Fraction` e `SymPy` para parsing/avaliação simbólica básica).

Para executar as questões da Lista 3:
```bash
python "List 3/questao1.py"
python "List 3/questao2.py"
python "List 3/questao3.py"
```

Para rodar a suíte completa de testes automatizados com `pytest`:
```bash
pytest -v
```

---
*Repositório mantido para fins acadêmicos e de registro do aprendizado na disciplina.*
