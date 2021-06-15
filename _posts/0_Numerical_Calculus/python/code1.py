

# O hashtag é usado para comentar alguma linha do seu código

# ================================================================
#		Trabalhando com Strings (str)
# ================================================================

print("Hello World!!")

# Variável recebendo um string:
var_texto = "Um texto"
print("Agora irei imprimir a variável de texto \033[1m var_text \033[0m na linha abaixo: \n %s" % var_texto)

# String em Negrito:
print("Este texto não contêm formatação")
print("\033[1m Este texto está em negrito \033[0m")

# Inserindo Tabulação:
print("Vamos adicioanr uma tabulação entre A e B:\n A \t B")

# Operação com Stings:
A = "Texto 1"
B = " + Texto 2"
C = A + B
print("Estou somando as variáveis de texto A e B:")
print("Impressão Simples:")
print("A + B = ", A + B)

print("Impresão usando o símbolo de texto \033[1m %s \033[0m:")
print("A + B = %s" % (A+B))

# ================================================================
#		Trabalhando com Inteiros e Reais (int, float)
# ================================================================

A = 1
B = 3.14
print("Vamos perguntar agora ao programa qual é o tipo de cada variável acima através",
      "do comando\033[1m type() \033[0m:")

type(A) # type() usado para identificar o tipo de uma variável
type(B)

# Operações entre int's e float's:
A, B, C = 3, 2, 3.14
print("A = ", A)
print("B = ", B)
print("C = ", C)

type(A), type(B), type(C)
type(A+B), type(A-B), type(A*B), type(A/B), type(B%A)
A+B, A-B, A*B, A/B, B%A
