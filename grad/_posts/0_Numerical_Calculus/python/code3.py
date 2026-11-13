

# ================================================================
#		Listas
# ================================================================
print("\n# ================================================================")
print("# \t Listas")
print("# ================================================================\n")

A = ["texto", 1, 3.45, True, 7, 8, 3.14] # A nossa lista pode ter qualquer nome. Aqui escolhemos A.
type(A) # Vamos confirmar que nossa variável chamada A recebeu uma lista

for a in A: # Uma outra forma de usar o laço for sem utilizar o "range()"
    print("Elemento (",a,")\t tipo de variável: ", type(a),"\n") # A cada execução "a" receberá um valor da lista.
    
print("\033[1m", A[0], "\033[0m é o primeiro elemento da nossa lista.")

print("\033[1m", A[1], "\033[0m é o segundo elemento da nossa lista.")

print("\033[1m", A[4], "\033[0m é o quinto elemento da nossa lista.")

N = len(A) # o comando len() retorna um número inteiro.
print("A nossa lista A possui \033[1m", N, "\033[0m elementos.")

# ================================================================
#		Operações com Listas
# ================================================================
print("\n# ================================================================")
print("# \t Operações com Listas")
print("# ================================================================\n")

B = ["b1", "b2", "b3"]
C1 = A + B
print("Nova lista:", C1)

A.append(B)
print("Nova lista:", A)

Lista = ["a1", "a2", "a3"]
Lista2 = Lista*2
Lista3 = Lista*3

print("Lista inicial: ", Lista)

Lista = [1,2,3,4,5]
print("Lista inicial: ", Lista)
print("Número de elementos da Lista: ", len(Lista))

del(Lista[3]) # deletando o quarto elemento da lista
print("\nLista final: ", Lista)
print("Número de elementos da Lista: ", len(Lista))
print("Lista dobrada: ", Lista2)
print("Lista triplicada: ", Lista3)

Lista = [1,2,3,4,5]
print("Lista inicial: ", Lista)
print("Número de elementos da Lista: ", len(Lista))

del(Lista[3]) # deletando o quarto elemento da lista
print("\nLista final: ", Lista)
print("Número de elementos da Lista: ", len(Lista))

# ================================================================
#		O comando range()
# ================================================================
print("\n# ================================================================")
print("# \t O comando range")
print("# ================================================================\n")

conj = range(5) # conjunto dos números inteiros não-negativos menores do que 5
for a in conj:
    print(a)

print("\nNúmero de elementos da nossa sequência:", len(conj))

conj = range(5) # conjunto dos números inteiros não-negativos menores do que 5
n = len(conj) # retorna o número de elementos de "conj"
for i in range(n):
    print(conj[i])

print("\nNúmero de elementos da nossa sequência:", len(conj))

type(conj)

nova_lista = list(conj)
"nova_lista é um objeto do tipo: " + str(type(nova_lista))

# range( valor inicial , valor final )
Conj = range(5,10) # conjunto dos números inteiros maiores ou iguas a 5 e menores do que 10
for a in Conj:
    print(a)

print("\nNúmero de elementos da nossa sequência:", len(Conj))

# range( valor inicial, valor final, intervalo entre elementos)
Conj = range(0,10,2) # conjunto dos números inteiros-não negativos pares menores do que 10:
for a in Conj: 
    print(a)
    
Conj = range(1,10,2) # conjunto dos números inteiros-não negativos ímpares menores do que 10:
for a in Conj: 
    print(a)
    
# ================================================================
#		Biblioteca Numpy
# ================================================================
print("\n# ================================================================")
print("# \t Biblioteca Numpy")
print("# ================================================================\n")

import numpy # sem simplificação
pi = numpy.pi
print("O número pi é igual a:",pi)

import numpy as np # com simplificação
pi = np.pi
print("O número pi é igual a:", pi)

# ================================================================
#		Vetores
# ================================================================
print("\n# ================================================================")
print("# \t Vetores")
print("# ================================================================\n")

import numpy

Conj = range(5)
Lista = [1,2,3,4,5]
vetor = numpy.array(Lista)
print("Tipo da variável (Conj): ", type(Conj))
print("Tipo da variável (Lista): ", type(Lista))
print("Tipo da variável (vetor): ", type(vetor))

# ================================================================
#		Operações com Vetores
# ================================================================
print("\n# ================================================================")
print("# \t Operações com Vetores")
print("# ================================================================\n")

import numpy as np

A = np.array([1,2,3,4,5])
B = np.array([6,7,8,9,10])
x = 3.14

print("A + B =", A + B)
print("A - B =", A - B)
print("A * B =", A * B)
print("A / B =", A / B)

print("A + x =", A + x)
print("A - x =", A - x)
print("A * x =", A * x)
print("A / x =", A / x)

# ================================================================
#		Comando numpy.linspace()
# ================================================================
print("\n# ================================================================")
print("# \t Comando numpy.linspace()")
print("# ================================================================\n")

v_inicial = 1 # O primeiro elemento do vetor;
v_final = 10 # O último elemento do vetor;
n_elementos = 10

Vetor = np.linspace(v_inicial, v_final, n_elementos) # ou simplesmente np.linspace(1,10,10)
print("Nosso vetor usando \033[1m linspace \033[0m é: Vetor =", Vetor)

# ================================================================
#		Comando numpy.arange()
# ================================================================
print("\n# ================================================================")
print("# \t Comando numpy.arange()")
print("# ================================================================\n")

v_inicial = 1
v_final = 20 # Nesse caso o último elemento do vetor será menor do que v_final!
v_intervalo = 3

Vetor = np.arange(v_inicial, v_final, v_intervalo) # ou simplesmente np.arange(1,20,3)
print("Nosso vetor usando \033[1m arange \033[0m é: Vetor =", Vetor)

# ================================================================
#		Valor Máximo e Mínimo de um Vetor/Lista
# ================================================================
print("\n# ================================================================")
print("# \t Valor Máximo e Mínim de um Vetor/Lista")
print("# ================================================================\n")

A = [1,2,3,4,5]
B = np.linspace(10,15,5)

MaxA = max(A) ; MinA = min(A)
MaxB = max(B) ; MinB = min(B)

print("Os valores máximo e mínimo da lista A são: max = %f \t min = %f" % (MaxA,MinA))
print("Os valores máximo e mínimo do vetor B são: max = %f \t min = %f" % (MaxB,MinB))

# ================================================================
#		Somatório de todos os elementos de um Vetor/Lista
# ================================================================
print("\n# ================================================================")
print("# \t Somatório de todos os elementos de um Vetor/Lista")
print("# ================================================================\n")

V = [1,2,3,4]

sum1 = 0
for i in range(len(V)):
    sum1 = sum1 + V[i]

print("A soma de todos os elementos de V é: Soma =", sum1)

sum2 = sum(V)
print("A soma de todos os elementos de V é: Soma =", sum2)

# ================================================================
#		Matrizes
# ================================================================
print("\n# ================================================================")
print("# \t Matrizes")
print("# ================================================================\n")

M = [[1,2,3], [4,5,6], [7,8,9]]

print("M[0] =",M[0])
print("M[1] =",M[1])
print("M[2] =",M[2])

print("M[0][0] =",M[0][0])
print("M[0][1] =",M[0][1])
print("M[0][2] =",M[0][2])

print("M[1][0] =",M[1][0])
print("M[1][1] =",M[1][1])
print("M[1][2] =",M[1][2])

print("M[2][0] =",M[2][0])
print("M[2][1] =",M[2][1])
print("M[2][2] =",M[2][2])
