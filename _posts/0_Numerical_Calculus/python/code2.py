

# ================================================================
#		Variáveis Boolenas (bool)
# ================================================================

A = 5 ; B = 6 # usamos o ; para não precisar digitar (B=6) em outra linha.
print(A==B, A!=B, A>B, A>=B, A<B, A<=B)

var = A==B # var recebeu o valor (A==B)
type(var) # veremos que var recebeu uma variável booleana

# ================================================================
#		Operadores: and, or, not
# ================================================================

True and True
True and False # mesmo que (False and True)
False and False

True or True
True or False # mesmo que (False or True)
False or False

not True
not False

True and (False or True)
(True and False) and True 

# ================================================================
#		Comando: if()
# ================================================================

A = 5
B = 6
C = "texto"

if(A==B): # Aqui a condição é A==B
    print("A é igual a B") # esse comando será executado apenas se a condição do if for True
else:
    print("A é diferente de B") # esse comando será executado apenas se a condição do if for False
    
if(C=="texto" or C=="palavra"): # Aqui a condição é (C=="texto" or C=="palavra")
    print("A condição acima foi True: \t C = %s" % C)
    
if(not (A>B)):
    print("A não é maior do que B")
else:
    print("A é maior do que B")
    
if(not (A<B)):
    print("A não é menor do que B")
else:
    print("A é menor do que B")
    
# ================================================================
#		Comando: while()
# ================================================================

x = 1
while(x<5): # Os comando abaixo serão executados 4 vezes até x ser igual ou maior do que 5.
    print("x = ", x) # A tabulção também é necessária no uso do while! 
    x = x+1 
    
# ================================================================
#		Comando: for()
# ================================================================

# Exemplo de contador:
for i in range(10): # Aqui "i" começa com o valor 0. Esse valor será incrementado até i<10
    print("i = ", i) # Esse comando será executado várias vezes até i<10 (tabulação necessária!)
    
# Exemplo de operação matemática:
for j in range(4): # a variável entre "for" e "in" pode ter qualquer nome. Aqui escolhemos "j"
    print("5 x %d = %d" % (j, 5*j))
    
# Exemplo de nomeação de arquivos: (Vamos trabalhar com arquivos na aula 5)
for k in range(5):
    name = ("arquivo_%d.txt" % k) # Uma forma de acrescentar um número inteiro à um string.
    print("[%d] Nome do arquivo: %s" % (k, name))
