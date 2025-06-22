from pulp import *

# Parâmetros
n = 6
L = 17  
# ou (n = 8 e L = 34)
# ou (n = 15 e L = 151)

model = LpProblem("Golomb_Ruler", LpMinimize)

# Variáveis de decisão
x = LpVariable.dicts("x", range(L + 1), cat="Binary")
y = LpVariable.dicts("y", [(i, j) for i in range(L + 1) for j in range(i + 1, L + 1)], cat="Binary")
t = LpVariable("t", lowBound=0, cat="Integer")

# Fixar a primeira marca em 0
model += x[0] == 1

# Total de marcas: n (x[0] já é 1, então n-1 restantes)
model += lpSum(x[i] for i in range(1, L + 1)) == n - 1

# Definir t como a última marca
for i in range(L + 1):
    model += t >= i * x[i]

# Restrições de distâncias distintas
for d in range(1, L + 1):
    model += lpSum(y[i, i + d] for i in range(L + 1 - d) if (i, i + d) in y) <= 1

# Linearização dos produtos x_i * x_j usando y_{ij}
for i in range(L + 1):
    for j in range(i + 1, L + 1):
        if (i, j) in y:
            model += y[i, j] <= x[i]
            model += y[i, j] <= x[j]
            model += x[i] + x[j] - y[i, j] <= 1

# Função objetivo: minimizar comprimento da régua
model += t

model.solve()

print("Status:", LpStatus[model.status])
print("Comprimento mínimo:", value(t))
print("Marcas:", [i for i in range(L + 1) if value(x[i]) == 1])
