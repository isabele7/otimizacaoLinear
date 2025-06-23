import pulp

prob = pulp.LpProblem("Sudoku", pulp.LpMinimize)

# Variáveis de decisão: x_{i,j,k} = 1 se célula (i,j) contém número k
x = pulp.LpVariable.dicts("x", (range(9), range(9), range(1, 10)), cat='Binary')

# Função objetivo
prob += 0

# Restrições:
# 1. Cada célula deve conter exatamente um número
for i in range(9):
    for j in range(9):
        prob += pulp.lpSum(x[i][j][k] for k in range(1, 10)) == 1

# 2. Cada número aparece uma vez por linha
for i in range(9):
    for k in range(1, 10):
        prob += pulp.lpSum(x[i][j][k] for j in range(9)) == 1

# 3. Cada número aparece uma vez por coluna
for j in range(9):
    for k in range(1, 10):
        prob += pulp.lpSum(x[i][j][k] for i in range(9)) == 1

# 4. Cada número aparece uma vez por bloco 3x3
for block_i in range(3):
    for block_j in range(3):
        for k in range(1, 10):
            prob += pulp.lpSum(
                x[i][j][k]
                for i in range(block_i * 3, block_i * 3 + 3)
                for j in range(block_j * 3, block_j * 3 + 3)
            ) == 1

entrada = [
    [0, 0, 9, 7, 0, 0, 0, 0, 3],
    [0, 0, 0, 9, 0, 0, 1, 0, 0],
    [0, 0, 0, 3, 0, 6, 0, 0, 8],
    [9, 0, 6, 0, 4, 0, 0, 0, 0],
    [2, 0, 3, 0, 0, 5, 0, 0, 6],
    [0, 0, 0, 0, 0, 0, 0, 5, 7],
    [0, 3, 0, 0, 0, 2, 0, 8, 5],
    [8, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
]

for i in range(9):
    for j in range(9):
        val = entrada[i][j]
        if val != 0:
            prob += x[i][j][val] == 1

prob.solve()

sudoku_resolvido = [[0]*9 for _ in range(9)]
for i in range(9):
    for j in range(9):
        for k in range(1, 10):
            if pulp.value(x[i][j][k]) == 1:
                sudoku_resolvido[i][j] = k

for linha in sudoku_resolvido:
    print(linha)
