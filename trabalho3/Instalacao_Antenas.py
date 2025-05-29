import pulp

problema = pulp.LpProblem("Instalacao_Antenas", pulp.LpMaximize)

locais = ["A", "B", "C", "D", "E"]
areas = [1, 2, 3, 4, 5, 6]

intensidade = {
    "A": {1: 10, 2: 20, 3: 16, 4: 25, 5: 0, 6: 10},
    "B": {1: 0, 2: 12, 3: 18, 4: 23, 5: 11, 6: 6},
    "C": {1: 21, 2: 8, 3: 5, 4: 6, 5: 23, 6: 19},
    "D": {1: 16, 2: 15, 3: 15, 4: 8, 5: 14, 6: 18},
    "E": {1: 21, 2: 13, 3: 13, 4: 17, 5: 18, 6: 22}
}

nivel_minimo = 18

# Variáveis de decisão
# x[i] = 1 se uma antena é instalada no local i, 0 caso contrário
x = pulp.LpVariable.dicts("instalar_antena", locais, cat=pulp.LpBinary)

# y[j] = 1 se a área j é coberta, 0 caso contrário
y = pulp.LpVariable.dicts("area_coberta", areas, cat=pulp.LpBinary)

# Função objetivo: maximizar o número de áreas cobertas
problema += pulp.lpSum([y[j] for j in areas]), "Número total de áreas cobertas"

# Restrições

# Uma área j só pode ser coberta se pelo menos um local i com sinal suficiente tiver uma antena
for j in areas:
    z = {}
    for i in locais:
        var_name = f"cobertura_{i}_{j}"
        z[(i, j)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)

        if intensidade[i][j] < nivel_minimo:
            problema += z[(i, j)] == 0, f"Sem_cobertura_{i}_{j}"
        else:
            problema += z[(i, j)] <= x[i], f"Necessita_antena_{i}_{j}"

    # A área j só pode ser considerada coberta se pelo menos um local i a cobrir
    problema += y[j] <= pulp.lpSum([z[(i, j)] for i in locais]), f"Necessita_cobertura_{j}"

    # Não pode haver mais de um sinal atingindo o nível mínimo na mesma área
    problema += pulp.lpSum([z[(i, j)] for i in locais]) <= 1, f"Evitar_interferencia_{j}"

# Uma antena pode ser colocada no local E somente se uma antena também for instalada no local D
problema += x["E"] <= x["D"], "Local_E_requer_D"

problema.solve()

print(f"Status da solução: {pulp.LpStatus[problema.status]}")

print("\nLocais onde as antenas devem ser instaladas:")
for i in locais:
    if pulp.value(x[i]) == 1:
        print(f"Local {i}")

print("\nÁreas cobertas:")
areas_cobertas = []
for j in areas:
    if pulp.value(y[j]) == 1:
        areas_cobertas.append(j)
        print(f"Área {j}")

print(f"\nTotal de áreas cobertas: {len(areas_cobertas)}")

# Verificação da solução
print("\nDetalhes da cobertura:")
for j in areas:
    if pulp.value(y[j]) == 1:
        for i in locais:
            if pulp.value(x[i]) == 1 and intensidade[i][j] >= nivel_minimo:
                print(f"Área {j} coberta pelo Local {i} com intensidade {intensidade[i][j]}")
                break
