import pulp

LOCAIS = [1, 2, 3, 4, 5]
REGIOES = ['sul', 'central', 'sudeste', 'oeste', 'norte']

cobertura = {
    (1, 'sul'): 1, (1, 'central'): 1, (1, 'sudeste'): 0, (1, 'oeste'): 0, (1, 'norte'): 0,
    (2, 'sul'): 1, (2, 'central'): 1, (2, 'sudeste'): 1, (2, 'oeste'): 0, (2, 'norte'): 0,
    (3, 'sul'): 0, (3, 'central'): 1, (3, 'sudeste'): 0, (3, 'oeste'): 1, (3, 'norte'): 0,
    (4, 'sul'): 0, (4, 'central'): 1, (4, 'sudeste'): 0, (4, 'oeste'): 1, (4, 'norte'): 1,
    (5, 'sul'): 0, (5, 'central'): 0, (5, 'sudeste'): 1, (5, 'oeste'): 1, (5, 'norte'): 1,
}

prob = pulp.LpProblem("Bases_SAMU", pulp.LpMinimize)

# Variáveis de decisão - x[i] = 1 se instalar base no local i, 0 caso contrário
x = pulp.LpVariable.dicts("x", LOCAIS, cat=pulp.LpBinary)

# Função objetivo - minimizar o número total de bases
prob += pulp.lpSum(x[i] for i in LOCAIS), "Total de bases instaladas"

# Restrições - cada região deve ser coberta por pelo menos uma base
for j in REGIOES:
    prob += pulp.lpSum(cobertura[i, j] * x[i] for i in LOCAIS) >= 1, f"Cobertura da região {j}"

prob.solve()

print(f"Status da solução: {pulp.LpStatus[prob.status]}")
print(f"Total de bases instaladas: {pulp.value(prob.objective)}")

print("\nBases instaladas nos locais:")
for i in LOCAIS:
    if pulp.value(x[i]) == 1:
        print(f"  - Local {i}")

print("\nRegiões atendidas por essas bases:")
for j in REGIOES:
    print(f"Região {j} é atendida por:", end=" ")
    for i in LOCAIS:
        if pulp.value(x[i]) == 1 and cobertura[i, j] == 1:
            print(f"{i}", end=" ")
    print()

for j in REGIOES:
    cobertura_total = sum(pulp.value(x[i]) * cobertura[i, j] for i in LOCAIS)
    print(f"Região {j} é coberta por {cobertura_total} base(s)")
