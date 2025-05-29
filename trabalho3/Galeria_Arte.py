import pulp

V = [1, 2, 3, 4, 5, 6, 7, 8]

E = [(1,2), (1,4), (2,3), (2,5), (3,6), (4,5), (4,7), (5,6), (5,8), (7,8)]

prob = pulp.LpProblem("Galeria_Arte", pulp.LpMinimize)

# Variáveis binárias: 1 se colocamos uma câmera no vértice i
x = pulp.LpVariable.dicts('x', V, cat='Binary')

# Função objetivo: minimizar número de câmeras
prob += pulp.lpSum([x[i] for i in V]), "MinCameras"

# Restrições: para cada aresta, pelo menos um dos extremos deve ter câmera
for (u, v) in E:
    prob += x[u] + x[v] >= 1, f"Edge_{u}_{v}"

prob.solve()

print("Status:", pulp.LpStatus[prob.status])
print("Câmeras colocadas nos vértices:")
for i in V:
    if x[i].value() == 1:
        print(f" - Vértice {i}")
