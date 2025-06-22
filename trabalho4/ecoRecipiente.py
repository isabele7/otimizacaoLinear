import pulp
import pandas as pd
import numpy as np

tipos = [35, 42]
plantas = [1, 2, 3, 4, 5]
trimestres = [1, 2, 3, 4]

# Demanda por planta, tipo e trimestre (da Tabela 1)
demanda = {
    (35, 1, 1): 138, (35, 1, 2): 142, (35, 1, 3): 139, (35, 1, 4): 140,
    (35, 2, 1): 32, (35, 2, 2): 33, (35, 2, 3): 34, (35, 2, 4): 36,
    (35, 3, 1): 61, (35, 3, 2): 66, (35, 3, 3): 67, (35, 3, 4): 73,
    (35, 4, 1): 284, (35, 4, 2): 278, (35, 4, 3): 305, (35, 4, 4): 322,
    (35, 5, 1): 0, (35, 5, 2): 0, (35, 5, 3): 0, (35, 5, 4): 0,  # Planta 5 não produz tipo 35

    (42, 1, 1): 226, (42, 1, 2): 255, (42, 1, 3): 272, (42, 1, 4): 289,
    (42, 2, 1): 141, (42, 2, 2): 160, (42, 2, 3): 175, (42, 2, 4): 188,
    (42, 3, 1): 134, (42, 3, 2): 116, (42, 3, 3): 126, (42, 3, 4): 130,
    (42, 4, 1): 1168, (42, 4, 2): 1138, (42, 4, 3): 1204, (42, 4, 4): 1206,
    (42, 5, 1): 0, (42, 5, 2): 0, (42, 5, 3): 0, (42, 5, 4): 0  # Assumindo que planta 5 não tem demanda própria
}

# Custos de produção por unidade (baseado na Tabela 2 - custo/100000 unidades)
custo_producao = {
    (35, 1): 7.60, (35, 2): 7.90, (35, 3): 8.15, (35, 4): 9.33,
    (42, 1): 5.20, (42, 2): 6.00, (42, 3): 5.50, (42, 4): 5.20, (42, 5): 5.00
}

# Dias de máquina necessários por 1000 unidades (baseado na Tabela 2)
dias_maquina_por_unidade = {
    (35, 1): 0.103, (35, 2): 0.109, (35, 3): 0.125, (35, 4): 0.113,
    (42, 1): 0.070, (42, 2): 0.050, (42, 3): 0.070, (42, 4): 0.061, (42, 5): 0.050
}

# Disponibilidade das máquinas por trimestre (Tabela 3) - dias disponíveis
disponibilidade_maquinas = {
    1: {1: 80, 2: 85, 3: 88, 4: 88},  # Média das máquinas por planta
    2: {1: 75, 2: 88, 3: 87, 4: 55},
    3: {1: 60, 2: 87, 3: 87, 4: 80},
    4: {1: 55, 2: 80, 3: 80, 4: 78},
    5: {1: 88, 2: 89, 3: 89, 4: 88}
}

# Custos de transporte por unidade (Tabela 4) - dividido por 1000 para normalizar
custo_transporte = {}
custos_transport_raw = {
    (1,1): 0, (1,2): 226, (1,3): 274, (1,4): 93, (1,5): 357,
    (2,1): 226, (2,2): 0, (2,3): 371, (2,4): 310, (2,5): 443,
    (3,1): 274, (3,2): 371, (3,3): 0, (3,4): 227, (3,5): 168,
    (4,1): 93, (4,2): 310, (4,3): 227, (4,4): 0, (4,5): 715,
    (5,1): 357, (5,2): 443, (5,3): 168, (5,4): 715, (5,5): 0
}

for (i,j), custo in custos_transport_raw.items():
    custo_transporte[(i,j)] = custo / 1000  # Normalizar para mesma unidade

# Capacidade de armazenamento (Tabela 5) - em milhares
capacidade_armazenamento = {
    1: {1: 376, 2: 325, 3: 348, 4: 410},
    2: {1: 55, 2: 47, 3: 62, 4: 58},
    3: {1: 875, 2: 642, 3: 573, 4: 813},
    4: {1: 10, 2: 15, 3: 30, 4: 24},
    5: {1: 103, 2: 103, 3: 30, 4: 410}
}

# Custos de armazenamento (Tabela 6) - por unidade
custo_armazenamento = {
    (35, 1): 0.085, (35, 2): 0.098, (35, 3): 0.075, (35, 4): 0.090, (35, 5): 0,
    (42, 1): 0.070, (42, 2): 0.098, (42, 3): 0.075, (42, 4): 0.080, (42, 5): 0.067
}

modelo = pulp.LpProblem("EcoRecipiente_Otimizacao", pulp.LpMinimize)

# Variáveis de decisão
# X[i,j,k] = quantidade produzida do tipo i na planta j no trimestre k
X = {}
for i in tipos:
    for j in plantas:
        for k in trimestres:
            if i == 35 and j == 5:  # Planta 5 não produz tipo 35
                X[i,j,k] = 0
            else:
                X[i,j,k] = pulp.LpVariable(f"X_{i}_{j}_{k}", lowBound=0, cat='Continuous')

# Y[i,j,k] = quantidade armazenada do tipo i na planta j no final do trimestre k
Y = {}
for i in tipos:
    for j in plantas:
        for k in trimestres:
            Y[i,j,k] = pulp.LpVariable(f"Y_{i}_{j}_{k}", lowBound=0, cat='Continuous')

# Z[i,j,l,k] = quantidade transportada do tipo i da planta j para planta l no trimestre k
Z = {}
for i in tipos:
    for j in plantas:
        for l in plantas:
            for k in trimestres:
                if j != l:
                    Z[i,j,l,k] = pulp.LpVariable(f"Z_{i}_{j}_{l}_{k}", lowBound=0, cat='Continuous')

# Função objetivo: minimizar custos totais
custo_total = 0

# Custos de produção
for i in tipos:
    for j in plantas:
        for k in trimestres:
            if i == 35 and j == 5:
                continue
            custo_total += custo_producao[(i,j)] * X[i,j,k]

# Custos de transporte
for i in tipos:
    for j in plantas:
        for l in plantas:
            for k in trimestres:
                if j != l:
                    custo_total += custo_transporte[(j,l)] * Z[i,j,l,k]

# Custos de armazenamento
for i in tipos:
    for j in plantas:
        for k in trimestres:
            custo_total += custo_armazenamento[(i,j)] * Y[i,j,k]

modelo += custo_total

# Restrições de atendimento à demanda (balanço por planta)
for i in tipos:
    for j in plantas:
        for k in trimestres:
            # Estoque anterior
            estoque_anterior = 0 if k == 1 else Y[i,j,k-1]

            # Produção na planta
            if i == 35 and j == 5:
                producao = 0
            else:
                producao = X[i,j,k]

            # Entrada por transporte
            entrada = pulp.lpSum([Z[i,l,j,k] for l in plantas if l != j])

            # Saída por transporte
            saida = pulp.lpSum([Z[i,j,l,k] for l in plantas if l != j])

            # Balanço: estoque_anterior + produção + entrada = demanda + saída + estoque_final
            modelo += estoque_anterior + producao + entrada == demanda[(i,j,k)] + saida + Y[i,j,k]

# Restrições de capacidade de produção (dias de máquina)
for i in tipos:
    for j in plantas:
        for k in trimestres:
            if i == 35 and j == 5:
                continue
            # Capacidade em dias disponíveis
            modelo += X[i,j,k] * dias_maquina_por_unidade[(i,j)] <= disponibilidade_maquinas[j][k] * 1000

# Restrições de capacidade de armazenamento
for j in plantas:
    for k in trimestres:
        modelo += pulp.lpSum([Y[i,j,k] for i in tipos]) <= capacidade_armazenamento[j][k]

print("Modelo criado com sucesso!")
print(f"Número de variáveis: {len([v for v in modelo.variables() if v is not None])}")
print(f"Número de restrições: {len(modelo.constraints)}")

# Resolver o modelo
print("\nResolvendo o modelo...")
modelo.solve()

# Exibir resultados
print(f"\nStatus da solução: {pulp.LpStatus[modelo.status]}")

if modelo.status == pulp.LpStatusOptimal:
    print(f"Custo ótimo: ${modelo.objective.value():,.2f}")
    total_producao = {35: 0, 42: 0}
    for i in tipos:
        print(f"\nTipo {i}:")
        for j in plantas:
            for k in trimestres:
                if i == 35 and j == 5:
                    continue
                if X[i,j,k].value() > 0.01:
                    valor = X[i,j,k].value()
                    total_producao[i] += valor
                    print(f"  Planta {j}, Trimestre {k}: {valor:.1f}")

    print(f"\nTotal produzido - Tipo 35: {total_producao[35]:.1f}, Tipo 42: {total_producao[42]:.1f}")

    # Verificar demanda total
    demanda_total = {35: 0, 42: 0}
    for i in tipos:
        for j in plantas:
            for k in trimestres:
                demanda_total[i] += demanda[(i,j,k)]

    print(f"Demanda total - Tipo 35: {demanda_total[35]}, Tipo 42: {demanda_total[42]}")
    for i in tipos:
        for j in plantas:
            for k in trimestres:
                if Y[i,j,k].value() > 1:
                    print(f"Tipo {i}, Planta {j}, Trimestre {k}: {Y[i,j,k].value():.1f}")
                  
    for i in tipos:
        for j in plantas:
            for l in plantas:
                for k in trimestres:
                    if j != l and Z[i,j,l,k].value() > 1:
                        print(f"Tipo {i}, Planta {j} → Planta {l}, Trimestre {k}: {Z[i,j,l,k].value():.1f}")
