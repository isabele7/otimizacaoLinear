import numpy as np
from pulp import *

# Matriz de padrões de corte
padroes_corte = np.array([
    [5, 2, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 2, 0, 0, 3, 1, 0, 0],
    [0, 0, 0, 2, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
])

# Demanda semanal para cada tipo de bobina
demanda = np.array([18, 31, 25, 15, 14])

# Número de tipos de bobinas e padrões de corte
num_tipos_bobinas = padroes_corte.shape[0] 
num_padroes_corte = padroes_corte.shape[1] 

custo_padrao = np.ones(num_padroes_corte)

print("Parte 2: Produção com custo mínimo com possibilidade de estoque")
print("=" * 50)

modelo2 = LpProblem("Fabrica_Bobinas_Com_Estoque", LpMinimize)

# Variáveis de decisão: quantas vezes cada padrão de corte será usado
y = [LpVariable(f"y{j}", lowBound=0, cat=LpInteger) for j in range(num_padroes_corte)]

# Variáveis para o estoque
estoque = [LpVariable(f"estoque{i}", lowBound=0, cat=LpInteger) for i in range(num_tipos_bobinas)]

# Função objetivo: minimizar o custo total (número de bobinas-mestre usadas)
modelo2 += lpSum([y[j] * custo_padrao[j] for j in range(num_padroes_corte)]), "Custo_Total"

# Restrições: atender pelo menos à demanda para cada tipo de bobina
for i in range(num_tipos_bobinas):
    modelo2 += lpSum([padroes_corte[i][j] * y[j] for j in range(num_padroes_corte)]) == demanda[i] + estoque[i], f"Demanda_Tipo_{i+1}"

modelo2.solve(PULP_CBC_CMD(msg=False))

print(f"Status da solução: {LpStatus[modelo2.status]}")

if modelo2.status == LpStatusOptimal:
    print("Quantidade de cada padrão de corte a ser usado:")
    for j in range(num_padroes_corte):
        if value(y[j]) > 0:
            print(f"Padrão {j+1}: {int(value(y[j]))} vezes")

    custo_total = sum(value(y[j]) * custo_padrao[j] for j in range(num_padroes_corte))
    print(f"\nCusto total (número de bobinas-mestre): {int(custo_total)}")

    # Verificação da solução
    producao = np.zeros(num_tipos_bobinas)
    estoques = np.zeros(num_tipos_bobinas)

    for i in range(num_tipos_bobinas):
        for j in range(num_padroes_corte):
            producao[i] += padroes_corte[i][j] * value(y[j])
        estoques[i] = value(estoque[i])

    print("\nVerificação da produção, demanda e estoque:")
    print("-" * 50)
    print("Tipo | Produção | Demanda | Estoque")
    for i in range(num_tipos_bobinas):
        print(f"  {i+1}  |    {int(producao[i])}   |   {demanda[i]}   |   {int(estoques[i])}")
