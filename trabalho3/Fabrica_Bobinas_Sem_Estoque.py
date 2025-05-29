import numpy as np
from pulp import *

padroes_corte = np.array([
    [5, 2, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 2, 0, 0, 3, 1, 0, 0],
    [0, 0, 0, 2, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
])

demanda = np.array([18, 31, 25, 15, 14])

num_tipos_bobinas = padroes_corte.shape[0] 
num_padroes_corte = padroes_corte.shape[1] 

custo_padrao = np.ones(num_padroes_corte)

print("Parte 1: Produção com custo mínimo sem estoque")
print("=" * 50)

modelo = LpProblem("Fabrica_Bobinas_Sem_Estoque", LpMinimize)

# Variáveis de decisão: quantas vezes cada padrão de corte será usado
x = [LpVariable(f"x{j}", lowBound=0, cat=LpInteger) for j in range(num_padroes_corte)]

# Função objetivo: minimizar o custo total (número de bobinas-mestre usadas)
modelo += lpSum([x[j] * custo_padrao[j] for j in range(num_padroes_corte)]), "Custo_Total"

# Restrições: atender exatamente à demanda para cada tipo de bobina
for i in range(num_tipos_bobinas):
    modelo += lpSum([padroes_corte[i][j] * x[j] for j in range(num_padroes_corte)]) == demanda[i], f"Demanda_Tipo_{i+1}"

modelo.solve(PULP_CBC_CMD(msg=False))

print(f"Status da solução: {LpStatus[modelo.status]}")

if modelo.status == LpStatusOptimal:
    print("Quantidade de cada padrão de corte a ser usado:")
    for j in range(num_padroes_corte):
        if value(x[j]) > 0:
            print(f"Padrão {j+1}: {int(value(x[j]))} vezes")

    custo_total = sum(value(x[j]) * custo_padrao[j] for j in range(num_padroes_corte))
    print(f"\nCusto total (número de bobinas-mestre): {int(custo_total)}")

    # Verificação da solução
    producao = np.zeros(num_tipos_bobinas)
    for i in range(num_tipos_bobinas):
        for j in range(num_padroes_corte):
            producao[i] += padroes_corte[i][j] * value(x[j])

    print("\nVerificação da produção vs. demanda:")
    print("-" * 50)
    print("Tipo | Produção | Demanda")
    for i in range(num_tipos_bobinas):
        print(f"  {i+1}  |    {int(producao[i])}   |   {demanda[i]}")
