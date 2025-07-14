import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, PULP_CBC_CMD

# 1. Parâmetros das coortes (valores das tabelas)
alpha = {  # ingestão de MS (kg/cabeça·mês) por coorte
    1: 189, 2: 222, 3: 255, 4: 289, 5: 322,
    6: 355, 7: 388, 8: 421, 9: 454, 10: 490
}
theta = {  # preço de venda (R$ por cabeça) por coorte
    1: 658, 2: 691, 3: 802, 4: 913, 5: 1044,
    6: 1158, 7: 1271, 8: 1411, 9: 1526, 10: 1278
}

lambda_k = {  # custo de manutenção (R$/cabeça·mês) por coorte
    1: 1.74, 2: 1.95, 3: 2.19, 4: 2.4, 5: 2.61,
    6: 2.82, 7: 3.06, 8: 3.27, 9: 3.48, 10: 3.72
}

mu = {  # mortalidade mensal por coorte
    1: 0.0042, 2: 0.0042, 3: 0.002, 4: 0.002, 5: 0.002,
    6: 0.002, 7: 0.0003, 8: 0.0003, 9: 0.0003, 10: 0.0003
}

# 2. Produtividade das pastagens
rho = {
    1: 1633, 2: 1550, 3: 1467, 4: 1258, 5: 1050,
    6: 892,  7: 725,  8: 608,  9: 483,  10: 408, 11: 325
}
rho_m = {(p, mes): rho[p] for p in rho for mes in range(1, 13)}
rho_jan = {p: rho[p] for p in rho}

# 3. Custos de restauração (R$/ha por ano)
eta = {
    (1,1): 267.0,
    (2,1): 364.8, (2,2): 222.0,
    (3,1): 462.6, (3,2): 319.8, (3,3): 177.0,
    (4,1): 525.2, (4,2): 382.4, (4,3): 239.6, (4,4): 106.5,
    (5,1): 587.8, (5,2): 445.0, (5,3): 302.2, (5,4): 169.0, (5,5): 35.9,
    (6,1): 767.1, (6,2): 624.3, (6,3): 481.5, (6,4): 348.4, (6,5): 215.2, (6,6): 29.2,
    (7,1): 946.4, (7,2): 803.6, (7,3): 660.8, (7,4): 527.7, (7,5): 394.6, (7,6): 208.5, (7,7): 22.4,
    (8,1):1055.9, (8,2): 913.1, (8,3): 770.3, (8,4): 637.2, (8,5): 504.0, (8,6): 318.0, (8,7): 131.9, (8,8): 18.1,
    (9,1):1165.4, (9,2):1022.6, (9,3): 879.7, (9,4): 746.6, (9,5): 613.5, (9,6): 427.4, (9,7): 241.4, (9,8): 127.6, (9,9): 13.8,
    (10,1):1204.2, (10,2):1061.4, (10,3): 918.6,(10,4): 785.5,(10,5): 652.4,(10,6): 466.3,(10,7): 280.2,(10,8): 166.4,(10,9): 52.6,(10,10): 6.9,
    (11,1):1243.1, (11,2):1100.3,(11,3): 957.5,(11,4): 824.4,(11,5): 691.2,(11,6): 505.2,(11,7): 319.1,(11,8): 205.3,(11,9): 91.5,(11,10): 45.7,(11,11): 0.0
}

# 4. Outros parâmetros
A        = 600       # área total (ha)
xi       = 0.6       # eficiência de pastejo
sigma_M  = {m:0.00014 for m in range(1,13)}  # perda estoque de forragem (mensal)
tau_M    = {1:1000,2:1000,3:1000,4:1000,5:1000,6:1000,
             7:2000,8:2000,9:2000,10:2000,11:2000,12:2000}  # estoque mínimo (kg/ha)
pi       = 30        # taxa fixa por compra de boi
r_anual  = 0.055
i_men    = (1+r_anual)**(1/12)-1
FC = 43.9           #  custo fixo (R$/ha·mês)
dmp_o = 4000         #  estoque inicial de forragem (kg/ha)
gamma_cr = 0.234     #  parâmetro de amortização
lcr = 1_000_000      #  limite de crédito

capital_inicial   = 500_000
max_emprest_total = lcr
max_divida_ativo  = 0.7

degradacao = {p: 0.15 if p<=5 else 0.3 for p in range(1,12)}

# 5. Áreas iniciais (ha)
total_bons = A * 0.7   # boa qualidade
total_ruins = A * 0.3  # degradada/improdutiva
A_p0 = {}
for p in range(1,12):
    if p <= 3:
        A_p0[p] = total_bons/3
    elif p >= 8:
        A_p0[p] = total_ruins/4
    else:
        A_p0[p] = 0

def t_of_m(m): return ((m-1)//12)+1
def M_of_m(m): return ((m-1)%12)+1

# 6. Montar o modelo
prob = LpProblem("Pecuaria_NPV", LpMaximize)

# 6.1 Variáveis
T_years = 20
T       = range(1, T_years+1)
M_total = T_years * 12
M       = range(1, M_total+1)
P       = range(1, 12)
K       = range(1, 11)

Z   = LpVariable.dicts("Z",   ((t,p)   for t in T for p in P), lowBound=0)
RZ  = LpVariable.dicts("RZ",  ((t,p,q) for t in T for (p,q) in eta), lowBound=0)
X   = LpVariable.dicts("X",   ((m,k)   for m in M for k in K), lowBound=0, cat='Integer')
Y   = LpVariable.dicts("Y",   ((m,k)   for m in M for k in K), lowBound=0)
W   = LpVariable.dicts("W",   M, lowBound=0)
G   = LpVariable.dicts("G",   M, lowBound=0)
H   = LpVariable.dicts("H",   M, lowBound=0)
D   = LpVariable("D", lowBound=0, upBound=max_emprest_total)

# 6.2 Dinâmica de pastagens
for p in P:
    prob += Z[(1,p)] == A_p0[p]
for t in range(2, T_years+1):
    for p in P:
        stay = (1-degradacao[p])*Z[(t-1,p)] if p<max(P) else Z[(t-1,p)]
        degrade_in = degradacao[p-1]*Z[(t-1,p-1)] if p>min(P) else 0
        entr = lpSum(RZ[(t,q,p)] for (q,p2) in eta if p2==p)
        sai  = lpSum(RZ[(t,p,q)] for (p2,q) in eta if p2==p)
        prob += Z[(t,p)] == stay + degrade_in + entr - sai

# 6.3 Limite de restauração anual
for t in T:
    for q in P:
        origem = A_p0[q] if t==1 else Z[(t-1,q)]
        prob += lpSum(RZ[(t,q,p)] for (q,p) in eta if q==q) <= origem

# 6.4 Dinâmica do rebanho
for m in M:
    prob += Y[(m,1)] == X[(m,1)]
    for k in range(2,10):
        if m>1:
            prob += Y[(m,k)] == (1-mu[k-1])*Y[(m-1,k-1)]
        else:
            prob += Y[(m,k)] == 0
    # coorte 10 (abate)
    if m>1:
        prob += Y[(m,10)] == (1-mu[9])*Y[(m-1,9)]
    else:
        prob += Y[(m,10)] == 0
    # só compra k=1
    for k in range(2,11):
        prob += X[(m,k)] == 0

# 6.5 Produção e consumo de forragem
for m in M:
    ta = t_of_m(m); ma = M_of_m(m)
    prod = lpSum(rho_m[(p,ma)] * Z[(ta,p)] for p in P)
    if m>1:
        cons = (1+xi) * lpSum(alpha[k]*Y[(m,k)] for k in K)
        prob += cons + W[m] <= prod + (1 - sigma_M[ma])*W[m-1]
    else:
        cons = (1+xi) * lpSum(alpha[k]*Y[(m,k)] for k in K)
        prob += cons + W[m] <= prod + dmp_o * A

# 6.6 Estoque mínimo
for m in M:
    ma = M_of_m(m)
    prob += W[m] >= tau_M[ma] * A

# 6.7 Receitas (coorte 10)
for m in M:
    prob += G[m] == theta[10] * Y[(m,10)]

# 6.8 Despesas
for m in M:
    ta = t_of_m(m); ma = M_of_m(m)
    if (m-1) % 12 == 0:
        custo_rest = lpSum(eta[(p,q)] * RZ[(ta,p,q)] for (p,q) in eta)
    else:
        custo_rest = 0
    custo_gado  = lpSum((pi + theta[k]) * X[(m,k)] for k in K)
    custo_man   = lpSum(lambda_k[k] * Y[(m,k)] for k in K)
    prob += H[m] == FC*A + custo_gado + custo_man + custo_rest

# 6.9 Fluxo de caixa
caixa = capital_inicial + D
for m in M:
    # Juros sobre empréstimo
    juros = D * (r_anual/12)
    caixa = caixa + G[m] - H[m] - juros
    prob += caixa >= 0

# 6.10 Conservação de área
for t in T:
    prob += lpSum(Z[(t,p)] for p in P) == A

# 6.11 Produtividade mínima final
prob += lpSum(rho_jan[p]*Z[(T_years,p)] for p in P) >= \
        lpSum(rho_jan[p]*A_p0[p] for p in P)

# 6.12 Objetivo: maximizar NPV
npv = []
for m in M:
    desc = 1/((1+r_anual)**(m/12))
    flux = G[m] - H[m] - D*(r_anual/12)
    npv.append(desc * flux)

preco_terra_ha = 5000
valor_terra_fin  = lpSum(Z[(T_years,p)] * preco_terra_ha for p in P)
valor_rebanho_fin= theta[10] * Y[(M_total,10)]
valor_residual   = (valor_terra_fin + valor_rebanho_fin) / ((1+r_anual)**T_years)

# NPV total
prob += lpSum(npv) + valor_residual - capital_inicial - D/((1+r_anual)**T_years)

prob.solve(PULP_CBC_CMD)
print("Status:", LpStatus[prob.status])

if LpStatus[prob.status] == "Optimal":
    npv_total = sum((G[m].varValue - H[m].varValue - D.varValue*(r_anual/12)) /
                     ((1+r_anual)**(m/12)) for m in M)
    valor_terra_fin_val = sum(Z[(T_years,p)].varValue * preco_terra_ha for p in P)
    valor_rebanho_fin_val = theta[10] * Y[(M_total,10)].varValue
    valor_residual_val = (valor_terra_fin_val + valor_rebanho_fin_val) / ((1+r_anual)**T_years)
    npv_total = npv_total + valor_residual_val - capital_inicial - D.varValue/((1+r_anual)**T_years)

    npv_ha     = npv_total / A
    npv_ha_ano = npv_ha / T_years

    print(f"NPV anual por hectare:         R$ {npv_ha_ano:,.2f} ha⁻¹·ano⁻¹")

# Imprimir valores das variáveis de decisão em anos selecionados
anos_mostrar = [1, 5, 10, 15, 20]
print("\nDISTRIBUIÇÃO DE PASTAGENS:")
for t in anos_mostrar:
  print(f"\nAno {t}:")
  for p in P:
    area_p = Z[(t, p)].varValue
    if area_p >= 0.1:
      print(f"   P{p}: {area_p:.1f} ha")

print("\nPRINCIPAIS RESTAURAÇÕES:")
for t in T:
  restauracoes = []
  for (p, q) in eta:
    area_rz = RZ[(t, p, q)].varValue
    custo_rz = eta[(p, q)] * area_rz
    if area_rz >= 1e-1:
      restauracoes.append((area_rz, p, q, custo_rz))
  if restauracoes:
    restauracoes.sort(reverse=True)
    texto_ano = f"Ano {t}: "
    linhas = []
    for area_rz, p, q, custo_rz in restauracoes[:5]:
      linhas.append(f"P{p}→P{q}: {area_rz:.1f} ha (R$ {custo_rz:,.0f})")
    print(f"{texto_ano}{', '.join(linhas)}")
