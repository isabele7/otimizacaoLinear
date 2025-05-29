import pulp
import pandas as pd

pessoas = ['Andre', 'Bruno', 'Carlos', 'Daniel']
jornais = ['O_Globo', 'Estadao', 'Folha', 'Lance']

tempos_leitura = {
    ('Andre', 'O_Globo'): 60,
    ('Andre', 'Estadao'): 30,
    ('Andre', 'Folha'): 2,
    ('Andre', 'Lance'): 5,
    ('Bruno', 'O_Globo'): 25,
    ('Bruno', 'Estadao'): 75,
    ('Bruno', 'Folha'): 3,
    ('Bruno', 'Lance'): 10,
    ('Carlos', 'O_Globo'): 10,
    ('Carlos', 'Estadao'): 15,
    ('Carlos', 'Folha'): 5,
    ('Carlos', 'Lance'): 30,
    ('Daniel', 'O_Globo'): 1,
    ('Daniel', 'Estadao'): 1,
    ('Daniel', 'Folha'): 1,
    ('Daniel', 'Lance'): 90
}

ordem_leitura = {
    'Andre': ['O_Globo', 'Estadao', 'Folha', 'Lance'],
    'Bruno': ['Estadao', 'Folha', 'O_Globo', 'Lance'],
    'Carlos': ['Folha', 'Estadao', 'O_Globo', 'Lance'],
    'Daniel': ['Lance', 'O_Globo', 'Estadao', 'Folha']
}

horario_acordar = {
    'Andre': 0,
    'Bruno': 15,
    'Carlos': 15,
    'Daniel': 60
}

model = pulp.LpProblem("Problema_Jornais", pulp.LpMinimize)

# Variáveis de início de leitura
t = pulp.LpVariable.dicts("t", ((p, j) for p in pessoas for j in jornais), lowBound=0, cat='Continuous')

# Variável para o tempo total
T = pulp.LpVariable("T", lowBound=0, cat='Continuous')

# Variáveis binárias para uso exclusivo dos jornais
y = pulp.LpVariable.dicts("y", ((p1, p2, j) for p1 in pessoas for p2 in pessoas for j in jornais if p1 != p2), cat='Binary')

# Função objetivo
model += T, "Tempo_Total"

# Restrições de horário de acordar
for p in pessoas:
    model += t[p, ordem_leitura[p][0]] >= horario_acordar[p], f"Acorda_{p}"

# Restrições de ordem de leitura por pessoa
for p in pessoas:
    ordem = ordem_leitura[p]
    for i in range(len(ordem) - 1):
        j_atual = ordem[i]
        j_prox = ordem[i + 1]
        model += t[p, j_prox] >= t[p, j_atual] + tempos_leitura[(p, j_atual)], f"Ordem_{p}_{j_atual}_{j_prox}"

# Restrições de exclusividade do jornal
M = 10000
for j in jornais:
    for i1 in pessoas:
        for i2 in pessoas:
            if i1 != i2:
                dur1 = tempos_leitura[(i1, j)]
                dur2 = tempos_leitura[(i2, j)]
                y_var = y[i1, i2, j]

                model += t[i1, j] + dur1 <= t[i2, j] + M * y_var, f"Sobreposicao1_{i1}_{i2}_{j}"
                model += t[i2, j] + dur2 <= t[i1, j] + M * (1 - y_var), f"Sobreposicao2_{i1}_{i2}_{j}"

# T deve ser maior que o fim da leitura do último jornal de cada pessoa
for p in pessoas:
    j_ult = ordem_leitura[p][-1]
    model += T >= t[p, j_ult] + tempos_leitura[(p, j_ult)], f"Final_{p}"

solver = pulp.PULP_CBC_CMD(msg=True)  
model.solve(solver)

if pulp.LpStatus[model.status] == 'Optimal':
    print(f"Tempo mínimo para todos saírem: {pulp.value(T):.0f} minutos após 8h30")

    horas = int(8.5 + pulp.value(T) // 60)
    minutos = int(pulp.value(T) % 60)
    print(f"Horário de saída: {horas:02d}h{minutos:02d}\n")

    for p in pessoas:
        print(f"CRONOGRAMA - {p.upper()}:")
        print("-" * 40)
        for j in ordem_leitura[p]:
            inicio = pulp.value(t[p, j])
            dur = tempos_leitura[(p, j)]
            fim = inicio + dur

            h_inicio = int(8.5 + inicio // 60)
            m_inicio = int(inicio % 60)
            h_fim = int(8.5 + fim // 60)
            m_fim = int(fim % 60)

            print(f"{j:10} | {h_inicio:02d}h{m_inicio:02d} - {h_fim:02d}h{m_fim:02d} ({dur:2d} min)")
        print()
