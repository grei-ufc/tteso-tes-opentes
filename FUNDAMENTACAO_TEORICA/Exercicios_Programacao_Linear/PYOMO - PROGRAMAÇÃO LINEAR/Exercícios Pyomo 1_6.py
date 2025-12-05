import pyomo.environ as pyo

# -----------------------
# Dados do problema
# -----------------------
T = range(1, 7)  # semanas 1 a 6
demand_swiss = {1:12, 2:8, 3:16, 4:24, 5:20, 6:12}   # mil lb
demand_cheddar = {1:9, 2:6, 3:12, 4:18, 5:15, 6:9}

prod_rate = {'swiss': 200, 'cheddar': 180}  # lb/hora
work_hours = 35  # horas/semana
max_workers = 90
init_workers = 30

# custos
cost_wage = 500
cost_hire = 1000
cost_fire = 2000
cost_hold = {'swiss': 20, 'cheddar': 15}

# -----------------------
# Modelo
# -----------------------
model = pyo.ConcreteModel()

# Variáveis
model.workers = pyo.Var(T, within=pyo.NonNegativeIntegers)
model.hires   = pyo.Var(T, within=pyo.NonNegativeIntegers)
model.fires   = pyo.Var(T, within=pyo.NonNegativeIntegers)

model.prod_swiss   = pyo.Var(T, within=pyo.NonNegativeReals)
model.prod_cheddar = pyo.Var(T, within=pyo.NonNegativeReals)

model.stock_swiss   = pyo.Var(T, within=pyo.NonNegativeReals)
model.stock_cheddar = pyo.Var(T, within=pyo.NonNegativeReals)

# -----------------------
# Restrições
# -----------------------

# Evolução da força de trabalho
def workforce_rule(m, t):
    if t == 1:
        return m.workers[t] == init_workers + m.hires[t] - m.fires[t]
    else:
        return m.workers[t] == m.workers[t-1] + m.hires[t] - m.fires[t]
model.workforce = pyo.Constraint(T, rule=workforce_rule)

# Capacidade de produção (1000 lb → horas)
def capacity_rule(m, t):
    swiss_hours   = (1000 * m.prod_swiss[t])   / prod_rate['swiss']
    cheddar_hours = (1000 * m.prod_cheddar[t]) / prod_rate['cheddar']
    return swiss_hours + cheddar_hours <= m.workers[t] * work_hours
model.capacity = pyo.Constraint(T, rule=capacity_rule)

# Balanço de estoque
def stock_swiss_rule(m, t):
    if t == 1:
        return m.stock_swiss[t] == m.prod_swiss[t] - demand_swiss[t]
    else:
        return m.stock_swiss[t] == m.stock_swiss[t-1] + m.prod_swiss[t] - demand_swiss[t]
model.stock_swiss_rule = pyo.Constraint(T, rule=stock_swiss_rule)

def stock_cheddar_rule(m, t):
    if t == 1:
        return m.stock_cheddar[t] == m.prod_cheddar[t] - demand_cheddar[t]
    else:
        return m.stock_cheddar[t] == m.stock_cheddar[t-1] + m.prod_cheddar[t] - demand_cheddar[t]
model.stock_cheddar_rule = pyo.Constraint(T, rule=stock_cheddar_rule)

# -----------------------
# Função objetivo
# -----------------------
def objective_rule(m):
    wages = sum(cost_wage * m.workers[t] for t in T)
    hiring = sum(cost_hire * m.hires[t] for t in T)
    firing = sum(cost_fire * m.fires[t] for t in T)
    holding = sum(cost_hold['swiss'] * m.stock_swiss[t] +
                  cost_hold['cheddar'] * m.stock_cheddar[t] for t in T)
    return wages + hiring + firing + holding

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# -----------------------
# Resolver
# -----------------------
solver = pyo.SolverFactory('glpk')
solver.solve(model, tee=True)

# -----------------------
# Resultados
# -----------------------
print("\n--- Resultados ---")
for t in T:
    print(f"Semana {t}: workers={pyo.value(model.workers[t]):.0f}, "
          f"hire={pyo.value(model.hires[t]):.0f}, fire={pyo.value(model.fires[t]):.0f}, "
          f"Swiss={pyo.value(model.prod_swiss[t]):.1f}, Cheddar={pyo.value(model.prod_cheddar[t]):.1f}, "
          f"Est_S={pyo.value(model.stock_swiss[t]):.1f}, Est_C={pyo.value(model.stock_cheddar[t]):.1f}")

print("\nCusto Total:", pyo.value(model.obj))
