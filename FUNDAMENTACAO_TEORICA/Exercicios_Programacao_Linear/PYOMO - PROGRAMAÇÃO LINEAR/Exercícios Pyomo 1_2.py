import pyomo.environ as pyo

# Criar modelo
model = pyo.ConcreteModel()

# Variáveis de decisão (x1, x2 ≥ 0)
model.x1 = pyo.Var(within=pyo.NonNegativeReals)
model.x2 = pyo.Var(within=pyo.NonNegativeReals)

# Função objetivo: minimizar -x1 - 3x2
model.obj = pyo.Objective(expr=-model.x1 - 3*model.x2, sense=pyo.minimize)

# Restrições
model.constr1 = pyo.Constraint(expr=model.x1 + model.x2 <= 6)
model.constr2 = pyo.Constraint(expr=-model.x1 + 2*model.x2 <= 8)

# Resolver usando GLPK
solver = pyo.SolverFactory('glpk')
result = solver.solve(model)

# Mostrar resultados
print("Status da Otimização:", result.solver.status)
print("Status da Solução:", result.solver.termination_condition)
print("x1 =", pyo.value(model.x1))
print("x2 =", pyo.value(model.x2))
print("Valor ótimo da função objetivo =", pyo.value(model.obj))
