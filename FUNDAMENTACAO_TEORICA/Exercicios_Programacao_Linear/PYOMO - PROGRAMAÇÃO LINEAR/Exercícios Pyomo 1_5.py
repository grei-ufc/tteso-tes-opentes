from pyomo.environ import *

# Modelo
model = ConcreteModel()

# Variáveis
model.x1 = Var(domain=NonNegativeReals)
model.x2 = Var(domain=NonNegativeReals)

# Função objetivo
model.obj = Objective(expr=-2*model.x1 - 3*model.x2, sense=minimize)

# Restrição
model.constr = Constraint(expr=model.x1 + 2*model.x2 >= 2)

# Resolver
solver = SolverFactory('glpk')
results = solver.solve(model, tee=True)

# Mostrar resultados
print("\nStatus:", results.solver.status)
print("Terminação:", results.solver.termination_condition)

# Verifica se é ilimitado ou ótimo
if results.solver.termination_condition == TerminationCondition.unbounded:
    print("➡️ O problema é ILIMITADO (não existe solução ótima finita).")
elif results.solver.termination_condition == TerminationCondition.optimal:
    print("x1 =", value(model.x1))
    print("x2 =", value(model.x2))
    print("Valor ótimo =", value(model.obj))
else:
    print("⚠️ O solver não encontrou uma solução utilizável.")
