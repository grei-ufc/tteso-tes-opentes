from pyomo.environ import *

# Modelo
model = ConcreteModel()

# Variáveis de decisão (todas >= 0)
model.x1 = Var(within=NonNegativeReals)
model.x2 = Var(within=NonNegativeReals)
model.x3 = Var(within=NonNegativeReals)

# Função objetivo (minimização)
model.obj = Objective(expr= -2*model.x1 - 3*model.x2, sense=minimize)

# Restrição
model.restr = Constraint(expr= model.x1 + 2*model.x2 + model.x3 == 2)

# Resolver usando GLPK
solver = SolverFactory('glpk')
result = solver.solve(model)

# Mostrar resultados
print("x1 =", value(model.x1))
print("x2 =", value(model.x2))
print("x3 =", value(model.x3))
print("Z  =", value(model.obj))
