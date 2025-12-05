import numpy as np
from cplex import Cplex

def solve_residential_dispatch(load, pv, tariff, p_bat):
    T = len(load)

    # Bateria
    soc = 5
    soc_min, soc_max = 0, 10
    charge_lim = 3
    discharge_lim = 3

    # --- 1) Valida limites antes de chamar o CPLEX -------
    for t in range(T):
        # Clip da pot�ncia
        if p_bat[t] < -charge_lim or p_bat[t] > discharge_lim:
            return 1e9

        # Atualiza SOC
        soc += p_bat[t]
        if soc < soc_min or soc > soc_max:
            return 1e9

    # --- 2) Verifica se existe pelo menos 1 solu��o poss�vel --------
    for t in range(T):
        min_possible = load[t] - pv[t] - p_bat[t]
        if min_possible < 0:
            # Pgrid n�o pode ser negativo -> invi�vel
            return 1e9

    # --- 3) Monta o modelo CPLEX -------------------------
    try:
        model = Cplex()
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None)

        Pgrid = [f"Pgrid_{t}" for t in range(T)]

        # Vari�veis
        model.variables.add(names=Pgrid, lb=[0]*T)

        # Objetivo
        model.objective.set_sense(model.objective.sense.minimize)
        model.objective.set_linear([(Pgrid[t], tariff[t]) for t in range(T)])

        # Balan�o de pot�ncia
        for t in range(T):
            demand = load[t] - pv[t] - p_bat[t]
            model.linear_constraints.add(
                lin_expr=[[[Pgrid[t]], [1]]],
                senses=["E"],
                rhs=[demand]
            )

        # Resolve
        model.solve()
        sol = np.array(model.solution.get_values(Pgrid))
        cost = np.sum(sol * tariff)

        return cost

    except:
        # Se o CPLEX acusar inviabilidade
        return 1e9
