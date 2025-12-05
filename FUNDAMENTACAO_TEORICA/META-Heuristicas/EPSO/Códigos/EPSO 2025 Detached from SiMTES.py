
# Valores iniciais do EPSO
npar=10
nvar=7 #Tamanho da partícula
T=1 #Taxa de mutação
max_it=15000

#Partícula Concentrada
# for i in range(nvar):
#     particula.append(5+i)

#Particula aleatória
particula = np.random.randint(7, 75, size=nvar) #7-74 ja que o 75 não é incluido
global_best = particula.copy()

Particulas = np.tile(particula, (npar, 1)).copy() #Cria a matriz de particulas repetindo o vetor de particulas npar vezes

Weighta = np.ones((npar, nvar))
Weightb = np.ones((npar, nvar))
Weightc = np.full((npar, nvar), 2.0)

a = np.zeros((npar, nvar))
b = np.zeros((npar, nvar))
c = np.zeros((npar, nvar))

w = 2 - ((2.2 - 0.4) / max_it) * (np.arange(max_it) / max_it)


Personal_bests = Particulas.copy()

best_cost_iteration = np.zeros(max_it)

Velocities = np.ones((npar, nvar))

bateria={"size": 100,
"rate": 0.4,
"max_energy_flow": 40,
"max_soc": 100,
"min_soc": 10}

# Target=[5,8,17,32,16,14,74,54,39,25] #Alvo a ser encontrado, apenas para testes

dif_absoluta = np.full((npar, nvar), 10000)
dif_absoluta=dif_absoluta.astype(float)
Fitness_Personal_Bests = np.full(npar, 10000)
Fitness_Personal_Bests = Fitness_Personal_Bests.astype(float)
Fitness_global_best = 10000

gb_list=list()
gb_particula_list=list()
it_list=list()
particulas_list=list()
Eval_list=list()


#Definição da função de arredondamento
def round_half_up(x):
    if isinstance(x, (int, float)):
        return int(np.floor(x + 0.5))
    elif isinstance(x, np.ndarray):
        return np.floor(x + 0.5).astype(int)
    else:
        return [int(np.floor(val + 0.5)) for val in x]


#Definição da função de unicidade da partícula
def uniquer(particle):
    unique_particle = np.unique(particle)
    if len(unique_particle) < nvar:
        # Add random values if needed to maintain size
        missing = nvar - len(unique_particle)
        new_values = np.random.randint(7, 75, size=missing)
        unique_particle = np.concatenate([unique_particle, new_values])
    return unique_particle[:nvar] 

for it in range(max_it):

    #Replicação
    Particulas = np.tile(particula, (npar, 1)).copy()
    Eval = np.full(npar, 9999)
    Eval=Eval.astype(float)
    
    #Checagem dos limites do global_best
    global_best = np.clip(global_best, 7, 74)
    
    #Mutação
    Na = np.random.normal(0, 1, (npar, nvar))
    Nb = np.random.normal(0, 1, (npar, nvar))
    Nc = np.random.normal(0, 1, (npar, nvar))

    Weighta = Na * T * w[it] * Weighta
    Weightb = Weightb * Nb * T
    Weightc = Weightc * Nc * T
    
    #Reprodução
    a = Weighta * Velocities
    b = Weightb * (Personal_bests - Particulas)
    c = Weightc * (global_best - Particulas)
    
    Velocities = a + b + c
    Velocities = np.clip(Velocities, -7.1, 7.1)

    Particulas = round_half_up(Particulas + Velocities).astype(int)
    Particulas = np.clip(Particulas, 7, 74)
    
    i = 0
    while i < npar:
        Particulas[i] = uniquer(Particulas[i])        
        if i > 0:
            cont = 0
            while cont < i:
                # Check if particles are too similar
                aux = np.concatenate([Particulas[cont], Particulas[i]])
                aux_set = np.unique(aux)
                
                if len(aux_set) <= nvar:
                    # Particles are too similar, modify current particle
                    Particulas[i, -1] = Particulas[i, -1] + 1
                    Particulas[i] = uniquer(Particulas[i])
                    cont = 0  # Restart checking from beginning
                else:
                    cont += 1
        i += 1
    
    
    #Avaliação

    
    for i in range(npar):           
        config_bess_dso_dict_ = dict()
        particula_int = Particulas[i].astype(int)  # Convert to int array
        particula_int = particula_int.tolist()
        for j in particula_int:
            config_bess_dso_dict_[j] = bateria
        model_dso = run_opt(bess_storage_nodes,
                            particula_int,
                            config_bess_prosumer_dict,
                            config_bess_dso_dict_)
        if model_dso.solutions.solutions:
            Eval[i] = value(model_dso.obj)
        else:
            Eval[i] = 9999

    #Seleção

    improved_mask = Eval < Fitness_Personal_Bests
    Fitness_Personal_Bests[improved_mask] = Eval[improved_mask]
    
    for i in np.where(improved_mask)[0]:
        Personal_bests[i] = Particulas[i].copy()
    
    min_index = np.argmin(Fitness_Personal_Bests)
    if Fitness_Personal_Bests[min_index] < Fitness_global_best:
        Fitness_global_best = Fitness_Personal_Bests[min_index]
        global_best = Personal_bests[min_index].copy()
    
    particula = global_best.copy()

    best_cost_iteration[it] = Fitness_global_best

    #Save results
    gb_list.append(Fitness_global_best)
    gb_particula_list.append(global_best)
    it_list.append(it)
    particulas_list.append(Particulas.tolist())
    Eval_list.append(Eval.tolist())

    
    # if Fitness_global_best <= 0.00000000005:
    #     print('\n Tolerância atingida \n Eval = \t',min(Eval),'\n global_best[',it,'] = ',global_best)

    #     break

    if it>10:
        break
    # if Fitness_global_best <=30:
    #     T=0.5