import numpy as np
import random

def inicializar_populacao(pop_size, n_genes):
    return [np.random.randint(0, 2, n_genes) for _ in range(pop_size)]

def avaliar_populacao(populacao, fitness_func):
    return [fitness_func(ind) for ind in populacao]

def selecionar_pais(populacao, aptidoes):
    aptidoes = np.array(aptidoes)
    prob = aptidoes / (np.sum(aptidoes) + 1e-6)
    idx = np.random.choice(len(populacao), p=prob)
    return populacao[idx]

def crossover_1p(pai1, pai2):
    n_genes = len(pai1)
    ponto = random.randint(1, n_genes - 1)
    filho1 = np.concatenate([pai1[:ponto], pai2[ponto:]])
    filho2 = np.concatenate([pai2[:ponto], pai1[ponto:]])
    return filho1, filho2

def mutacao_bitflip(individuo, taxa_mutacao=0.1):
    for i in range(len(individuo)):
        if random.random() < taxa_mutacao:
            individuo[i] = 1 - individuo[i]
    return individuo

def ag_run(fitness_func, n_genes, pop_size=40, n_generations=30, crossover_rate=0.8, mutation_rate=0.1, elitism=True, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    populacao = inicializar_populacao(pop_size, n_genes)
    best_fitness = -np.inf
    best_individual = None
    history = []

    for gen in range(n_generations):
        aptidoes = avaliar_populacao(populacao, fitness_func)
        gen_best_idx = np.argmax(aptidoes)
        gen_best_fit = aptidoes[gen_best_idx]
        gen_best_ind = populacao[gen_best_idx].copy()
        history.append(gen_best_fit)

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_individual = gen_best_ind.copy()

        nova_populacao = []
        if elitism:
            nova_populacao.append(gen_best_ind)

        while len(nova_populacao) < pop_size:
            pai1 = selecionar_pais(populacao, aptidoes)
            pai2 = selecionar_pais(populacao, aptidoes)
            if random.random() < crossover_rate:
                filho1, filho2 = crossover_1p(pai1, pai2)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()
            filho1 = mutacao_bitflip(filho1, mutation_rate)
            filho2 = mutacao_bitflip(filho2, mutation_rate)
            nova_populacao.extend([filho1, filho2])
        populacao = nova_populacao[:pop_size]
        print(f"Geração {gen+1:02d} | Melhor fitness: {gen_best_fit:.4f}")

    return best_individual, best_fitness, history
