import numpy as np
import random

def inicializar_populacao(pop_size, n_genes):
    return [np.random.randint(0, 2, n_genes) for _ in range(pop_size)]

def avaliar_populacao(populacao, fitness_func):
    return [fitness_func(ind) for ind in populacao]

def selecionar_pais(populacao, aptidoes):
    aptidoes = np.array(aptidoes, dtype=float)
    prob = aptidoes / (np.sum(aptidoes) + 1e-6)
    idx = np.random.choice(len(populacao), p=prob)
    return populacao[idx]

def crossover_1p(pai1, pai2):
    n = len(pai1)
    ponto = random.randint(1, n - 1)
    filho1 = np.concatenate([pai1[:ponto], pai2[ponto:]])
    filho2 = np.concatenate([pai2[:ponto], pai1[ponto:]])
    return filho1, filho2

def mutacao_bitflip(individuo, taxa_mutacao):
    for i in range(len(individuo)):
        if random.random() < taxa_mutacao:
            individuo[i] = 1 - individuo[i]
    return individuo

def ag_run(
    fitness_func,
    n_genes,
    pop_size=40,
    n_generations=30,
    crossover_rate=0.8,
    mutation_rate=0.1,
    elitism_frac=0.1,
    mutation_rate_late=None,
    mutation_switch_gen=None,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    melhores_fitness = []
    best_individual = None
    best_fitness = -np.inf
    elite_size = max(1, int(pop_size * elitism_frac))
    populacao = inicializar_populacao(pop_size, n_genes)

    for gen in range(n_generations):
        # mutação adaptativa
        if mutation_rate_late is not None and mutation_switch_gen is not None:
            if gen < mutation_switch_gen:
                taxa_mutacao = mutation_rate
            else:
                taxa_mutacao = mutation_rate_late
        else:
            taxa_mutacao = mutation_rate

        aptidoes = avaliar_populacao(populacao, fitness_func)
        apt_array = np.array(aptidoes)
        idx_ordenados = np.argsort(-apt_array)  # fitness maior é melhor
        idx_melhor = int(idx_ordenados[0])
        melhor_fit_geracao = apt_array[idx_melhor]
        melhor_solucao_geracao = populacao[idx_melhor]

        # atualiza melhor global
        if melhor_fit_geracao > best_fitness:
            best_fitness = melhor_fit_geracao
            best_individual = melhor_solucao_geracao.copy()

        melhores_fitness.append(best_fitness)

        # log simples
        if (gen + 1) % 5 == 0 or gen == 0 or gen + 1 == n_generations:
            print(
                f"Geração {gen + 1:3d}/{n_generations} | "
                f"Taxa mutação: {taxa_mutacao:.3f} | "
                f"Melhor geração: {melhor_fit_geracao:7.2f} | "
                f"Melhor global: {best_fitness:7.2f}"
            )

        # nova população com elitismo
        nova_populacao = []
        # 1) preserva elite (cópias)
        for idx in idx_ordenados[:elite_size]:
            nova_populacao.append(populacao[idx].copy())
        # 2) gera o restante por recombinação + mutação
        while len(nova_populacao) < pop_size:
            pai1 = selecionar_pais(populacao, aptidoes)
            pai2 = selecionar_pais(populacao, aptidoes)
            filho1, filho2 = crossover_1p(pai1, pai2)
            filho1 = mutacao_bitflip(filho1, taxa_mutacao)
            filho2 = mutacao_bitflip(filho2, taxa_mutacao)
            nova_populacao.append(filho1)
            if len(nova_populacao) < pop_size:
                nova_populacao.append(filho2)
        populacao = nova_populacao

    return best_individual, best_fitness, melhores_fitness
