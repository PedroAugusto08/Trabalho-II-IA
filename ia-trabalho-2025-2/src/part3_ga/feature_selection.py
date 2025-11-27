import numpy as np
import matplotlib.pyplot as plt
from ga import ag_run

N_FEATURES = 16
features = [f"Atributo {i+1}" for i in range(N_FEATURES)]

def fitness(individuo):
    if np.sum(individuo) == 0:
        return 0
    score = np.sum(individuo)
    penalty = 0.2 * np.sum(individuo)
    return score - penalty

def plotar_evolucao(melhores_fitness):
    plt.figure(figsize=(8, 4))
    geracoes = np.arange(1, len(melhores_fitness) + 1)
    plt.plot(geracoes, melhores_fitness, linestyle="-", marker="o")
    plt.xlabel("Geração")
    plt.ylabel("Melhor fitness global")
    plt.title("Evolução do melhor fitness ao longo das gerações")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("evolucao_fitness.png")
    plt.show()

if __name__ == "__main__":
    best_ind, best_fit, history = ag_run(
        fitness_func=fitness,
        n_genes=N_FEATURES,
        pop_size=100,
        n_generations=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_frac=0.05,
        mutation_rate_late=0.3,
        mutation_switch_gen=20,
        seed=42
    )
    print("\nMelhor indivíduo (seleção de atributos):", best_ind)
    print("Atributos selecionados:", [features[i] for i in range(N_FEATURES) if best_ind[i] == 1])
    print(f"Fitness do melhor indivíduo: {best_fit:.2f}")
    plotar_evolucao(history)