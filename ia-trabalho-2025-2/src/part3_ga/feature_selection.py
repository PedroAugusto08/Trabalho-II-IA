import numpy as np
import matplotlib.pyplot as plt
from ga import ag_run

# Definição do problema de feature selection didático
N_FEATURES = 16
features = [f"Atributo {i+1}" for i in range(N_FEATURES)]

# Função de fitness artificial: maximizar soma dos bits, penalizando excesso de atributos
def fitness(individuo):
    if np.sum(individuo) == 0:
        return 0
    score = np.sum(individuo)
    penalty = 0.2 * np.sum(individuo)
    return score - penalty

if __name__ == "__main__":
    best_ind, best_fit, history = ag_run(
        fitness_func=fitness,
        n_genes=N_FEATURES,
        pop_size=40,
        n_generations=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism=True,
        seed=42
    )
    print("\nMelhor indivíduo (seleção de atributos):", best_ind)
    print("Atributos selecionados:", [features[i] for i in range(N_FEATURES) if best_ind[i] == 1])
    print(f"Fitness do melhor indivíduo: {best_fit:.2f}")

    # Visualização da evolução do fitness
    plt.figure(figsize=(8, 5))
    plt.plot(history, marker='o')
    plt.title("Evolução do Fitness - AG Feature Selection")
    plt.xlabel("Geração")
    plt.ylabel("Melhor Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()