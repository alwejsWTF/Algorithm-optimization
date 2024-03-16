import numpy as np


# Objective function example
def fobj(x):
    return np.sum(x**2)


# Mutation function for DE/current/2/bin
def mutation(population, target_idx, F):
    pop_size = population.shape[0]
    idxs = [idx for idx in range(pop_size) if idx != target_idx]
    r1, r2, r3, r4 = population[np.random.choice(idxs, 4, replace=False)]
    mutant = population[target_idx] + F * (r1 - r2) + F * (r3 - r4)
    return mutant


# Crossover function for DE/current/2/bin
def crossover(target, mutant, crossp):
    cross_points = np.random.rand(target.size) < crossp
    trial = np.where(cross_points, mutant, target)
    return trial


# Ensure the trial vector is within bounds
def ensure_bounds(vec, bounds):
    vec_new = np.clip(vec, bounds[:, 0], bounds[:, 1])
    return vec_new


# Define the Differential Evolution function with DE/current/2/bin strategy
def differential_evolution(fun, bounds, mutation_probability=0.8,
                           crossover_probability=0.7, population_size=20,
                           iterations=1000):
    dimensions = len(bounds)
    pop = np.random.rand(population_size, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fun(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(iterations):
        for j in range(population_size):
            mutant = mutation(pop, j, mutation_probability)
            trial = crossover(pop[j], mutant, crossover_probability)
            trial_denorm = ensure_bounds(trial, np.asarray(bounds))
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = (trial_denorm - min_b) / diff
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


# Example usage
bounds = [(-5, 5)] * 2 # Define bounds for a 2-D problem

# Run DE
result = next(differential_evolution(fobj, bounds))
print("Best solution:", result[0])
print("Fitness:", result[1])
