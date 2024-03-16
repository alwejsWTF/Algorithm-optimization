import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_function, bounds, population_size=50, F=0.5, CR=0.9, max_generations=1000, verbose=True):
        self.objective_function = objective_function
        self.bounds = np.asarray(bounds)
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.max_generations = max_generations
        self.verbose = verbose
        self.dimension = self.bounds.shape[1]  # Number of dimensions
        self.population = np.random.uniform(low=self.bounds[0], high=self.bounds[1],
                                           size=(self.population_size, self.dimension))
        self.fitness = np.asarray([self.objective_function(ind) for ind in self.population])
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx]
        self.best_fitness = self.fitness[self.best_idx]

    def mutate(self, target_idx):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        r1, r2, r3, r4 = self.population[np.random.choice(idxs, 4, replace=False)]
        mutant_vector = r1 + self.F * (r1 - r2) + self.F * (r3 - r4) #self.population[target_idx]
        return mutant_vector

    def recombine(self, target_vector, mutant_vector):
        crossover_points = np.random.rand(self.dimension) < self.CR
        trial_vector = np.where(crossover_points, mutant_vector, target_vector)
        return trial_vector

    def select(self, target_idx, trial_vector):
        trial_fitness = self.objective_function(trial_vector)
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial_vector
            self.fitness[target_idx] = trial_fitness
            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial_vector

    def optimize(self):
        for generation in range(self.max_generations):
            for i in range(self.population_size):
                mutant_vector = self.mutate(i)
                trial_vector = self.recombine(self.population[i], mutant_vector)
                trial_vector = np.clip(trial_vector, self.bounds[0], self.bounds[1])
                self.select(i, trial_vector)

            if self.verbose and generation % 100 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness

# Example usage:
def trid_function(x):
    sum_squares = np.sum((x - 1)**2)
    sum_product = np.sum(x[:-1] * x[1:])
    return sum_squares - sum_product

bounds = (np.array([-10] * 30), np.array([10] * 30))
de_optimizer = DifferentialEvolution(objective_function=trid_function, bounds=bounds)
best_solution, best_fitness = de_optimizer.optimize()
print(f"Best solution found: {best_solution}, Best fitness: {best_fitness}")
