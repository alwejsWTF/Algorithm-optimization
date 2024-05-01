import numpy as np


#  DE/current/2/bin strategy
class DifferentialEvolution:
    def __init__(self, function, scope, dimension, population_size=50, F=0.5,
                 CR=0.5, generations=1000, verbose=True):
        self.function = function
        self.bounds = np.tile(scope, (dimension, 1))
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.generations = generations
        self.verbose = verbose
        self.dimension = dimension
        # Initialize the population within the bounds
        self.population = np.random.uniform(low=self.bounds[:, 0],
                                            high=self.bounds[:, 1],
                                            size=(self.population_size,
                                                  self.dimension))
        # Calculate the fitness of the initial population
        self.fitness = np.asarray(
            [self.function(ind) for ind in self.population])
        # Find the index of the best solution
        self.best_idx = np.argmin(self.fitness)
        # Store the best solution and its fitness
        self.best_solution = self.population[self.best_idx]
        self.best_fitness = self.fitness[self.best_idx]

    def mutate(self, target_idx):
        idxs = [idx for idx in range(self.population_size) if
                idx != target_idx]
        r1, r2, r3, r4 = self.population[
            np.random.choice(idxs, 4, replace=False)]
        mutant_vector = (self.population[target_idx] + self.F
                         * (r1 - r2) + self.F * (r3 - r4))
        return mutant_vector

    def recombine(self, target_vector, mutant_vector):
        crossover_points = np.random.rand(self.dimension) < self.CR
        trial_vector = np.where(crossover_points, mutant_vector, target_vector)
        return trial_vector

    def select(self, target_idx, trial_vector):
        trial_fitness = self.function(trial_vector)
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial_vector
            self.fitness[target_idx] = trial_fitness
            if trial_fitness < self.best_fitness:
                self.best_fitness = trial_fitness
                self.best_solution = trial_vector

    def optimize(self):
        best_values_per_iteration = []
        for generation in range(self.generations):
            for i in range(self.population_size):
                mutant_vector = self.mutate(i)
                trial_vector = self.recombine(
                    self.population[i], mutant_vector
                )
                trial_vector = np.clip(
                    trial_vector, self.bounds[:, 0], self.bounds[:, 1]
                )
                self.select(i, trial_vector)

            best_values_per_iteration.append(self.best_fitness)
            if self.verbose and ((generation + 1) % 100 == 0
                                 or generation == 0):
                print(f"Generation {generation + 1}: "
                      f"Best Fitness = {self.best_fitness}")

        return self.best_solution, self.best_fitness, best_values_per_iteration
