import numpy as np
from functions import choose_fun


class MultiSwarmGWO:
    def __init__(self, function, scope, dimension, population_size, iterations,
                 verbose):
        self.function = function
        self.scope = scope
        self.dimension = dimension
        self.population_size = population_size
        self.iterations = iterations
        self.verbose = verbose
        self.population = np.random.uniform(
            low=scope[0], high=scope[1], size=(population_size, dimension)
        )
        self.fitness = np.array([function(ind) for ind in self.population])
        self.alpha = self.population[np.argmin(self.fitness)]
        self.alpha_fitness = np.min(self.fitness)

        sorted_indices = np.argsort(self.fitness)
        self.beta = self.population[sorted_indices[1]]
        self.beta_fitness = self.fitness[sorted_indices[1]]
        self.delta = self.population[sorted_indices[2]]
        self.delta_fitness = self.fitness[sorted_indices[2]]

    def optimize(self):
        best_values_per_iteration = []
        for i in range(self.iterations):
            a = 2 - 2 * (i / self.iterations)
            for idx in range(self.population_size):
                # alpha
                r1, r2 = np.random.rand(
                    self.dimension), np.random.rand(self.dimension)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * self.alpha - self.population[idx])
                X1 = self.alpha - A1 * D_alpha

                # beta
                r1, r2 = np.random.rand(
                    self.dimension), np.random.rand(self.dimension)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * self.beta - self.population[idx])
                X2 = self.beta - A2 * D_beta

                # delta
                r1, r2 = np.random.rand(
                    self.dimension), np.random.rand(self.dimension)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * self.delta - self.population[idx])
                X3 = self.delta - A3 * D_delta

                # omega
                new_position = (X1 + X2 + X3) / 3
                new_position = np.clip(
                    new_position, self.scope[0], self.scope[1]
                )
                new_fitness = self.function(new_position)

                if new_fitness < self.fitness[idx]:
                    self.population[idx] = new_position
                    self.fitness[idx] = new_fitness

            sorted_indices = np.argsort(self.fitness)
            self.alpha = self.population[sorted_indices[0]]
            self.alpha_fitness = self.fitness[sorted_indices[0]]

            self.beta = self.population[sorted_indices[1]]
            self.beta_fitness = self.fitness[sorted_indices[1]]

            self.delta = self.population[sorted_indices[2]]
            self.delta_fitness = self.fitness[sorted_indices[2]]

            best_values_per_iteration.append(self.alpha_fitness)
            if self.verbose and ((i + 1) % 100 == 0 or i == 0):
                print(f"Iteration {i + 1}: "
                      f"Best value = {self.alpha_fitness}")

        return self.alpha, self.alpha_fitness, best_values_per_iteration


fun, bounds = choose_fun(6)
msgwo = MultiSwarmGWO(
    fun, bounds, 20, 50, 100, False
)
best_pos, best_fit, _ = msgwo.optimize()
print("Best Position:", best_pos)
print("Best Fitness:", best_fit)
