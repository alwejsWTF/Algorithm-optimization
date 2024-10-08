import numpy as np
import math


class ButterflyOptimizer:
    def __init__(self, function, scope, dimension, population_size, iterations,
                 sensory_modality, power_exponent, switch_probability,
                 verbose):
        self.function = function
        self.scope = scope
        self.dimension = dimension
        self.population_size = population_size
        self.iterations = iterations
        self.c = sensory_modality
        self.a = power_exponent
        self.p = switch_probability
        self.verbose = verbose
        self.butterflies = np.random.uniform(
            low=scope[0], high=scope[1], size=(population_size, dimension)
        )
        self.fitness = np.array(
            [self.function(butterfly) for butterfly in self.butterflies]
        )
        self.best_position = self.butterflies[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def levy_flight(self, Lambda):
        sigma = (
            math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
            (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))
        ) ** (1 / Lambda)
        u = np.random.normal(0, sigma ** 2, size=self.dimension)
        v = np.random.normal(0, 1, size=self.dimension)
        step = u / abs(v) ** (1 / Lambda)
        return step

    def get_random_neighbors(self, current_index):
        while True:
            index1, index2 = np.random.choice(
                self.population_size, size=2, replace=False)
            if index1 != current_index and index2 != current_index:
                break
        return index1, index2

    def optimize(self):
        best_values_per_iteration = []
        for t in range(self.iterations):
            self.a = 0.1 + 0.2 * t / self.iterations
            for i in range(self.population_size):
                fragrance = self.c * self.fitness[i] ** self.a
                r = np.random.rand()
                if r < self.p:
                    candidate = (
                        self.butterflies[i] + self.levy_flight(1.5) *
                        (self.best_position - self.butterflies[i]) *
                        fragrance
                    )
                else:
                    j, k = self.get_random_neighbors(i)
                    candidate = (
                        self.butterflies[i] + self.levy_flight(1.5) *
                        (self.butterflies[j] - self.butterflies[k]) *
                        fragrance
                    )

                candidate = np.clip(candidate, self.scope[0], self.scope[1])
                fitness_candidate = self.function(candidate)

                if fitness_candidate <= self.fitness[i]:
                    self.butterflies[i] = candidate
                    self.fitness[i] = fitness_candidate
                    if fitness_candidate < self.best_fitness:
                        self.best_fitness = fitness_candidate
                        self.best_position = candidate

            best_values_per_iteration.append(self.best_fitness)
            if self.verbose and ((t + 1) % 100 == 0 or t == 0):
                print(f"Iteration {t + 1}: "
                      f"Best value = {self.best_fitness}")

        return self.best_position, self.best_fitness, best_values_per_iteration
