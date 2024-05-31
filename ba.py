import numpy as np


class BatAlgorithm:
    def __init__(self, function, scope, dimension, population_size, iterations,
                 loudness, pulse_rate, alpha, gamma, f_min, f_max, verbose):
        self.function = function
        self.population_size = population_size
        self.scope = scope
        self.dimension = dimension
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.f_min = f_min
        self.f_max = f_max
        self.verbose = verbose
        self.frequency = np.zeros(population_size)
        self.loudness = np.full(population_size, loudness)
        self.pulse_rate = np.full(population_size, pulse_rate)
        self.bats = np.random.uniform(
            low=scope[0], high=scope[1], size=(population_size, dimension)
        )
        self.velocity = np.zeros((population_size, dimension))
        self.fitness = np.array([self.function(bat) for bat in self.bats])
        self.best_bat = self.bats[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def optimize(self):
        best_values_per_iteration = []
        for t in range(self.iterations):
            for i in range(self.population_size):
                self.frequency[i] = np.clip(
                    self.f_min + (self.f_max - self.f_min) * np.random.rand(),
                    self.f_min, self.f_max
                )
                self.velocity[i] += (
                        (self.bats[i] - self.best_bat) * self.frequency[i]
                )
                candidate = np.clip(
                    self.bats[i] + self.velocity[i],
                    self.scope[0], self.scope[1]
                )

                if np.random.rand() > self.pulse_rate[i]:
                    rnd_numbers = np.random.uniform(
                        low=-1, high=1, size=self.dimension
                    )
                    solution = (
                            self.best_bat + rnd_numbers * self.loudness.mean()
                    )
                    solution = np.clip(solution, self.scope[0], self.scope[1])
                    if self.function(solution) <= self.function(candidate):
                        candidate = solution

                fitness_candidate = self.function(candidate)
                if (fitness_candidate < self.fitness[i]
                        and np.random.rand() < self.loudness[i]):
                    self.bats[i] = candidate
                    self.fitness[i] = fitness_candidate
                    self.loudness[i] *= self.alpha
                    self.pulse_rate[i] *= (1 - np.exp(-self.gamma * t))
                    if fitness_candidate < self.function(self.best_bat):
                        self.best_bat = candidate
                        self.best_fitness = fitness_candidate

            best_values_per_iteration.append(self.best_fitness)
            if self.verbose and ((t + 1) % 100 == 0 or t == 0):
                print(f"Iteration {t + 1}: "
                      f"Best value = {self.best_fitness}")

        return self.best_bat, self.best_fitness, best_values_per_iteration
