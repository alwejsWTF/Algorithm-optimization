import numpy as np
from functions import choose_fun


class BatAlgorithm:
    def __init__(self, function, scope, dimension, population_size, iterations,
                 loudness, pulse_rate, alpha, gamma, f_min, f_max):
        self.function = function
        self.population_size = population_size
        self.scope = scope
        self.dimension = dimension
        self.iterations = iterations
        self.alpha = alpha
        self.gamma = gamma
        self.f_min = f_min
        self.f_max = f_max
        self.frequency = np.zeros(population_size)
        # self.loudness = np.random.uniform(
        #     low=0, high=2, size=population_size
        # )
        # self.pulse_rate = np.random.uniform(
        #     low=0, high=1, size=population_size
        # )
        self.loudness = np.full(population_size, loudness)
        self.pulse_rate = np.full(population_size, pulse_rate)
        self.bats = np.random.uniform(low=scope[0], high=scope[1],
                                      size=(population_size, dimension))
        self.velocity = np.zeros((population_size, dimension))
        self.fitness = np.array([self.function(bat) for bat in self.bats])
        self.best_bat = self.bats[np.argmin(self.fitness)]

    def optimize(self):
        for t in range(self.iterations):
            for i in range(self.population_size):
                self.frequency[i] = (
                        self.f_min + (self.f_max - self.f_min) *
                        np.random.rand()
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

        return self.best_bat, self.function(self.best_bat)


fun, sc = choose_fun(6, True)
ba = BatAlgorithm(function=fun, scope=sc, dimension=20, population_size=100,
                  iterations=1000, loudness=0.7, pulse_rate=0.5,
                  alpha=0.9, gamma=0.9, f_min=0, f_max=2)
best_bat, best_fitness = ba.optimize()
print("Best bat position:\n", best_bat)
print("Best fitness:", best_fitness)
