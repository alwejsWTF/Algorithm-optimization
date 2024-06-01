import numpy as np


class GreyWolfOptimizer:
    def __init__(self, function, scope, dimension, population_size, iterations,
                 mutation_rate, no_improve_limit, verbose):
        self.function = function
        self.scope = scope
        self.dimension = dimension
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.no_improve_limit = no_improve_limit
        self.verbose = verbose
        self.population = np.random.uniform(
            low=scope[0], high=scope[1], size=(population_size, dimension)
        )
        self.fitness = np.array([function(ind) for ind in self.population])
        self.no_improve_counter = 0
        self.update_wolves()

    def update_wolves(self):
        sorted_indices = np.argsort(self.fitness)
        self.alpha = self.population[sorted_indices[0]]
        self.alpha_fitness = self.fitness[sorted_indices[0]]

        self.beta = self.population[sorted_indices[1]]
        self.beta_fitness = self.fitness[sorted_indices[1]]

        self.delta = self.population[sorted_indices[2]]
        self.delta_fitness = self.fitness[sorted_indices[2]]

    def crossover(self, pi, gi):
        rd = np.random.rand(self.dimension)
        return rd * pi + (1 - rd) * gi

    def mutate(self, individual):
        for i in range(self.dimension):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.uniform(self.scope[0], self.scope[1])
        return individual

    def tournament_selection(self, population):
        selected = []
        for _ in range(self.population_size):
            participants_idx = np.random.choice(
                len(population), size=2, replace=False
            )
            participants = population[participants_idx]
            selected.append(participants[np.argmin(
                [self.function(part) for part in participants])])
        return np.array(selected)

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
                new_position = np.clip(
                    (X1 + X2 + X3) / 3, self.scope[0], self.scope[1]
                )
                new_fitness = self.function(new_position)

                if new_fitness < self.fitness[idx]:
                    self.population[idx] = new_position
                    self.fitness[idx] = new_fitness

                k = np.random.randint(self.population_size)
                offspring = self.crossover(
                    self.population[idx], self.population[k]
                )
                offspring = self.mutate(offspring)
                offspring_fitness = self.function(offspring)
                if offspring_fitness < self.fitness[idx]:
                    self.population[idx] = offspring
                    self.fitness[idx] = offspring_fitness

            prev_best_fitness = self.alpha_fitness
            self.update_wolves()

            if self.alpha_fitness < prev_best_fitness:
                self.no_improve_counter = 0
            else:
                self.no_improve_counter += 1

            if self.no_improve_counter >= self.no_improve_limit:
                self.population = self.tournament_selection(self.population)
                self.no_improve_counter = 0

            best_values_per_iteration.append(self.alpha_fitness)
            if self.verbose and ((i + 1) % 100 == 0 or i == 0):
                print(f"Iteration {i + 1}: "
                      f"Best value = {self.alpha_fitness}")

        return self.alpha, self.alpha_fitness, best_values_per_iteration
