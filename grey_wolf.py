import numpy as np
from functions import choose_fun


class MultiSwarmGWO:
    def __init__(self, function, scope, dimension, swarm_size, iterations,
                 num_swarms, verbose):
        self.function = function
        self.scope = scope
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.num_swarms = num_swarms
        self.verbose = verbose
        self.swarms = [np.random.uniform(
            low=scope[0], high=scope[1], size=(swarm_size, dimension)
        ) for _ in range(num_swarms)]
        self.fitness = [np.array([function(ind) for ind in swarm])
                        for swarm in self.swarms]
        self.alpha_positions = [swarm[np.argmin(fitness)] for swarm, fitness
                                in zip(self.swarms, self.fitness)]
        self.alpha_scores = [np.min(fitness) for fitness in self.fitness]
        self.global_alpha_position = (
            self.alpha_positions[np.argmin(self.alpha_scores)]
        )
        self.global_alpha_score = np.min(self.alpha_scores)

    def optimize(self):
        best_values_per_iteration = []
        for i in range(self.iterations):
            for swarm_idx, swarm in enumerate(self.swarms):
                fitness = self.fitness[swarm_idx]
                sorted_indices = np.argsort(fitness)

                alpha_idx = sorted_indices[0]
                beta_idx = sorted_indices[1]
                delta_idx = sorted_indices[2]

                alpha_position = swarm[alpha_idx]
                beta_position = swarm[beta_idx]
                delta_position = swarm[delta_idx]

                a = 2 - 2 * (i / self.iterations)

                for j in range(self.swarm_size):
                    r1, r2 = np.random.rand(
                        self.dimension), np.random.rand(self.dimension)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = np.abs(C1 * alpha_position - swarm[j])
                    X1 = alpha_position - A1 * D_alpha

                    r1, r2 = np.random.rand(
                        self.dimension), np.random.rand(self.dimension)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = np.abs(C2 * beta_position - swarm[j])
                    X2 = beta_position - A2 * D_beta

                    r1, r2 = np.random.rand(
                        self.dimension), np.random.rand(self.dimension)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = np.abs(C3 * delta_position - swarm[j])
                    X3 = delta_position - A3 * D_delta

                    new_position = (X1 + X2 + X3) / 3
                    new_position = np.clip(
                        new_position, self.scope[0], self.scope[1]
                    )
                    new_fitness = self.function(new_position)

                    if new_fitness < fitness[j]:
                        swarm[j] = new_position
                        fitness[j] = new_fitness
                        if new_fitness < self.alpha_scores[swarm_idx]:
                            self.alpha_scores[swarm_idx] = new_fitness
                            self.alpha_positions[swarm_idx] = new_position
                            if new_fitness < self.global_alpha_score:
                                self.global_alpha_score = new_fitness
                                self.global_alpha_position = new_position

                self.fitness[swarm_idx] = fitness
                self.swarms[swarm_idx] = swarm

            for swarm in self.swarms:
                for j in range(self.swarm_size):
                    swarm[j] += 0.1 * (self.global_alpha_position - swarm[j])
                    swarm[j] = np.clip(swarm[j], self.scope[0], self.scope[1])

            best_values_per_iteration.append(self.global_alpha_score)
            if self.verbose and ((i + 1) % 100 == 0 or i == 0):
                print(f"Iteration {i + 1}: "
                      f"Best value = {self.global_alpha_score}")

        return (self.global_alpha_position, self.global_alpha_score,
                best_values_per_iteration)


fun, bounds = choose_fun(1)
msgwo = MultiSwarmGWO(
    fun, bounds, 20, swarm_size=30, iterations=100, num_swarms=3, verbose=False
)
best_pos, best_fit, _ = msgwo.optimize()
print("Best Position:", best_pos)
print("Best Fitness:", best_fit)
