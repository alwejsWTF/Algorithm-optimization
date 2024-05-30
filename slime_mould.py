import numpy as np
from functions import choose_fun


class MultiSwarmSMA:
    def __init__(self, function, scope, dimension, swarm_size, iterations,
                 num_swarms, z, verbose):
        self.function = function
        self.scope = scope
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.num_swarms = num_swarms
        self.z = z
        self.verbose = verbose
        self.swarms = [np.random.uniform(
            low=scope[0], high=scope[1], size=(swarm_size, dimension)
        ) for _ in range(num_swarms)]
        self.weights = [np.zeros((swarm_size, dimension))
                        for _ in range(num_swarms)]
        self.fitness = [np.array([function(ind) for ind in swarm])
                        for swarm in self.swarms]
        self.best_positions = [swarm[np.argmin(fitness)] for swarm, fitness
                               in zip(self.swarms, self.fitness)]
        self.best_fitness = [np.min(fitness) for fitness in self.fitness]
        self.global_best_position = (
            self.best_positions[np.argmin(self.best_fitness)]
        )
        self.global_best_fitness = np.min(self.best_fitness)
        self.epsilon = 1e-10

    def optimize(self):
        best_values_per_iteration = []
        for i in range(self.iterations):
            for swarm_idx, swarm in enumerate(self.swarms):
                fitness = self.fitness[swarm_idx]
                best_fitness = self.best_fitness[swarm_idx]
                worst_fitness = fitness[np.argmax(fitness)]
                best_position = self.best_positions[swarm_idx]
                for j in range(self.swarm_size):
                    if j <= int(self.swarm_size / 2):
                        self.weights[swarm_idx][j] = (
                            1 + np.random.rand() *
                            np.log10(np.maximum(
                                (best_fitness - fitness[j]) /
                                (best_fitness - worst_fitness + self.epsilon),
                                self.epsilon) + 1)
                        )
                    else:
                        self.weights[swarm_idx][j] = (
                             1 - np.random.rand() *
                             np.log10(np.maximum(
                                 (best_fitness - fitness[j]) /
                                 (best_fitness - worst_fitness + self.epsilon),
                                 self.epsilon) + 1)
                        )
                a = np.arctanh(-((i + 1) / self.iterations) + 1)
                for j in range(self.swarm_size):
                    if np.random.rand() < self.z:
                        new_position = np.random.uniform(
                            self.scope[0], self.scope[1], size=self.dimension
                        )
                    else:
                        p = np.tanh(fitness[j] - self.global_best_fitness)
                        vb = np.random.uniform(-a, a, size=self.dimension)
                        vc = np.random.uniform(-1, 1, size=self.dimension)
                        if np.random.random() < p:
                            index1, index2 = np.random.choice(
                                list(set(range(0, self.swarm_size))-{j}),
                                size=2, replace=False
                            )
                            new_position = (
                                self.global_best_position + vb *
                                (self.weights[swarm_idx][j] * swarm[index1]
                                 - swarm[index2])
                            )
                        else:
                            new_position = vc * swarm[j]

                    new_position = np.clip(
                        new_position, self.scope[0], self.scope[1]
                    )
                    new_fitness = self.function(new_position)
                    if new_fitness < fitness[j]:
                        swarm[j] = new_position
                        fitness[j] = new_fitness
                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
                            best_position = new_position
                            if new_fitness < self.global_best_fitness:
                                self.global_best_fitness = new_fitness
                                self.global_best_position = new_position

                self.best_fitness[swarm_idx] = best_fitness
                self.best_positions[swarm_idx] = best_position
                self.fitness[swarm_idx] = fitness
                self.swarms[swarm_idx] = swarm

            for swarm in self.swarms:
                for j in range(self.swarm_size):
                    swarm[j] += 0.1 * (self.global_best_position - swarm[j])
                    swarm[j] = np.clip(swarm[j], self.scope[0], self.scope[1])

            best_values_per_iteration.append(self.global_best_fitness)
            if self.verbose and ((i + 1) % 100 == 0 or i == 0):
                print(f"Iteration {i + 1}: "
                      f"Best value = {self.global_best_fitness}")

        return (self.global_best_position, self.global_best_fitness,
                best_values_per_iteration)


fun, bounds = choose_fun(1)
mssma = MultiSwarmSMA(
    fun, bounds, 30, 10, 100, 10, 0.6, False
)
best_pos, best_fit, _ = mssma.optimize()
print("Best Position:", best_pos)
print("Best Fitness:", best_fit)
