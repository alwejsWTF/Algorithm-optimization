import numpy as np
from functions import choose_fun


class MultiSwarmSMA:
    def __init__(self, function, scope, dimension, swarm_size, iterations,
                 num_swarms, z, migration_threshold, mutation_rate,
                 no_improve_limit, verbose):
        self.function = function
        self.scope = scope
        self.dimension = dimension
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.num_swarms = num_swarms
        self.z = z
        self.migration_threshold = migration_threshold
        self.mutation_rate = mutation_rate
        self.no_improve_limit = no_improve_limit
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
        self.no_improve_counter = 0

    def crossover(self, pi, gi, pk):
        rd = np.random.rand(self.dimension)
        o = np.where(
            self.function(pi) < self.function(pk), rd * pi + (1 - rd) * gi, pk
        )
        return o

    def mutate(self, individual):
        for i in range(self.dimension):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.uniform(self.scope[0], self.scope[1])
        return individual

    def tournament_selection(self, swarm):
        selected = []
        for _ in range(int(0.2 * len(swarm))):
            participants_idx = np.random.choice(
                len(swarm), size=2, replace=False
            )
            participants = swarm[participants_idx]
            selected.append(min(participants, key=self.function))
            selected.append(participants[np.argmin(
                [self.function(part) for part in participants])])
        return np.array(selected)

    def optimize(self):
        best_values_per_iteration = []
        for i in range(self.iterations):
            improved = False
            # Main loop
            for swarm_idx in range(self.num_swarms):
                swarm = self.swarms[swarm_idx]
                fitness = self.fitness[swarm_idx]

                idx = np.argsort(fitness)
                worst = fitness[idx[-1]]
                best = fitness[idx[0]]

                for j in range(self.swarm_size):
                    current_fitness = self.function(swarm[j])
                    if j <= int(self.swarm_size / 2):
                        self.weights[swarm_idx][j] = (
                            1 + np.random.rand() *
                            np.log10(np.maximum(
                                (best - current_fitness) /
                                (best - worst + self.epsilon),
                                self.epsilon) + 1)
                        )
                    else:
                        self.weights[swarm_idx][j] = (
                             1 - np.random.rand() *
                             np.log10(np.maximum(
                                 (best - current_fitness) /
                                 (best - worst + self.epsilon),
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
                                list(set(range(0, self.swarm_size)) - {j}),
                                size=2, replace=False
                            )
                            new_position = (
                                self.global_best_position + vb *
                                (self.weights[swarm_idx][j] * swarm[index1] -
                                 swarm[index2])
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
                        improved = True
                        if new_fitness < self.best_fitness[swarm_idx]:
                            self.best_fitness[swarm_idx] = new_fitness
                            self.best_positions[swarm_idx] = new_position
                            if new_fitness < self.global_best_fitness:
                                self.global_best_fitness = new_fitness
                                self.global_best_position = new_position

                self.fitness[swarm_idx] = fitness
                self.swarms[swarm_idx] = swarm

            # Pattern
            for swarm_idx in range(self.num_swarms):
                new_swarm = []
                for j in range(self.swarm_size):
                    parent1 = self.best_positions[swarm_idx]
                    k_idx = np.random.randint(self.num_swarms)
                    parent2 = self.best_positions[k_idx]
                    global_best = self.global_best_position
                    offspring = self.crossover(
                        parent1, global_best, parent2)
                    new_swarm.append(self.mutate(offspring))
                for j in range(self.swarm_size):
                    offspring_fitness = self.function(
                        self.swarms[swarm_idx][j])
                    if offspring_fitness < self.fitness[swarm_idx][j]:
                        self.swarms[swarm_idx][j] = new_swarm[j]
                        self.fitness[swarm_idx][j] = offspring_fitness

            # Migration
            for swarm_idx in range(self.num_swarms - 1):
                best_f1 = self.best_fitness[swarm_idx]
                best_f2 = self.best_fitness[swarm_idx + 1]
                if abs(best_f1 - best_f2) > self.migration_threshold:
                    migration_rate = (
                            abs(best_f1 - best_f2) / max(best_f1, best_f2)
                    )
                    num_migrate = int(
                        max(1, min(migration_rate * self.swarm_size,
                                   self.swarm_size))
                    )
                    if best_f1 > best_f2:
                        source_swarm_idx, target_swarm_idx = (
                            swarm_idx, swarm_idx + 1
                        )
                    else:
                        source_swarm_idx, target_swarm_idx = (
                            swarm_idx + 1, swarm_idx
                        )
                    migrants = self.swarms[source_swarm_idx][:num_migrate]
                    self.swarms[target_swarm_idx][-num_migrate:] = migrants

            # Tournament selection
            if not improved:
                self.no_improve_counter += 1
                if self.no_improve_counter >= self.no_improve_limit:
                    for swarm_idx in range(self.num_swarms):
                        selected_individuals = self.tournament_selection(
                            self.swarms[swarm_idx])
                        for j in range(len(selected_individuals)):
                            self.swarms[swarm_idx][j] = selected_individuals[j]
                    self.no_improve_counter = 0
            else:
                self.no_improve_counter = 0

            best_values_per_iteration.append(self.global_best_fitness)
            if self.verbose and ((i + 1) % 100 == 0 or i == 0):
                print(f"Iteration {i + 1}: "
                      f"Best value = {self.global_best_fitness}")

        return (self.global_best_position, self.global_best_fitness,
                best_values_per_iteration)


fun, bounds = choose_fun(1)
mssma = MultiSwarmSMA(
    fun, bounds, 30, 10, 100, 10, 0.6, 0.4, 0.2, 30, False
)
best_pos, best_fit, _ = mssma.optimize()
print("Best Position:", best_pos)
print("Best Fitness:", best_fit)
