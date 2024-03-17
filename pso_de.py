import numpy as np

from functions import choose_fun


# PSO with DE
class Particle:
    def __init__(self, ps_bounds):
        self.position = np.array(
            [np.random.uniform(low, high) for low, high in ps_bounds])
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        self.bounds = ps_bounds

    def mutate(self, population, F):
        # DE/current/2/bin mutation strategy
        idxs = [i for i in range(len(population)) if
                not np.array_equal(population[i].position, self.position)]
        random_indices = np.random.choice(idxs, 4, replace=False)
        x1, x2, x3, x4 = [population[idx] for idx in random_indices]
        mutant_vector = self.position + F * (x1.position - x2.position) + F * (
                x3.position - x4.position)
        return mutant_vector

    def update_velocity(self, global_best_position, iw, cc, sc):
        r1, r2 = np.random.rand(2)
        cognitive_velocity = cc * r1 * (self.best_position - self.position)
        social_velocity = sc * r2 * (global_best_position - self.position)
        self.velocity = iw * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, particles, F, CR, fun):
        # Mutation
        mutant_vector = self.mutate(particles, F)
        # Crossover (binomial)
        cross_points = np.random.rand(len(self.bounds)) < CR
        trial_vector = np.where(cross_points, mutant_vector,
                                self.position + self.velocity)
        trial_value = fun(trial_vector)
        if trial_value < self.best_value:
            self.position = trial_vector
            self.best_position = trial_vector
            self.best_value = trial_value
        else:
            self.position += self.velocity
        self.position = np.clip(self.position, self.bounds[:, 0],
                                self.bounds[:, 1])


class ParticleSwarmOptimizer:
    def __init__(self, fun, scope, dimension, F=0.5, CR=0.5, num_particles=50,
                 max_iter=1000, iw=0.5, cc=2, sc=2, animated=False, verbose=True):
        self.fun = fun
        self.bounds = np.array([scope] * dimension)
        self.dimension = dimension
        self.F = F
        self.CR = CR
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.iw = iw
        self.cc = cc
        self.sc = sc
        self.global_best_value = float('inf')
        self.global_best_position = None
        self.animated = animated
        self.verbose = verbose
        self.particles = [Particle(self.bounds) for _ in
                          range(self.num_particles)]

    def optimize(self):
        if self.animated:
            self.max_iter = 1
        for i in range(self.max_iter):
            for particle in self.particles:
                value = self.fun(particle.position)
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_position = particle.position.copy()

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.iw, self.cc, self.sc)
                particle.update_position(self.particles, self.F, self.CR, self.fun)

            if self.verbose and (i + 1) % 100 == 0 or i == 0:
                print(f"Iteration {i + 1}: "
                      f"Best value = {self.global_best_value}")

        if self.animated:
            return [(p.position, self.fun(p.position)) for p in self.particles]
        return self.global_best_position, self.global_best_value


f, scope = choose_fun(6)
pso_optimizer = ParticleSwarmOptimizer(fun=f, scope=scope, dimension=30)
best_position, best_value = pso_optimizer.optimize()

print("Best position:", best_position)
print("Best value:", best_value)
