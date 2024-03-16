import argparse
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Definicja funkcji przystosowania
def function(x, y, flag):
    if flag:
        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    return math.sin(3 * math.pi * x) ** 2 + (x - 1) ** 2 * (1 + math.sin(3 * math.pi * y) ** 2) \
                                          + (y - 1) ** 2 * (1 + math.sin(2 * math.pi * y) ** 2)


# Implementacja algorytmu
class Particle:
    def __init__(self, ps_bounds):
        self.bounds = ps_bounds
        self.position = np.array([random.uniform(self.bounds[i][0], self.bounds[i][1])
                                  for i in range(len(self.bounds))])
        self.velocity = np.array([0, 0])
        self.best_position = self.position.copy()
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, iw, cc, sc):
        cognitive_velocity = cc * random.random() * (self.best_position - self.position)
        social_velocity = sc * random.random() * (global_best_position - self.position)
        self.velocity = iw * self.velocity + cognitive_velocity + social_velocity

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, self.bounds[:, 0], self.bounds[:, 1])


class ParticleSwarmOptimizer:
    def __init__(self, fun, main_bounds, num_particles=30, max_iter=100, iw=0.5, cc=2, sc=2, function_flag=True,
                 use_stagnation=False, stagnation_limit=10, animated=False):
        self.fun = fun
        self.bounds = np.array(main_bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.iw = iw
        self.cc = cc
        self.sc = sc
        self.function_flag = function_flag
        self.global_best_value = float('inf')
        self.global_best_position = None
        self.use_stagnation = use_stagnation
        self.stagnation_limit = stagnation_limit
        self.animated = animated
        self.particles = [Particle(self.bounds) for _ in range(self.num_particles)]

    def optimize(self):
        if self.animated:
            self.max_iter = 1
        stagnation_count = 0
        for _ in range(self.max_iter):
            for particle in self.particles:
                value = self.fun(particle.position[0], particle.position[1], self.function_flag)
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_position = particle.position.copy()

                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = particle.position.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1

            if self.use_stagnation and stagnation_count >= self.stagnation_limit:
                break

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.iw, self.cc, self.sc)
                particle.update_position()

        if self.animated:
            return [(p.position, self.fun(p.position[0], p.position[1], self.function_flag)) for p in self.particles]
        return self.global_best_position, self.global_best_value


def setup_argparse():
    parser = argparse.ArgumentParser(description="Particle Swarm Optimization Algorithm")
    parser.add_argument('-x1', '--x1', type=float, default=-10, help="Lower boundary of x")
    parser.add_argument('-x2', '--x2', type=float, default=10, help="Upper boundary of x")
    parser.add_argument('-y1', '--y1', type=float, default=-10, help="Lower boundary of y")
    parser.add_argument('-y2', '--y2', type=float, default=10, help="Upper boundary of y")
    parser.add_argument('-pn', '--particle_number', type=int, default=30, help="Number of particles")
    parser.add_argument('-i', '--iterations', type=int, default=100, help="Number of iterations")
    parser.add_argument('-iw', '--inertia_weight', type=float, default=0.5, help="Weight of inertia")
    parser.add_argument('-cc', '--cognitive_component', type=float, default=2, help="Value of cognitive component")
    parser.add_argument('-sc', '--social_component', type=float, default=2, help="Value of social component")
    parser.add_argument('-ff', '--function_flag', action='store_true', help="Flag defining which function to use")
    parser.add_argument('-sf', '--stagnation_flag', action='store_true', help="Flag whether to use stagnation")
    parser.add_argument('-sl', '--stagnation_limit', type=int, default=10, help="Stop limit for stagnation")
    parser.add_argument('-a', '--animated', action='store_true', help="Flag whether to generate a gif")
    return parser


def main():
    # GIF generation
    def animated_heatmap(i):
        plt.clf()
        positions_values = pso.optimize()
        x_positions = [pos_val[0][0] for pos_val in positions_values]
        y_positions = [pos_val[0][1] for pos_val in positions_values]
        values = [pos_val[1] for pos_val in positions_values]

        scatter = plt.scatter(x_positions, y_positions, c=values, marker='x', cmap='viridis')
        plt.colorbar(scatter, label='Wartość funkcji')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Iteracja {i + 1}')
        plt.grid(True)
        plt.xlim(bounds[0])
        plt.ylim(bounds[1])
        if i == args.iterations - 1:
            # plt.clf()
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(x_positions, y_positions, c=values, marker='x', cmap='viridis')
            plt.colorbar(scatter, label='Wartość funkcji')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Heatmapa punktowa cząstek dla ostaniej iteracji w wygenerowanej animacji')
            plt.grid(True)
            plt.savefig(f'PSO_{args.iterations}_iteration.png', bbox_inches='tight')

    # Main Code
    parser = setup_argparse()
    args = parser.parse_args()
    bounds = [(args.x1, args.x2), (args.y1, args.y2)]
    pso = ParticleSwarmOptimizer(function, bounds, args.particle_number, args.iterations, args.inertia_weight,
                                 args.cognitive_component, args.social_component, args.function_flag,
                                 args.stagnation_flag, args.stagnation_limit, args.animated)
    if args.animated:
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, animated_heatmap, frames=args.iterations)
        ani.save('pso_heatmap_animation.gif', writer='pillow', fps=math.ceil(args.iterations / 50))
        print(
            f'Najlepsze położenie: (x, y) = ({pso.global_best_position[0]:.15f}, {pso.global_best_position[1]:.15f})\n'
            f'Najlepsza wartość: {pso.global_best_value}')
    else:
        heatmap(args)


def heatmap(args):
    x_positions, y_positions, values = [], [], []
    bounds = [(args.x1, args.x2), (args.y1, args.y2)]

    for _ in range(20):
        pso = ParticleSwarmOptimizer(function, bounds, args.particle_number, args.iterations, args.inertia_weight,
                                     args.cognitive_component, args.social_component, args.function_flag,
                                     args.stagnation_flag, args.stagnation_limit, args.animated)
        best_position, best_value = pso.optimize()
        x_positions.append(best_position[0])
        y_positions.append(best_position[1])
        values.append(best_value)

    min_index = np.argmin(values)
    print(f'Najlepsze położenie: (x, y) = ({x_positions[min_index]:.15f}, {y_positions[min_index]:.15f})\n'
          f'Najlepsza wartość: {min(values)}')

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_positions, y_positions, c=values, marker='x', cmap='viridis')
    plt.colorbar(scatter, label='Wartość funkcji')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Heatmapa punktowa uzyskanych wyników dla 20 wywołań algorytmu')
    plt.grid(True)
    # plt.savefig(f'plots/sf{args.stagnation_flag}sl{args.stagnation_limit}ff{args.function_flag}.png',
    # bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
