import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pso_de import ParticleSwarmOptimizer

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
        ani.save('pso_heatmap_animation.gif', writer='pillow', fps=np.ceil(args.iterations / 50))
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
