import argparse
import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import DifferentialEvolution
from pso_de import ParticleSwarmOptimizer
from functions import choose_fun


def plot_best_values(best_values, title):
    plt.figure(figsize=(10, 5))
    iterations = np.arange(1, len(best_values) + 1)
    plt.plot(iterations, best_values)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Best Value')
    plt.grid(True)
    plt.show()


def plot_both(best_pso, best_de):
    plt.figure(figsize=(10, 5))
    iterations = np.arange(1, len(best_pso) + 1)
    plt.plot(iterations, best_pso, label='PSO')
    plt.plot(iterations, best_de, label='DE')
    plt.title("Best Value per Iteration: PSO vs DE")
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_results(best_pso, best_de, worst_pso, worst_de):
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(best_pso) + 1)
    plt.plot(iterations, best_pso, label='PSO', lw=2, color='orangered')
    plt.plot(iterations, best_de, label='DE', lw=2, color='darkviolet')
    plt.title("Value per Iteration: Best PSO vs Best DE")
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/best_.png', bbox_inches='tight')
    # plt.show()
    # plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, worst_pso, label='PSO', lw=2, color='orangered')
    plt.plot(iterations, worst_de, label='DE', lw=2, color='darkviolet')
    plt.title("Value per Iteration: Worst PSO vs Worst DE")
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/worst_.png', bbox_inches='tight')
    # plt.show()
    # plt.close()


def setup_argparse():
    parser = argparse.ArgumentParser(description="Particle Swarm Optimization vs Differential Evolution")
    # Global options
    parser.add_argument('-f', '--function', type=int, default=1, help='Function to optimize (1-6)')
    parser.add_argument('-sf', '--scope_flag', action='store_true', help='Whether to use default scope for chosen function')
    parser.add_argument('-s', '--scope', type=tuple[float, float], help='Scope if scope_flag is not set')
    parser.add_argument('-a', '--algorithm', type=str, default='both', help='Algorithm to use. Choose from: [de, pso, both]')
    parser.add_argument('-d', '--dimensions', type=int, default=20, help='number of dimensions')
    parser.add_argument('-i', '--iterations', type=int, default=100, help="Number of iterations")
    parser.add_argument('-dw', '--differential_weight', type=float, default=0.5, help='Value of differential weight')
    parser.add_argument('-cp', '--crossover_probability', type=float, default=0.5, help='Value of crossover probability')
    # DE options
    parser.add_argument('-ps', '--population_size', type=int, default=50, help='Population size, DE ONLY')
    # PSO options
    parser.add_argument('-pn', '--particle_number', type=int, default=30, help="Number of particles, PSO ONLY")
    parser.add_argument('-iw', '--inertia_weight', type=float, default=0.5, help="Weight of inertia, PSO ONLY")
    parser.add_argument('-cc', '--cognitive_component', type=float, default=2, help="Value of cognitive component, PSO ONLY")
    parser.add_argument('-sc', '--social_component', type=float, default=2, help="Value of social component, PSO ONLY")
    # Additional options
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether to print detailed information')
    return parser


def main():
    # Main Code
    parser = setup_argparse()
    args = parser.parse_args()
    function, scope = choose_fun(args.function, True)
    if not args.scope_flag:
        string_numbers = ''.join(args.scope).replace('(', '').replace(')', '')
        split_numbers = string_numbers.split(',')
        scope = tuple(map(float, split_numbers))

    optimizer_pso = None
    optimizer_de = None
    if args.algorithm in ['pso', 'both']:
        optimizer_pso = ParticleSwarmOptimizer(
            fun=function,
            scope=scope,
            dimension=args.dimensions,
            F=args.differential_weight,
            CR=args.crossover_probability,
            num_particles=args.particle_number,
            max_iter=args.iterations,
            iw=args.inertia_weight,
            cc=args.cognitive_component,
            sc=args.social_component,
            verbose=args.verbose
        )
        # Optimize with PSO
        if args.algorithm == 'pso':
            best_position, best_value, best_iter_pso = optimizer_pso.optimize()
            print(f"PSO Best position: {best_position}\nBest value: {best_value}")
            plot_best_values(best_iter_pso, "Best Value per Iteration for PSO")

    if args.algorithm in ['de', 'both']:
        optimizer_de = DifferentialEvolution(
            function=function,
            scope=scope,
            dimension=args.dimensions,
            population_size=args.population_size,
            F=args.differential_weight,
            CR=args.crossover_probability,
            generations=args.iterations,
            verbose=args.verbose
        )
        # Optimize with DE
        if args.algorithm == 'de':
            best_position_de, best_value_de, best_iter_de = optimizer_de.optimize()
            print(f"DE Best position: {best_position_de}\nBest value: {best_value_de}")
            plot_best_values(best_iter_de, "Best Value per Iteration for DE")

    if args.algorithm == 'both':
        best_position_pso, best_value_pso, best_iter_pso = optimizer_pso.optimize()
        print(f"PSO-DE Best PSO position: {best_position_pso}\nBest PSO value: {best_value_pso}")
        best_position_de, best_value_de, best_iter_de = optimizer_de.optimize()
        print(f"PSO-DE Best DE position: {best_position_de}\nBest DE value: {best_value_de}")
        plot_both(best_iter_pso, best_iter_de)


def generate_plots():
    function, scope = choose_fun(1, True)
    optimizer_pso = ParticleSwarmOptimizer(
        fun=function, scope=scope, dimension=30, F=0.3, CR=0.5,
        num_particles=50, max_iter=100, iw=0.5, cc=2, sc=2, verbose=False
    )
    optimizer_de = DifferentialEvolution(
        function=function, scope=scope, dimension=30, population_size=50,
        F=0.3, CR=0.5, generations=100, verbose=False
    )
    global_pso, global_de =  float('inf'), float('inf')
    worst_pso, worst_de =  -1, -1
    values_pso, values_de, values_pso_worst, values_de_worst = 0, 0, 0, 0
    for _ in range(10):
        _, best_pso, best_iter_pso = optimizer_pso.optimize()
        _, best_de, best_iter_de = optimizer_de.optimize()
        if best_pso < global_pso:
            values_pso = best_iter_pso
            global_pso = best_pso
        if best_de < global_de:
            values_de = best_iter_de
            global_de = best_de
        if best_pso > worst_pso:
            values_pso_worst = best_iter_pso
            worst_pso = best_pso
        if best_de > worst_de:
            values_de_worst = best_iter_de
            worst_de = best_de
    plot_results(values_pso, values_de, values_pso_worst, values_de_worst)


if __name__ == "__main__":
    main()
    #generate_plots()
