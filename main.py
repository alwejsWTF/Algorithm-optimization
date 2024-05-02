import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from differential_evolution import DifferentialEvolution
from pso_de import ParticleSwarmOptimizer
from bat_algorithm import BatAlgorithm
from boa import ButterflyOptimizer
from functions import choose_fun


def plot_best_values(best_values, title):
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(best_values) + 1)
    plt.plot(iterations, best_values)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Best Value')
    plt.grid(True)
    plt.show()


def plot_results(best_pso, best_de, worst_pso, worst_de, avg_ba, avg_boa):
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(best_pso) + 1)
    plt.plot(iterations, best_pso, label='BA', lw=2, color='orangered')
    plt.plot(iterations, best_de, label='BOA', lw=2, color='darkviolet')
    plt.title("Best Fitness Over Iteration: Best BA vs Best BOA")
    plt.xlabel('Iteration')
    plt.ylabel('Best Value')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/best_.png', bbox_inches='tight')
    # plt.show()
    # plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, worst_pso, label='BA', lw=2, color='orangered')
    plt.plot(iterations, worst_de, label='BOA', lw=2, color='darkviolet')
    plt.title("Worst Fitness Over Iterations: Worst BA vs Worst BOA")
    plt.xlabel('Iteration')
    plt.ylabel('Worst Value')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/worst_.png', bbox_inches='tight')
    # plt.show()
    # plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_ba, label='BA', lw=2, color='orangered')
    plt.plot(avg_boa, label='BOA', lw=2, color='darkviolet')
    plt.title("Average Best Fitness Over Iterations: BA vs BOA")
    plt.xlabel('Iteration')
    plt.ylabel('Average Value')
    # plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/avg_.png', bbox_inches='tight')
    # plt.show()
    # plt.close()


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Different optimization algorithms"
    )
    # Global options
    parser.add_argument(
        '-f', '--function', type=int, default=1,
        help="Function to optimize (1-6)"
    )
    parser.add_argument(
        '-sf', '--scope_flag', action='store_true',
        help="Whether to use default scope for chosen function"
    )
    parser.add_argument(
        '-s', '--scope', type=tuple[float, float],
        help="Scope if scope_flag is not set"
    )
    parser.add_argument(
        '-a', '--algorithm', type=str, required=True,
        help="Algorithm to use. Choose from: [de, pso, ba, boa]"
    )
    parser.add_argument(
        '-d', '--dimensions', type=int, default=20,
        help="Number of dimensions"
    )
    parser.add_argument(
        '-i', '--iterations', type=int, default=100,
        help="Number of iterations"
    )
    # PSO-DE options
    parser.add_argument(
        '-dw', '--differential_weight', type=float, default=0.5,
        help="Value of differential weight, PSO-DE ONLY"
    )
    parser.add_argument(
        '-cp', '--crossover_probability', type=float, default=0.5,
        help="Value of crossover probability, PSO-DE ONLY"
    )
    # DE, BA options
    parser.add_argument(
        '-ps', '--population_size', type=int, default=50,
        help="Population size, DE, BA ONLY"
    )
    # PSO options
    parser.add_argument(
        '-pn', '--particle_number', type=int, default=30,
        help="Number of particles, PSO ONLY"
    )
    parser.add_argument(
        '-iw', '--inertia_weight', type=float, default=0.5,
        help="Weight of inertia, PSO ONLY"
    )
    parser.add_argument(
        '-cc', '--cognitive_component', type=float, default=2,
        help="Value of cognitive component, PSO ONLY"
    )
    parser.add_argument(
        '-sc', '--social_component', type=float, default=2,
        help="Value of social component, PSO ONLY"
    )
    # BA options
    parser.add_argument(
        '-l', '--loudness', type=float, default=0.7,
        help="Initial value of loudness, BA ONLY"
    )
    parser.add_argument(
        '-pr', '--pulse_rate', type=float, default=0.5,
        help="Initial value of pulse emission rate, BA ONLY"
    )
    parser.add_argument(
        '--alpha', type=float, default=0.9,
        help="Value of alpha parameter that affects balance "
             "between exploration and exploitation, BA ONLY"
    )
    parser.add_argument(
        '--gamma', type=float, default=0.9,
        help="Value of gamma parameter that affects "
             "the convergence speed, BA ONLY"
    )
    parser.add_argument(
        '--f_min', type=float, default=0,
        help="Minimum frequency value, BA ONLY"
    )
    parser.add_argument(
        '--f_max', type=float, default=2,
        help="Maximum frequency value, BA ONLY"
    )
    # BOA options
    parser.add_argument(
        '-sm', '--sensory_modality', type=float, default=0.4,
        help="Value of sensory modality, BOA ONLY"
    )
    parser.add_argument(
        '-pe', '--power_exponent', type=float, default=0.8,
        help="Value of power exponent, BOA ONLY"
    )
    parser.add_argument(
        '-sp', '--switch_probability', type=float, default=0.8,
        help="Value of switch probability, BOA ONLY"
    )
    # Additional options
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Whether to print detailed information')
    return parser


def main():
    # Main Code
    plt.style.use('seaborn-v0_8-whitegrid')
    parser = setup_argparse()
    args = parser.parse_args()
    function, scope = choose_fun(args.function, True)
    if not args.scope_flag:
        string_numbers = (
            ''.join(args.scope)
            .replace('(', '')
            .replace(')', '')
        )
        split_numbers = string_numbers.split(',')
        scope = tuple(map(float, split_numbers))

    if args.algorithm == 'pso':
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
        best_position_pso, best_value_pso, best_iter_pso = (
            optimizer_pso.optimize()
        )
        print(f"PSO Best position:\n{best_position_pso}\n"
              f"PSO Best value: {best_value_pso}")
        plot_best_values(best_iter_pso, "Best Value per Iteration for PSO")
    elif args.algorithm == 'de':
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
        best_position_de, best_value_de, best_iter_de = optimizer_de.optimize()
        print(f"DE Best position:\n{best_position_de}\n"
              f"DE Best value: {best_value_de}")
        plot_best_values(best_iter_de, "Best Value per Iteration for DE")
    elif args.algorithm == 'ba':
        optimizer_ba = BatAlgorithm(
            function=function,
            scope=scope,
            dimension=args.dimensions,
            population_size=args.population_size,
            iterations=args.iterations,
            loudness=args.loudness,
            pulse_rate=args.pulse_rate,
            alpha=args.alpha,
            gamma=args.gamma,
            f_min=args.f_min,
            f_max=args.f_max,
            verbose=args.verbose
        )
        # Optimize with BA
        best_position_ba, best_value_ba, best_iter_ba = optimizer_ba.optimize()
        print(f"BA Best position:\n{best_position_ba}\n"
              f"BA Best value: {best_value_ba}")
        plot_best_values(best_iter_ba, "Best Value per Iteration for BA")
    elif args.algorithm == 'boa':
        optimizer_boa = ButterflyOptimizer(
            function=function,
            scope=scope,
            dimension=args.dimensions,
            population_size=args.population_size,
            iterations=args.iterations,
            sensory_modality=args.sensory_modality,
            power_exponent=args.power_exponent,
            switch_probability=args.switch_probability,
            verbose=args.verbose
        )
        # Optimize with BOA
        best_position_boa, best_value_boa, best_iter_boa = (
            optimizer_boa.optimize()
        )
        print(f"BOA Best position:\n{best_position_boa}\n"
              f"BOA Best value: {best_value_boa}")
        plot_best_values(best_iter_boa, "Best Value per Iteration for BOA")


def generate_plots():
    loop_counts = 30
    function, scope = choose_fun(1, True)
    optimizer_ba = BatAlgorithm(
        function=function, scope=scope, dimension=30, population_size=50,
        iterations=100, loudness=0.7, pulse_rate=0.5, alpha=0.9, gamma=0.9,
        f_min=0, f_max=2, verbose=False
    )
    optimizer_boa = ButterflyOptimizer(
        function=function, scope=scope, dimension=30, population_size=50,
        iterations=100, sensory_modality=0.4, power_exponent=0.8,
        switch_probability=0.8, verbose=False
    )
    global_ba, global_boa = float('inf'), float('inf')
    worst_ba, worst_boa = -1, -1
    values_ba, values_boa, values_ba_worst, values_boa_worst = 0, 0, 0, 0
    results_ba = np.zeros((loop_counts, optimizer_ba.iterations))
    results_boa = np.zeros((loop_counts, optimizer_boa.iterations))
    for i in range(loop_counts):
        _, best_ba, best_iter_ba = optimizer_ba.optimize()
        _, best_boa, best_iter_boa = optimizer_boa.optimize()
        if best_ba < global_ba:
            values_ba = best_iter_ba
            global_ba = best_ba
        if best_boa < global_boa:
            values_boa = best_iter_boa
            global_boa = best_boa
        if best_ba > worst_ba:
            values_ba_worst = best_iter_ba
            worst_ba = best_ba
        if best_boa > worst_boa:
            values_boa_worst = best_iter_boa
            worst_boa = best_boa
        results_ba[i] = best_iter_ba
        results_boa[i] = best_iter_boa

    avg_ba = np.mean(results_ba, axis=0)
    avg_boa = np.mean(results_boa, axis=0)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plot_results(
        values_ba, values_boa,
        values_ba_worst, values_boa_worst,
        avg_ba, avg_boa
    )


if __name__ == "__main__":
    # main()
    generate_plots()
