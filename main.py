import argparse
import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import DifferentialEvolution
from pso_de import ParticleSwarmOptimizer
from bat_algorithm import BatAlgorithm
from boa import ButterflyOptimizer
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
    function, scope = choose_fun(1, True)
    optimizer_pso = ParticleSwarmOptimizer(
        fun=function, scope=scope, dimension=30, F=0.3, CR=0.5,
        num_particles=50, max_iter=100, iw=0.5, cc=2, sc=2, verbose=False
    )
    optimizer_de = DifferentialEvolution(
        function=function, scope=scope, dimension=30, population_size=50,
        F=0.3, CR=0.5, generations=100, verbose=False
    )
    global_pso, global_de = float('inf'), float('inf')
    worst_pso, worst_de = -1, -1
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
    # generate_plots()
