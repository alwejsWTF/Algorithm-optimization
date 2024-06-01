import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from de import DifferentialEvolution
from pso_de import ParticleSwarmOptimizer
from ba import BatAlgorithm
from boa import ButterflyOptimizer
from sma import MultiSwarmSMA
from gwo import GreyWolfOptimizer
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


def plot_results(best_sma, best_gwo, worst_sma, worst_gwo, avg_sma, avg_gwo):
    plt.figure(figsize=(10, 6))
    iterations = np.arange(1, len(best_sma) + 1)
    plt.plot(
        iterations, best_sma, label='Best SMA', lw=2, color='mediumseagreen'
    )
    plt.plot(
        iterations, best_gwo, label='Best GWO', lw=2, color='mediumspringgreen'
    )
    plt.plot(
        iterations, worst_sma, label='Worst SMA', lw=2, color='mediumvioletred'
    )
    plt.plot(iterations, worst_gwo, label='Worst GWO', lw=2, color='crimson')
    plt.plot(iterations, avg_sma, label='Avg SMA', lw=2, color='mediumblue')
    plt.plot(iterations, avg_gwo, label='Avg GWO', lw=2, color='royalblue')
    plt.title("Fitness Over Iteration: SMA vs GWO (iterations - 100)(function rastrigin)")
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/iterations_100.png', bbox_inches='tight')
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
        help="Algorithm to use. Choose from: [de, pso, ba, boa, sma, gwo]"
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
    # DE, BA, BOA options
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
    # SMA
    parser.add_argument(
        '-ss', '--swarm_size', type=int, default=20,
        help='Size of swarm for SMA')
    parser.add_argument(
        '-ns', '--num_swarms', type=int, default=10,
        help='Number of swarms for SMA')
    parser.add_argument(
        '-z', '--z', type=float, default=0.6,
        help='probability of random updating position for SMA')
    parser.add_argument(
        '-mt', '--migration_threshold', type=float, default=0.4,
        help='Migration threshold for SMA')
    # SMA and GWO
    parser.add_argument(
        '-mr', '--mutation_rate', type=float, default=0.4,
        help='Mutation rate for SMA and GWO')
    parser.add_argument(
        '-nil', '--no_improve_limit', type=int, default=30,
        help='Limit for result stagnation for SMA and GWO')
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
    elif args.algorithm == 'sma':
        optimizer_sma = MultiSwarmSMA(
            function=function,
            scope=scope,
            dimension=args.dimensions,
            swarm_size=args.swarm_size,
            iterations=args.iterations,
            num_swarms=args.num_swarms,
            z=args.z,
            migration_threshold=args.migration_threshold,
            mutation_rate=args.mutation_rate,
            no_improve_limit=args.no_improve_limit,
            verbose=args.verbose
        )
        # Optimize with SMA
        best_position_sma, best_value_sma, best_iter_sma = (
            optimizer_sma.optimize()
        )
        print(f"BOA Best position:\n{best_position_sma}\n"
              f"BOA Best value: {best_value_sma}")
        plot_best_values(best_iter_sma, "Best Value per Iteration for SMA")
    elif args.algorithm == 'gwo':
        optimizer_gwo = GreyWolfOptimizer(
            function=function,
            scope=scope,
            dimension=args.dimensions,
            population_size=args.population_size,
            iterations=args.iterations,
            mutation_rate=args.mutation_rate,
            no_improve_limit=args.no_improve_limit,
            verbose=args.verbose
        )
        # Optimize with GWO
        best_position_gwo, best_value_gwo, best_iter_gwo = (
            optimizer_gwo.optimize()
        )
        print(f"BOA Best position:\n{best_position_gwo}\n"
              f"BOA Best value: {best_value_gwo}")
        plot_best_values(best_iter_gwo, "Best Value per Iteration for GWO")


def generate_plots():
    loop_counts = 30
    function, scope = choose_fun(1, True)
    values_sma = np.zeros(loop_counts)
    values_gwo = np.zeros(loop_counts)
    results_sma = np.zeros((loop_counts, 100))
    results_gwo = np.zeros((loop_counts, 100))
    for i in range(loop_counts):
        optimizer_sma = MultiSwarmSMA(
            function=function, scope=scope, dimension=30, swarm_size=25,
            iterations=100, num_swarms=10, z=0.5, migration_threshold=0.4,
            mutation_rate=0.2, no_improve_limit=20, verbose=False
        )
        optimizer_gwo = GreyWolfOptimizer(
            function=function, scope=scope, dimension=30, population_size=100,
            iterations=100, mutation_rate=0.2, no_improve_limit=20,
            verbose=False
        )
        _, best_sma, best_iter_sma = optimizer_sma.optimize()
        _, best_gwo, best_iter_gwo = optimizer_gwo.optimize()
        values_sma[i] = best_sma
        values_gwo[i] = best_gwo
        results_sma[i] = best_iter_sma
        results_gwo[i] = best_iter_gwo

    avg_sma = np.mean(values_sma)
    avg_gwo = np.mean(values_gwo)
    differences_sma = np.abs(np.array(values_sma) - avg_sma)
    differences_gwo = np.abs(np.array(values_gwo) - avg_gwo)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    plot_results(
        results_sma[np.argmin(values_sma)],
        results_gwo[np.argmin(values_gwo)],
        results_sma[np.argmax(values_sma)],
        results_gwo[np.argmax(values_gwo)],
        results_sma[np.argmin(differences_sma)],
        results_gwo[np.argmin(differences_gwo)],
    )


if __name__ == "__main__":
    # main()
    generate_plots()
