import argparse
import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import DifferentialEvolution
from pso_de import ParticleSwarmOptimizer
from functions import choose_fun


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
        scope = args.scope

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
        bounds_pso = optimizer_pso.bounds
        # Optimize with PSO
        if args.algorithm == 'pso':
            best_position, best_value = optimizer_pso.optimize()
            print(f"PSO Best position: {best_position}, Best value: {best_value}")

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
        bounds_de = optimizer_de.bounds
        # Optimize with DE
        if args.algorithm == 'de':
            best_position_de, best_value_de = optimizer_de.optimize()
            print(f"DE Best position: {best_position_de}, Best value: {best_value_de}")

    if args.algorithm == 'both':
        best_position, best_value = optimizer_pso.optimize()
        print(f"PSO-DE Best PSO position: {best_position}, Best PSO value: {best_value}")
        best_position_de, best_value_de = optimizer_de.optimize()
        print(f"PSO-DE Best DE position: {best_position_de}, Best DE value: {best_value_de}")


if __name__ == "__main__":
    main()
