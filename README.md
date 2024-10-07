# Algorithm optimization

This project is an implementation of several popular optimization algorithms.

## Used algorithms

* Grey Wolf Algorithm
* Multi-Swarm Slime Mould Algorithm
* Bat Algorithm
* Butterfly Optimization Algorithm
* Particle Swarm Optimization
* Particle Swarm Optimization with modification from Differential Evolution (mutation and crossover)
* Differential Evolution

## Features

* **Optimization**: Use any of the above-listed algorithms to optimize functions within a given scope.
* **Plotting**: The program generates plots that track the best solution value for each iteration of the algorithm.
* **Comparative Results**: You can compare the results of SMA and GWO using custom functions over multiple iterations.
* **Function Customization**: Users can specify the mathematical function to optimize, along with its scope, dimensionality, and iterations.

## Setup and Dependencies

The project uses Python and the following dependencies:
- `numpy`
- `matplotlib`
- `argparse`
- Custom modules for the optimization algorithms (`de.py`, `pso_de.py`, `ba.py`, `boa.py`, `sma.py`, `gwo.py`)

## Usage

You can run the script using the command line. The script accepts several arguments to customize the optimization process.

### Basic Usage:

```sh
python main.py -a [algorithm] -f [function_id]
```

Where `algorithm` can be one of the following:
- `de` (Differential Evolution)
- `pso` (Particle Swarm Optimization)
- `ba` (Bat Algorithm)
- `boa` (Butterfly Optimization Algorithm)
- `sma` (Slime Mould Algorithm)
- `gwo` (Grey Wolf Optimizer)

### Example:

```sh
python main.py -a de -f 1 -ps 50 -d 30 -i 200
```

## Customization Options

You can adjust several parameters depending on the algorithm and function:
* `-f`, `--function`: Select which function to optimize (1–6)
* `-sf`, `--scope_flag`: Use default function scope
* `-s`, `--scope`: Define a custom scope as a tuple (min, max)
* `-d`, `--dimensions`: Number of dimensions (default: 20)
* `-i`, `--iterations`: Number of iterations (default: 100)
* `-f`, `--function`: Select which function to optimize (1–6)
* `-sf`, `--scope_flag`: Use default function scope
* `-s`, `--scope`: Define a custom scope as a tuple (min, max)
* `-d`, `--dimensions`: Number of dimensions (default: 20)
* `-i`, `--iterations`: Number of iterations (default: 100)
* `-v`, `--verbose`: Whether to print detailed information
* `-dw`, `--differential_weight`: Value of differential weight, PSO-DE ONLY (default: 0.5)
* `-cp`, `--crossover_probability`: Value of crossover probability, PSO-DE ONLY (default: 0.5)
* `-ps`, `--population_size`: Population size, DE/BA/BOA ONLY (default: 50) 
* `-pn`, `--particle_number`: Number of particles, PSO ONLY (default: 30) 
* `-iw`, `--inertia_weight`: Weight of inertia, PSO ONLY (default: 0.5) 
* `-cc`, `--cognitive_component`: Value of cognitive component, PSO ONLY (default: 2)
* `-sc`, `--social_component`: Value of social component, PSO ONLY (default: 2)
* `-l`, `--loudness`: Initial value of loudness, BA ONLY (default: 0.7)
* `-pr`, `--pulse_rate`: Initial value of pulse emission rate, BA ONLY (default: 0.5) 
* `--alpha`: Value of alpha parameter that affects balance between exploration and exploitation, BA ONLY (default: 0.9) 
* `--gamma`: Value of gamma parameter that affects the convergence speed, BA ONLY (default: 0.9)
* `--f_min`: Minimum frequency value, BA ONLY (default: 0)
* `--f_max`: Maximum frequency value, BA ONLY (default: 2)
* `-sm`, `--sensory_modality`: Value of sensory modality, BOA ONLY (default: 0.4) 
* `-pe`, `--power_exponent`: Value of power exponent, BOA ONLY (default: 0.8)
* `-sp`, `--switch_probability`: Value of switch probability, BOA ONLY (default: 0.8)
* `-ss`, `--swarm_size`: Size of swarm, SMA ONLY (default: 20)
* `-ns`, `--num_swarms`: Number of swarms, SMA ONLY (default: 10)
* `-z`, `--z`: Probability of random updating position, SMA ONLY (default: 0.6)
* `-mt`, `--migration_threshold`: Migration threshold, SMA ONLY (default: 0.4)
* `-mr`, `--mutation_rate`: Mutation rate, SMA/GWO ONLY (default: 0.4) 
* `-nil`, `--no_improve_limit`: Limit for result stagnation, SMA/GWO ONLY (default: 30)
  
## License
[MIT](https://github.com/alwejsWTF/TechnikiInteligentnejAnalizyDanych/blob/main/LICENSE)
