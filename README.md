#GMEL (Gpu-based MEtaheuristics Library)

## Description

Nowadays, any personal computer includes a GPU that allows to use its parallelism to speedup computations. Unfortunately, it is not a trivial task to take advantage of such parallel architectures. In this repo, we implement a library of bio-inspired metaheuristics providing automatic parallelizations. The library is implemented by using CUDA, and it in cludes parallel versions of metaheuristics for both continuous and discrete domains.

## Metaheuristics

In this repo the parallel `par_*` and `seq_*` sequential versions of some metaheuristics can be found. PSO (Particle Swarm Optimization), DE (Differential Evolution) and ABC (Artificial Bee Colony) are applied to continuous benchmark problems, while MTPSO (Modified Turbulent Particle Swarm Optimization) and ACO (Ant Colony Optimization) are applied to some well known NP-complete discrete problems.

## Generators

In this repository we also include two auxiliar scripts, writen in python. `Grap_generator.py` can be used to generate some graph examples in order to test the performance of ACO when solving Travelling Salesman Problem. `MapGen.py` genetares examples of the Graph Coloring problem, which are used to test the performace of MTPSO.
