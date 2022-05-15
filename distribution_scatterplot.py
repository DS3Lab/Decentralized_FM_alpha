import numpy as np

with open('random_scheduling.npy', 'rb') as f:
    random_scheduling_results = np.load(f)

with open('vanilla_evolutionary_scheduling.npy', 'rb') as f:
    vanilla_evolutionary_scheduling_results = np.load(f)

with open('heuristic_evolutionary_scheduling.npy', 'rb') as f:
    heuristic_evolutionary_scheduling_results = np.load(f)

assert(len(random_scheduling_results) == len(
    vanilla_evolutionary_scheduling_results))
assert(len(random_scheduling_results) == len(
    heuristic_evolutionary_scheduling_results))
