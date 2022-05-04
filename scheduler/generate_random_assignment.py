import numpy as np


def random_assignment_0_datacenter(nodes=64):
    np.random.seed(2022)
    gpu_per_instances = min(nodes // 2, 8)
    instances = nodes // gpu_per_instances
    arr = np.arange(1, nodes)
    np.random.shuffle(arr)
    result = arr.tolist()
    result.insert(0, 0)
    print('nodes_per_node=(', end='')
    for i in range(instances):
        print(gpu_per_instances, end='' if i==instances-1 else ' ')
    print(')')
    print('rank_map=(', end='')
    for i in range(len(result)):
        val = result[i]
        print(val, end='' if i==nodes-1 else ' ')
    print(')')


random_assignment_0_datacenter()