from operator import le
import numpy as np
import itertools

# GPT-3 XL
batch_size = 0.5e6
layer_size = 24
para_size = 1.3e9

# physical topology
num_devices = 16
bandwidth = 1e9  # 1GB/s
# https://www.cloudping.co/grid
peer_latency = np.random.rand(num_devices, num_devices) * 200
peer_latency = np.tril(peer_latency) + np.tril(peer_latency, -1).T

# assigned task
batch_size_per_task = 0.25e6
layer_size_per_task = 3
send_activation_size = 1e6 * batch_size_per_task  # bytes
send_gradient_size = para_size * layer_size_per_task / layer_size  # bytes


def multiway_partition(num_vertices=None, way=None):
    # https://dl.acm.org/doi/pdf/10.5555/2933718.2933740
    subsets = [set() for _ in range(way)]
    subset_size = num_vertices/way
    vertix_idx = 0
    while vertix_idx < num_vertices:
        subset_idx = np.random.randint(way)
        if len(subsets[subset_idx]) >= subset_size:
            continue
        else:
            subsets[subset_idx].add(vertix_idx)
            vertix_idx += 1
    return subsets


def compute_data_parallel_cost(candidate_partition=None):
    data_parallel_cost = 0
    for partition in candidate_partition:
        within_partition_cost = float('inf')
        for primary in partition:
            cur_cost = 0
            for secondary in partition:
                if primary != secondary:
                    cur_cost += peer_latency[primary, secondary]
            if cur_cost < within_partition_cost:
                within_partition_cost = cur_cost
        data_parallel_cost += within_partition_cost
    return data_parallel_cost


class open_loop_tsp:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5388488
    def __init__(self, cost_matrix, start_node):
        self.cost_matrix = cost_matrix
        self.num_nodes = self.cost_matrix.shape[0]
        self.start_node = start_node
        self.dp_table = np.full(
            shape=(self.num_nodes, pow(2, self.num_nodes)), fill_value=np.inf)
        self.trace_table = np.zeros(
            shape=(self.num_nodes, pow(2, self.num_nodes)))

    def convert(self, future_nodes):
        binary_future_nodes = 0
        for future_node in future_nodes:
            binary_future_nodes += pow(2, future_node)
        return binary_future_nodes

    def solve(self, node, future_nodes):
        if len(future_nodes) == 0:
            # closed loop tsp problem: return self.cost_matrix[node][self.start_node]
            # open loop tsp problem: return 0
            return 0

        all_distance = []
        for next_node in future_nodes:
            next_future_nodes = future_nodes.copy()
            next_future_nodes.remove(next_node)
            binary_next_future_nodes = self.convert(next_future_nodes)
            if self.dp_table[next_node][binary_next_future_nodes] == np.inf:
                all_distance.append(
                    self.cost_matrix[node][next_node] + self.solve(next_node, next_future_nodes))
            else:
                all_distance.append(
                    self.cost_matrix[node][next_node] + self.dp_table[next_node][binary_next_future_nodes])

        min_distance = min(all_distance)
        next_node = future_nodes[all_distance.index(min_distance)]

        binary_future_nodes = self.convert(future_nodes)
        self.dp_table[node][binary_future_nodes] = min_distance
        self.trace_table[node][binary_future_nodes] = next_node
        return min_distance

    def get_shortest_path(self):
        future_nodes = list(range(self.num_nodes))
        future_nodes.remove(self.start_node)
        cost = self.solve(self.start_node, future_nodes)

        path = [self.start_node]
        cur_node = self.start_node
        while len(future_nodes) > 0:
            binary_future_nodes = self.convert(future_nodes)
            cur_node = int(self.trace_table[cur_node][binary_future_nodes])
            future_nodes.remove(cur_node)
            path.append(cur_node)
        return cost, path


def compute_pipeline_parallel_cost(candidate_partition=None):
    way = len(candidate_partition)

    # bipartite matching
    crose_partition_cost = np.zeros(shape=(way, way))
    for i in range(way):
        for j in range(i+1, way):
            bipartite_matches = []
            for x in itertools.permutations(candidate_partition[i]):
                bipartite_matches.append(list(zip(x, candidate_partition[j])))
            all_transfer_times = []
            for bipartite_match in bipartite_matches:
                cur_transfer_times = []
                for pair in bipartite_match:
                    cur_transfer_times.append(peer_latency[pair[0], pair[1]])
                all_transfer_times.append(max(cur_transfer_times))
            crose_partition_cost[i, j] = min(all_transfer_times)

    crose_partition_cost = crose_partition_cost + crose_partition_cost.T

    import time
    start = time.perf_counter()
    pipeline_parallel_cost = []
    pipeline_parallel_path = []
    for start_node in range(way):
        tsp = open_loop_tsp(crose_partition_cost, start_node)
        cost, path = tsp.get_shortest_path()
        pipeline_parallel_cost.append(cost)
        pipeline_parallel_path.append(path)
    dp_pipeline_parallel_cost = min(pipeline_parallel_cost)
    end = time.perf_counter()
    print("open loop tsp program solver")
    print("dynamic programming: " + str(end - start) + " seconds")

    start = time.perf_counter()
    pipeline_parallel_cost = float('inf')
    pipeline_parallel_path = None
    for path in itertools.permutations(range(way)):
        cur_cost = 0
        for i in range(way - 1):
            cur_cost += crose_partition_cost[path[i], path[i+1]]
        if cur_cost < pipeline_parallel_cost:
            pipeline_parallel_cost = cur_cost
            pipeline_parallel_path = path
    end = time.perf_counter()
    print("brute force: " + str(end - start) + " seconds")

    assert(dp_pipeline_parallel_cost == pipeline_parallel_cost)
    return pipeline_parallel_cost, pipeline_parallel_path


if __name__ == "__main__":
    assert(batch_size % batch_size_per_task == 0)
    assert(layer_size % layer_size_per_task == 0)
    assert(num_devices == batch_size * layer_size /
           (batch_size_per_task * layer_size_per_task))
    candidate_partition = multiway_partition(
        num_vertices=num_devices, way=int(layer_size/layer_size_per_task))
    print("candidate partition: " + str(candidate_partition))

    data_parallel_cost = compute_data_parallel_cost(
        candidate_partition=candidate_partition)
    print("data parallel cost: " + str(data_parallel_cost))

    pipeline_parallel_cost, pipeline_parallel_path = compute_pipeline_parallel_cost(
        candidate_partition)
    print("pipeline parallel cost: " + str(pipeline_parallel_cost))
    print("pipeline parallel path: " + str(pipeline_parallel_path))
