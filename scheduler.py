import random
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
import config


# GPT-3 XL
batch_size = 0.5e6
layer_size = 24
para_size = 1.3e9

# physical topology
num_devices = config.nodes
peer_delay = None
peer_bandwidth = None

# assigned task
batch_size_per_task = 0.625e5
layer_size_per_task = 3
send_activation_size = 4  # gigabytes
send_gradient_size = 1  # gigabytes

way = None
partition_size = None


def GCMA(nodes=None, population_size=None, trails=None):
    # https://dl.acm.org/doi/10.5555/2933718.2933740
    def normalization(parent1=None, parent2=None):
        parent1_str = [0] * num_devices
        parent2_str = [0] * num_devices
        for i in range(num_devices):
            parent1_str[parent1[i]] = i // partition_size
            parent2_str[parent2[i]] = i // partition_size

        count = np.zeros(shape=(way, way))
        for i in range(num_devices):
            count[parent1_str[i], parent2_str[i]] += 1

        map = [0] * way
        for i in range(way):
            max_idx = np.argmax(count)
            p = max_idx // way
            q = max_idx - p * way
            for j in range(way):
                count[p][j] = float('-inf')
                count[j][q] = float('-inf')
            map[q] = p

        for i in range(num_devices):
            parent2_str[i] = map[parent2_str[i]]

        return parent1_str, parent2_str

    def five_point_crossover(parent1_str=None, parent2_str=None):
        points = list(range(num_devices))
        random.shuffle(points)
        points = points[:5]

        for point in points:
            parent2_str[point] = parent1_str[point]

        partition_sizes = [0] * way
        for partition_idx in parent2_str:
            partition_sizes[partition_idx] += 1
        for i in range(num_devices):
            if partition_sizes[parent2_str[i]] > partition_size:
                for j in range(way):
                    if partition_sizes[j] < partition_size:
                        partition_sizes[j] += 1
                        break
                partition_sizes[parent2_str[i]] -= 1
                parent2_str[i] = j
        return parent2_str

    def cyclic_partitioning(offspring=None):
        def calculate_gain(cur_offspring=None, locked_v_idx=None):
            gain = np.full(shape=(num_devices, way), fill_value=-np.inf)
            for v_idx, partition_idx in enumerate(cur_offspring):
                if locked_v_idx[v_idx] == 0:
                    gain[v_idx] = 0
                    for target_idx, target_partition_idx in enumerate(cur_offspring):
                        if target_partition_idx == partition_idx:
                            gain[v_idx] += peer_delay[v_idx, target_idx]/1e3 + (
                                send_activation_size + send_gradient_size) * 8 / peer_bandwidth[v_idx, target_idx]
                        else:
                            gain[v_idx][target_partition_idx] -= peer_delay[v_idx, target_idx]/1e3 + (
                                send_activation_size + send_gradient_size) * 8 / peer_bandwidth[v_idx, target_idx]

            G_i = np.full(shape=(way), fill_value=-np.inf)
            G_i_trace = [[None, None] for i in range(way)]
            for v_idx, partition_idx in enumerate(cur_offspring):
                if locked_v_idx[v_idx] == 0:
                    if gain[v_idx][partition_idx] > G_i[partition_idx]:
                        G_i[partition_idx] = gain[v_idx][partition_idx]
                        G_i_trace[partition_idx][0] = v_idx

            G_i = np.full(shape=(way), fill_value=-np.inf)
            G_ij = np.full(shape=(way, way), fill_value=-np.inf)
            for partition_idx, trace in enumerate(G_i_trace):
                v_idx = trace[0]
                if v_idx != None:
                    for target_partition_idx, target_gain in enumerate(gain[v_idx]):
                        if target_partition_idx != partition_idx:
                            if target_gain > G_ij[partition_idx, target_partition_idx]:
                                G_ij[partition_idx,
                                     target_partition_idx] = target_gain
                            if target_gain > G_i[partition_idx]:
                                G_i[partition_idx] = target_gain
                                G_i_trace[partition_idx] = [
                                    v_idx, target_partition_idx]

            return G_ij, G_i, G_i_trace

        def move_cycles(offspring=None):
            sum = [0]
            locked_partition_idx = [0] * way
            locked_v_idx = [0] * num_devices
            offsprings = [offspring]
            for _ in range(way):  # how many cycles
                cur_offspring = offsprings[-1].copy()
                movements = []
                epsilon = []
                tau = []
                G_ij, G_i, G_i_trace = calculate_gain(
                    cur_offspring, locked_v_idx)
                S0 = Si = np.argmax(G_i)
                for _ in range(num_devices):  # how many movement per cycle
                    v_idx, Pv = G_i_trace[Si]
                    if v_idx == None:
                        v_idx = movements[-1][0]
                        Pv = S0
                    cur_offspring[v_idx] = Pv
                    locked_v_idx[v_idx] = 1
                    locked_partition_idx[Pv] = 1
                    movements.append((v_idx, Si, Pv))
                    epsilon.append(G_i[Si])
                    tau.append(G_ij[Si, S0])
                    Si = Pv
                    if Si == S0:
                        break
                    G_ij, G_i, G_i_trace = calculate_gain(
                        cur_offspring, locked_v_idx)

                max_sum = 0
                l = 0
                for i in range(1, len(epsilon)):
                    if np.sum(epsilon[:i]) + tau[i] > max_sum:
                        max_sum = np.sum(epsilon[:i]) + tau[i]
                        l = i - 1

                for i in range(len(epsilon) - 1, l, -1):
                    cur_offspring[movements[i][0]] = movements[i][1]
                cur_offspring[movements[l][0]] = S0
                offsprings.append(cur_offspring)
                sum.append(max_sum)

                if np.sum(locked_partition_idx) == len(locked_partition_idx):
                    break

            max_sum = 0
            m = 0
            for i in range(1, len(sum)):
                if np.sum(sum[:i]) > max_sum:
                    max_sum = np.sum(sum[:i])
                    m = i - 1
            offspring = offsprings[m]

            return offspring

        for _ in range(1):
            offspring = move_cycles(offspring)
        return offspring

    candidate_scores = []
    candidate_partitions = []
    pre_clustering = False
    for i in range(population_size):
        candidate_scores.append(None)
        cur_nodes = nodes.copy()
        random.seed = i
        random.shuffle(cur_nodes)
        if pre_clustering:
            clusters = []
            for _ in range(way):
                clusters.append([cur_nodes.pop(0)])
                cost_array = []
                for cur_node in cur_nodes:
                    cost_array.append(peer_delay[clusters[-1][0], cur_node]/1e3 + (
                        send_activation_size + send_gradient_size) * 8 / peer_bandwidth[clusters[-1][0], cur_node])
                while len(clusters[-1]) < partition_size:
                    min_cost_idx = np.argmin(cost_array)
                    clusters[-1].append(cur_nodes.pop(min_cost_idx))
                    cost_array.pop(min_cost_idx)
            cur_nodes = list(itertools.chain.from_iterable(clusters))
        candidate_partitions.append(cur_nodes)

    for i in range(trails):
        np.random.seed = i
        parent1_idx, parent2_idx = np.random.randint(population_size, size=2)
        parent1_str, parent2_str = normalization(
            candidate_partitions[parent1_idx], candidate_partitions[parent2_idx])
        offspring_str = five_point_crossover(parent1_str, parent2_str)
        offspring_str = cyclic_partitioning(offspring_str)

        if candidate_scores[parent1_idx] == None:
            parent1 = [[] for _ in range(way)]
            for v_idx, partition_idx in enumerate(parent1_str):
                parent1[partition_idx].append(v_idx)
            parent1_data_parallel_cost = compute_data_parallel_cost(
                candidate_partition=parent1)
            parent1_pipeline_parallel_cost, parent1_parallel_path = compute_pipeline_parallel_cost(
                parent1)
            candidate_scores[parent1_idx] = parent1_data_parallel_cost + \
                parent1_pipeline_parallel_cost

        if candidate_scores[parent2_idx] == None:
            parent2 = [[] for i in range(way)]
            for v_idx, partition_idx in enumerate(parent2_str):
                parent2[partition_idx].append(v_idx)
            parent2_data_parallel_cost = compute_data_parallel_cost(
                candidate_partition=parent2)
            parent2_pipeline_parallel_cost, parent2_parallel_path = compute_pipeline_parallel_cost(
                parent2)
            candidate_scores[parent2_idx] = parent2_data_parallel_cost + \
                parent2_pipeline_parallel_cost

        offspring = [[] for _ in range(way)]
        for v_idx, partition_idx in enumerate(offspring_str):
            offspring[partition_idx].append(v_idx)
        offspring_data_parallel_cost = compute_data_parallel_cost(
            candidate_partition=offspring)
        offspring_pipeline_parallel_cost, offspring_parallel_path = compute_pipeline_parallel_cost(
            offspring)
        offspring_score = offspring_data_parallel_cost + offspring_pipeline_parallel_cost
        offspring = list(itertools.chain.from_iterable(offspring))

        if offspring_score > candidate_scores[parent1_idx] and offspring_score > candidate_scores[parent2_idx]:
            candidate_partitions.append(offspring)
            candidate_scores.append(offspring_score)
        else:
            replaced_idx = parent1_idx if candidate_scores[
                parent1_idx] > candidate_scores[parent2_idx] else parent2_idx
            replaced_candidate = candidate_partitions[replaced_idx]
            candidate_partitions[replaced_idx] = offspring
            candidate_partitions.append(replaced_candidate)
            replaced_score = candidate_scores[replaced_idx]
            candidate_scores[replaced_idx] = offspring_score
            candidate_scores.append(replaced_score)
    return candidate_partitions, candidate_scores


def all_candidate_partitions(nodes=None):
    candidate_partitions = []
    if len(nodes) == partition_size:
        candidate_partitions.append([tuple(nodes)])
    else:
        for cur_partition in itertools.combinations(nodes, partition_size):
            rest_nodes = nodes.copy()
            for node in cur_partition:
                rest_nodes.remove(node)

            rest_partitions = all_candidate_partitions(rest_nodes)
            for rest_partition in rest_partitions:
                candidate_partitions.append([cur_partition])
                candidate_partitions[-1].extend(rest_partition)
    return candidate_partitions


def compute_data_parallel_cost(candidate_partition=None):
    data_parallel_cost = 0
    for partition in candidate_partition:
        within_partition_cost = float('inf')
        for primary in partition:
            cur_cost = 0
            for secondary in partition:
                if primary != secondary:
                    cur_cost += peer_delay[primary, secondary] / 1e3 + \
                        send_activation_size * 8 / \
                        peer_bandwidth[primary, secondary]
            if cur_cost < within_partition_cost:
                within_partition_cost = cur_cost
        data_parallel_cost += within_partition_cost
    return data_parallel_cost


def compute_pipeline_parallel_cost(candidate_partition=None):
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

        def get_least_cost_route(self):
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

    def bipartite_matching(candidate_partition_0, candidate_partition_1):
        cost_matrix = np.zeros(shape=(partition_size, partition_size))
        for i in range(partition_size):
            for j in range(partition_size):
                cost_matrix[i, j] = peer_delay[candidate_partition_0[i], candidate_partition_1[j]]/1e3 + \
                    send_gradient_size * 8 / \
                    peer_bandwidth[candidate_partition_0[i],
                                   candidate_partition_1[j]]

        descending_order = np.argsort(cost_matrix.flatten())[::-1]
        inf_weight = 1e6
        for idx in descending_order:
            cur_max_weight = cost_matrix[idx //
                                         partition_size][idx % partition_size]
            cost_matrix[idx//partition_size][idx % partition_size] = inf_weight
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            if cost_matrix[row_ind, col_ind].sum() >= inf_weight:
                return cur_max_weight

    cross_partition_cost = np.zeros(shape=(way, way))
    for i in range(way):
        for j in range(i+1, way):
            #bipartite_matches = []
            # for x in itertools.permutations(candidate_partition[i]):
            #    bipartite_matches.append(list(zip(x, candidate_partition[j])))
            #all_transfer_times = []
            # for bipartite_match in bipartite_matches:
            #    cur_transfer_times = []
            #    for pair in bipartite_match:
            #        cur_transfer_times.append(
            #            peer_delay[pair[0], pair[1]]/1e3 + send_gradient_size * 8 / peer_bandwidth[pair[0], pair[1]])
            #    all_transfer_times.append(max(cur_transfer_times))
            #cross_partition_cost[i, j] = min(all_transfer_times)
            #assert(min(all_transfer_times) == bipartite_matching(candidate_partition[i], candidate_partition[j]))
            cross_partition_cost[i, j] = bipartite_matching(
                candidate_partition[i], candidate_partition[j])
    cross_partition_cost = cross_partition_cost + cross_partition_cost.T

    pipeline_parallel_cost = []
    pipeline_parallel_path = []
    for start_node in range(way):
        tsp = open_loop_tsp(cross_partition_cost, start_node)
        cost, path = tsp.get_least_cost_route()
        pipeline_parallel_cost.append(cost)
        pipeline_parallel_path.append(path)
    dp_pipeline_parallel_cost = min(pipeline_parallel_cost)
    dp_pipeline_parallel_path = pipeline_parallel_path[pipeline_parallel_cost.index(
        dp_pipeline_parallel_cost)]

    # pipeline_parallel_cost = float('inf')
    # pipeline_parallel_path = None
    # for path in itertools.permutations(range(way)):
    #    cur_cost = 0
    #    for i in range(way - 1):
    #        cur_cost += cross_partition_cost[path[i], path[i+1]]
    #    if cur_cost < pipeline_parallel_cost:
    #        pipeline_parallel_cost = cur_cost
    #        pipeline_parallel_path = path
    # assert(dp_pipeline_parallel_cost == pipeline_parallel_cost)

    return dp_pipeline_parallel_cost, dp_pipeline_parallel_path


if __name__ == "__main__":
    assert(batch_size % batch_size_per_task == 0)
    assert(layer_size % layer_size_per_task == 0)
    assert(num_devices == batch_size * layer_size /
           (batch_size_per_task * layer_size_per_task))
    way = int(layer_size / layer_size_per_task)
    partition_size = int(batch_size / batch_size_per_task)

    simulate_cases = [config.simulate_0_datacenter, config.simulate_1_datacenter_spot_gpu, config.simulate_2_multi_universities,
                      config.simulate_3_regional_geo_distributed, config.simulate_4_worldwide_geo_distributed]
    import time
    for simulate_case in simulate_cases:
        peer_delay, peer_bandwidth = simulate_case()
        start = time.perf_counter()
        min_total_cost = float('inf')
        candidate_partition = None
        data_parallel_cost = None
        pipeline_parallel_cost = None
        pipeline_parallel_path = None

        # all_cost_records = []
        # for cur_candidate_partition in all_candidate_partitions(list(range(num_devices))):
        #    cur_data_parallel_cost = compute_data_parallel_cost(
        #        candidate_partition=cur_candidate_partition)
        #    cur_pipeline_parallel_cost, cur_pipeline_parallel_path = compute_pipeline_parallel_cost(
        #        cur_candidate_partition)
        #    cur_total_cost = cur_data_parallel_cost + cur_pipeline_parallel_cost
        #    all_cost_records.append(cur_total_cost)
        #    if min_total_cost >= cur_total_cost:
        #        min_total_cost = cur_total_cost
        #        candidate_partition = cur_candidate_partition
        #        pipeline_parallel_path = cur_pipeline_parallel_path
        #        data_parallel_cost = cur_data_parallel_cost
        #        pipeline_parallel_cost = cur_pipeline_parallel_cost

        candidate_partitions, all_cost_records = GCMA(
            nodes=list(range(num_devices)), population_size=50, trails=450)
        candidate_partition_idx = np.argmin(all_cost_records)
        candidate_partition = [candidate_partitions[candidate_partition_idx][i: i + partition_size]
                               for i in range(0, num_devices, partition_size)]
        data_parallel_cost = compute_data_parallel_cost(
            candidate_partition=candidate_partition)
        pipeline_parallel_cost, pipeline_parallel_path = compute_pipeline_parallel_cost(
            candidate_partition)
        min_total_cost = data_parallel_cost + pipeline_parallel_cost

        end = time.perf_counter()
        print("run time(" + str(len(all_cost_records)) +
              " candidates): " + str(end - start) + " seconds")
        print("candidate partition: " + str(candidate_partition))
        print("pipeline parallel path: " + str(pipeline_parallel_path))
        print("total cost: " + str(data_parallel_cost + pipeline_parallel_cost))
        print("data parallel cost: " + str(data_parallel_cost))
        print("pipeline parallel cost: " + str(pipeline_parallel_cost))
        if len(config.regions):
            for pipeline_idx, partition_idx in enumerate(pipeline_parallel_path):
                print("pipeline " + str(pipeline_idx) +
                      ", partition " + str(partition_idx) + ": ", end="")
                for region_id in candidate_partition[partition_idx]:
                    print(config.regions[region_id], end=", ")
                print()
