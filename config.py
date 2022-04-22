import numpy as np

nodes = 64


def simulate_0_datacenter():
    print("Simulate case 0: on-demand datacenter.")
    delay = np.zeros((nodes, nodes))
    bandwidth = np.zeros((nodes, nodes))
    split = nodes//2
    for i in range(nodes):
        for j in range(nodes):
            if (i < split and j < split) or (i >= split and j >= split):
                bandwidth[i][j] = 100
            else:
                bandwidth[i][j] = 25
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth


def simulate_1_datacenter_spot_gpu():
    print("Simulate case 1: spot datacenter.")
    delay = np.zeros((nodes, nodes))
    bandwidth = np.ones((nodes, nodes)) * 10
    spot_pair = {(1, 2), (3, 5)}  # pair of instances on the same machine.
    for (i, j) in spot_pair:
        bandwidth[i][j] = 100
        bandwidth[j][i] = 100
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth


def simulate_2_multi_universities():
    print("Simulate case 2: multi universities. 0~5 in Ohio, 6~15 in Virginia.")
    delay = np.zeros((nodes, nodes))
    bandwidth = np.ones((nodes, nodes)) * 10
    split = nodes//2 - 1
    for i in range(nodes):
        for j in range(nodes):
            if not ((i < split and j < split) or (i >= split and j >= split)):
                delay[i][j] = 11
                bandwidth[i][j] = 1.12
    print('delay:', delay)
    print('bandwidth:', bandwidth)
    return delay, bandwidth


delay_bandwidth_dict = {
    "Oregon-Virginia": (67, 0.79),
    "Oregon-Ohio": (49, 1.10),
    "Oregon-Tokyo": (96, 0.523),
    "Oregon-Seoul": (124, 0.46),
    "Oregon-Singapore": (163, 0.341),
    "Oregon-Sydney": (139, 0.36),
    "Oregon-London": (136, 0.42),
    "Oregon-Frankfurt": (143, 0.404),
    "Oregon-Ireland": (124, 0.482),
    "Virginia-Ohio": (11, 1.12),
    "Virginia-Tokyo": (143, 0.524),
    "Virginia-Seoul": (172, 0.500),
    "Virginia-Singapore": (230, 0.364),
    "Virginia-Sydney": (197, 0.383),
    "Virginia-London": (76, 1.16),
    "Virginia-Frankfurt": (90, 1.02),
    "Virginia-Ireland": (67, 1.05),
    "Ohio-Tokyo": (130, 0.694),
    "Ohio-Seoul": (159, 0.529),
    "Ohio-Singapore": (197, 0.452),
    "Ohio-Sydney": (185, 0.484),
    "Ohio-London": (86, 1.05),
    "Ohio-Frankfurt": (99, 0.799),
    "Ohio-Ireland": (77, 1.14),
    "Tokyo-Seoul": (34, 1.10),
    "Tokyo-Singapore": (73, 1.01),
    "Tokyo-Sydney": (100, 0.761),
    "Tokyo-London": (210, 0.366),
    "Tokyo-Frankfurt": (223, 0.36),
    "Tokyo-Ireland": (199, 0.465),
    "Seoul-Singapore": (74, 1.14),
    "Seoul-Sydney": (148, 0.58),
    "Seoul-London": (238, 0.342),
    "Seoul-Frankfurt": (235, 0.358),
    "Seoul-Ireland": (228, 335),
    "Singapore-Sydney": (92, 0.816),
    "Singapore-London": (169, 0.500),
    "Singapore-Frankfurt": (155, 0.535),
    "Singapore-Ireland": (179, 0.492),
    "Sydney-London": (262, 0.326),
    "Sydney-Frankfurt": (265, 0.328),
    "Sydney-Ireland": (254, 0.344),
    "London-Frankfurt": (14, 1.14),
    "London-Ireland": (12, 1.09),
    "Frankfurt-Ireland": (24, 1.08)
}

# Assume within region is 2 GB, 5 ms.


def simulate_3_regional_geo_distributed():
    print("Simulate case 3: regional geo distributed: 0~5 in Virgina; 5~10 in Oregon, 11~15 in Ohio")

    def in_virgina(index: int):
        return index <= nodes//4

    def in_oregon(index: int):
        return nodes//2+1 >= index > nodes//4

    def in_ohio(index: int):
        return index > nodes//2+1

    delay = np.ones((nodes, nodes)) * 5
    bandwidth = np.ones((nodes, nodes)) * 2
    for i in range(nodes):
        for j in range(i, nodes):
            if in_virgina(i) and in_oregon(j):
                delay[i][j] = 67
                delay[j][i] = 67
                bandwidth[i][j] = 1.15
                bandwidth[j][i] = 1.15
            elif in_virgina(i) and in_ohio(j):
                delay[i][j] = 11
                delay[j][i] = 11
                bandwidth[i][j] = 1.12
                bandwidth[j][i] = 1.12
            elif in_oregon(i) and in_ohio(j):
                delay[i][j] = 49
                delay[j][i] = 49
                bandwidth[i][j] = 1.10
                bandwidth[j][i] = 1.10
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth


# Assume within region is 2 GB, 5 ms.
def simulate_4_worldwide_geo_distributed():
    print("Simulate case 4: worldwide geo distributed")
    cities = ["Oregon", "Virginia", "Ohio", "Tokyo", "Seoul",
              "Singapore", "Sydney", "London", "Frankfurt", "Ireland"]

    regions = []
    for i in np.random.randint(low=0, high=len(cities), size=nodes):
        regions.append(cities[i])

    # regions = ["Oregon", "Oregon", "Virginia", "Ohio", "Ohio", "Tokyo", "Seoul", "Seoul",
    #           "Singapore", "Sydney", "London", "London", "Frankfurt", "Frankfurt", "Ireland", "Ireland"]
    # regions = ["Oregon", "Virginia", "Tokyo", "Seoul",
    #           "Singapore", "London", "Frankfurt", "Ireland"]
    assert len(regions) == nodes

    def get_delay_bandwidth(region1: str, region2: str):
        if region1 == region2:
            return 5, 2
        else:
            if region1+'-'+region2 in delay_bandwidth_dict:
                return delay_bandwidth_dict[region1+'-'+region2]
            elif region2+'-'+region1 in delay_bandwidth_dict:
                return delay_bandwidth_dict[region2+'-'+region1]
            else:
                print(region1, region2)
                assert False

    delay = np.ones((nodes, nodes)) * 5
    bandwidth = np.ones((nodes, nodes)) * 2

    for i in range(nodes):
        for j in range(i, nodes):
            d_val, b_val = get_delay_bandwidth(regions[i], regions[j])
            delay[i][j] = d_val
            delay[j][i] = d_val
            bandwidth[i][j] = b_val
            bandwidth[j][i] = b_val
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth
