import numpy as np
import argparse

# No space for IPs!!!!
private_ip = [
    "172.31.47.149",
    "172.31.36.151",
    "172.31.33.20",
    "172.31.45.149",
    "172.31.44.153",
    "172.31.39.155",
    "172.31.43.24",
    "172.31.37.24",
    "172.31.36.30",
    "172.31.37.156",
    "172.31.33.158",
    "172.31.39.224",
    "172.31.34.98",
    "172.31.41.101",
    "172.31.39.98",
    "172.31.41.226",
    "172.31.36.109",
    "172.31.41.109",
    "172.31.47.103",
    "172.31.33.104",
    "172.31.33.113",
    "172.31.37.241",
    "172.31.46.111",
    "172.31.39.112",
    "172.31.40.194",
    "172.31.43.195",
    "172.31.36.116",
    "172.31.43.64",
    "172.31.32.201",
    "172.31.40.74",
    "172.31.42.198",
    "172.31.33.198",
    "172.31.39.219",
    "172.31.33.94",
    "172.31.47.82",
    "172.31.38.216",
    "172.31.37.46",
    "172.31.33.47",
    "172.31.46.165",
    "172.31.45.173",
    "172.31.37.52",
    "172.31.36.181",
    "172.31.46.49",
    "172.31.43.49",
    "172.31.34.185",
    "172.31.45.186",
    "172.31.40.55",
    "172.31.34.55",
    "172.31.35.0",
    "172.31.41.128",
    "172.31.39.59",
    "172.31.45.190",
    "172.31.34.132",
    "172.31.45.6",
    "172.31.35.129",
    "172.31.42.129",
    "172.31.45.9",
    "172.31.43.139",
    "172.31.42.136",
    "172.31.38.136",
    "172.31.37.143",
    "172.31.46.144",
    "172.31.34.13",
    "172.31.46.14"
]

def simulate_8_clusters(nodes=64, bw=1):
    delay = np.zeros((nodes, nodes))
    bandwidth = np.ones((nodes, nodes)) * 10
    split = nodes//2
    for i in range(nodes):
        for j in range(nodes):
            if i%8 != j%8:
                bandwidth[i, j] = bw
    print('delay:', delay)
    print('bandwidth:', bandwidth)
    return delay, bandwidth, []

def generate_tc_scripts(args):
    assert args.nodes == len(private_ip)
    delay, bandwidth, _ = simulate_8_clusters(args.nodes, args.bandwidth)
    with open("./tc_scripts/heterogeneous_setup.sh", 'w') as script:
        tc_setting_dict = {}
        handle_i = 1
        for i in range(len(private_ip)):
            if i != args.rank:
                current_key = (delay[args.rank][i], bandwidth[args.rank][i])
                if current_key not in tc_setting_dict:
                    tc_setting_dict[current_key] = handle_i
                    handle_i += 1
        assert len(tc_setting_dict) <= 16
        # setup delay and bandwidth subclass qdisc
        script.write("sudo tc qdisc add dev ens3 root handle 1: prio bands {}\n"
                     .format(max(3, len(tc_setting_dict))))
        for key in tc_setting_dict.keys():
            current_delay, current_bandwidth = key
            handle_index = tc_setting_dict[key]
            limit_pkts = current_delay * 22500 * current_bandwidth
            script.write("sudo tc qdisc add dev ens3 parent 1:{} handle {}: netem delay {}ms rate {}Gbit limit {}\n"
                         .format(handle_index, handle_index*10, current_delay, current_bandwidth, limit_pkts))
        # setup filter
        for i in range(len(private_ip)):
            if i != args.rank:
                current_key = (delay[args.rank][i], bandwidth[args.rank][i])
                script.write("sudo tc filter add dev ens3 parent 1:0 protocol ip prio 1 u32 match ip dst {}/32 flowid 1:{}\n"
                             .format(private_ip[i], tc_setting_dict[current_key]))


def main():
    parser = argparse.ArgumentParser(description='Test PyTorch Distributed')
    parser.add_argument('--bandwidth', type=str, default=1, metavar='R',
                        help='bandwith between clusters')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for this IP')
    parser.add_argument('--nodes', type=int, default=64, metavar='R',
                        help='Total number of nodes')
    args = parser.parse_args()
    generate_tc_scripts(args)


if __name__ == '__main__':
    main()