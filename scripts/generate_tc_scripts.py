import numpy as np
import argparse

# No space for IPs!!!!
private_ip = [
"172.31.29.65",
"172.31.28.64",
"172.31.16.254",
"172.31.17.189",
"172.31.18.124",
"172.31.30.59",
"172.31.26.186",
"172.31.23.249",
"172.31.21.78",
"172.31.19.142",
"172.31.27.205",
"172.31.29.203",
"172.31.16.137",
"172.31.24.8",
"172.31.28.197",
"172.31.30.4",
"172.31.16.25",
"172.31.22.150",
"172.31.28.211",
"172.31.16.146",
"172.31.25.146",
"172.31.24.18",
"172.31.29.208",
"172.31.20.207",
"172.31.29.31",
"172.31.25.118",
"172.31.24.245",
"172.31.25.181",
"172.31.22.240",
"172.31.30.39",
"172.31.19.38",
"172.31.22.100",
]

def simulate_8_clusters(nodes=32, bw=1):
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
    with open("./scripts/tc_scripts/heterogeneous_setup.sh", 'w') as script:
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
            # limit_pkts = current_delay * 22500 * current_bandwidth
            limit_pkts = int(current_bandwidth * 1500 * 10 * 1.5)
            script.write("sudo tc qdisc add dev ens3 parent 1:{} handle {}: netem rate {}Gbit limit {}\n"
                         .format(handle_index, handle_index*10, current_bandwidth, limit_pkts))
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
    parser.add_argument('--nodes', type=int, default=32, metavar='R',
                        help='Total number of nodes')
    args = parser.parse_args()
    generate_tc_scripts(args)


if __name__ == '__main__':
    main()