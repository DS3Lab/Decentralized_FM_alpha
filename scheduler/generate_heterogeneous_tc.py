import argparse
from generate_sim_com_matrices import *

# No space for IPs!!!!
private_ip = [
    "172.31.36.100",
    "172.31.39.103",
    "172.31.35.99",
    "172.31.41.104",
    "172.31.43.105",
    "172.31.45.103",
    "172.31.32.232",
    "172.31.36.116",
    "172.31.38.118",
    "172.31.38.235",
    "172.31.43.112",
    "172.31.47.122",
    "172.31.47.123",
    "172.31.45.119",
    "172.31.39.248",
    "172.31.32.67",
    "172.31.37.68",
    "172.31.47.124",
    "172.31.35.255",
    "172.31.37.78",
    "172.31.44.80",
    "172.31.39.203",
    "172.31.34.204",
    "172.31.47.214",
    "172.31.43.87",
    "172.31.47.85",
    "172.31.35.214",
    "172.31.37.90",
    "172.31.32.220",
    "172.31.39.215",
    "172.31.38.88",
    "172.31.37.32",
    "172.31.38.164",
    "172.31.33.220",
    "172.31.45.222",
    "172.31.38.40",
    "172.31.46.40",
    "172.31.35.168",
    "172.31.43.40",
    "172.31.41.173",
    "172.31.38.47",
    "172.31.33.171",
    "172.31.45.43",
    "172.31.42.57",
    "172.31.33.60",
    "172.31.41.50",
    "172.31.37.55",
    "172.31.36.191",
    "172.31.47.130",
    "172.31.47.188",
    "172.31.42.189",
    "172.31.47.139",
    "172.31.40.11",
    "172.31.43.3",
    "172.31.44.6",
    "172.31.32.151",
    "172.31.38.23",
    "172.31.46.139",
    "172.31.32.141",
    "172.31.44.154",
    "172.31.33.157",
    "172.31.37.153",
    "172.31.41.154",
    "172.31.46.30"
]


def get_delay_bandwidth(args):
    if args.case == 0:
        return simulate_0_datacenter(args.nodes)
    elif args.case == 1:
        return simulate_1_datacenter_spot_gpu(args.nodes)
    elif args.case == 2:
        return simulate_2_multi_universities(args.nodes)
    elif args.case == 3:
        return simulate_3_regional_geo_distributed(args.nodes)
    elif args.case == 4:
        return simulate_4_worldwide_geo_distributed(args.nodes)
    elif args.case == 5:
        return simulate_5_homogeneous_tc(args.nodes)
    elif args.case == 6:
        return simulate_6_debug(args.nodes)
    else:
        assert False


def generate_tc_scripts(args):
    assert args.nodes == len(private_ip)
    delay, bandwidth, _ = get_delay_bandwidth(args)
    with open("../scripts/tc_scripts/heterogeneous_setup_case"+str(args.case)+".sh", 'w') as script:
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
    parser.add_argument('--case', type=int, default=6, metavar='R',
                        help='size of the tensor to be sent.')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for this IP')
    parser.add_argument('--nodes', type=int, default=8, metavar='R',
                        help='Total number of nodes')
    args = parser.parse_args()
    generate_tc_scripts(args)


if __name__ == '__main__':
    main()
