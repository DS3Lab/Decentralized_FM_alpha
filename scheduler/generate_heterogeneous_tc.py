import argparse
from generate_sim_com_matrices import *

# No space for IPs!!!!
private_ip = [
    "172.31.37.23",
    "172.31.44.152",
    "172.31.45.152",
    "172.31.47.154",
    "172.31.32.157",
    "172.31.35.31",
    "172.31.36.137",
    "172.31.47.138",
    "172.31.36.139",
    "172.31.47.14",
    "172.31.47.142",
    "172.31.33.146",
    "172.31.42.21",
    "172.31.33.151",
    "172.31.41.48",
    "172.31.44.50",
    "172.31.33.55",
    "172.31.36.186",
    "172.31.40.4",
    "172.31.43.132",
    "172.31.33.6",
    "172.31.34.7",
    "172.31.35.35",
    "172.31.40.40",
    "172.31.38.41",
    "172.31.43.170",
    "172.31.39.44",
    "172.31.35.46",
    "172.31.41.175",
    "172.31.46.47",
    "172.31.46.216",
    "172.31.32.220",
    "172.31.44.222",
    "172.31.47.222",
    "172.31.36.94",
    "172.31.38.160",
    "172.31.46.35",
    "172.31.37.35",
    "172.31.33.78",
    "172.31.32.207",
    "172.31.32.80",
    "172.31.37.209",
    "172.31.45.209",
    "172.31.36.83",
    "172.31.32.212",
    "172.31.39.213",
    "172.31.41.250",
    "172.31.34.124",
    "172.31.46.252",
    "172.31.44.127",
    "172.31.35.196",
    "172.31.41.202",
    "172.31.46.74",
    "172.31.36.77",
    "172.31.35.229",
    "172.31.39.105",
    "172.31.32.237",
    "172.31.41.244",
    "172.31.35.118",
    "172.31.37.119",
    "172.31.34.120",
    "172.31.36.248",
    "172.31.40.225",
    "172.31.47.97"
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
