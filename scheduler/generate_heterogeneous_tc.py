import argparse
from generate_sim_com_matrices import *

# No space for IPs!!!!
private_ip = [
    "172.31.22.106",
    "172.31.27.231",
    "172.31.31.109",
    "172.31.21.108",
    "172.31.28.117",
    "172.31.29.242",
    "172.31.16.250",
    "172.31.18.117",
    "172.31.28.224",
    "172.31.16.226",
    "172.31.30.225",
    "172.31.25.230",
    "172.31.17.229",
    "172.31.29.27",
    "172.31.29.155",
    "172.31.20.29",
    "172.31.31.145",
    "172.31.27.143",
    "172.31.17.20",
    "172.31.23.18",
    "172.31.22.21",
    "172.31.29.149",
    "172.31.29.25",
    "172.31.20.24",
    "172.31.27.59",
    "172.31.28.59",
    "172.31.31.128",
    "172.31.16.63",
    "172.31.24.4",
    "172.31.16.131",
    "172.31.24.15",
    "172.31.29.4",
    "172.31.26.47",
    "172.31.27.174",
    "172.31.31.48",
    "172.31.23.176",
    "172.31.19.183",
    "172.31.30.183",
    "172.31.30.185",
    "172.31.30.55",
    "172.31.16.33",
    "172.31.19.33",
    "172.31.16.163",
    "172.31.29.162",
    "172.31.20.166",
    "172.31.19.166",
    "172.31.17.170",
    "172.31.17.40",
    "172.31.27.206",
    "172.31.31.206",
    "172.31.23.86",
    "172.31.22.211",
    "172.31.28.220",
    "172.31.24.88",
    "172.31.28.92",
    "172.31.27.92",
    "172.31.29.127",
    "172.31.27.125",
    "172.31.26.199",
    "172.31.26.192",
    "172.31.21.201",
    "172.31.27.72",
    "172.31.25.205",
    "172.31.26.201"
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
