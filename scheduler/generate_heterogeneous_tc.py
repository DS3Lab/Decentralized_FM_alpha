import argparse
from generate_sim_com_matrices import *

# No space for IPs!!!!
private_ip = [
    "172.31.14.16",
    "172.31.13.145",
    "172.31.13.148",
    "172.31.14.20",
    "172.31.14.137",
    "172.31.9.139",
    "172.31.14.139",
    "172.31.7.141",
    "172.31.11.27",
    "172.31.3.155",
    "172.31.7.29",
    "172.31.3.22",
    "172.31.2.150",
    "172.31.13.150",
    "172.31.15.26",
    "172.31.12.79",
    "172.31.2.80",
    "172.31.0.214",
    "172.31.15.88",
    "172.31.10.64",
    "172.31.9.71",
    "172.31.6.72",
    "172.31.13.203",
    "172.31.4.163",
    "172.31.7.164",
    "172.31.3.39",
    "172.31.8.41",
    "172.31.13.216",
    "172.31.6.221",
    "172.31.5.94",
    "172.31.5.32",
    "172.31.2.52",
    "172.31.3.56",
    "172.31.5.58",
    "172.31.7.188",
    "172.31.3.169",
    "172.31.8.178",
    "172.31.14.51",
    "172.31.6.180",
    "172.31.4.3",
    "172.31.8.133",
    "172.31.1.5",
    "172.31.9.9",
    "172.31.6.61",
    "172.31.4.130",
    "172.31.13.131",
    "172.31.14.3",
    "172.31.6.96",
    "172.31.11.228",
    "172.31.7.229",
    "172.31.0.102",
    "172.31.11.232",
    "172.31.8.98",
    "172.31.15.227",
    "172.31.14.99",
    "172.31.1.228",
    "172.31.0.246",
    "172.31.14.122",
    "172.31.13.250",
    "172.31.15.123",
    "172.31.6.237",
    "172.31.5.110",
    "172.31.5.239",
    "172.31.10.115"
]


def get_delay_bandwidth(args):
    if args.case == '1':
        return simulate_1_datacenter(args.nodes)
    elif args.case == '2':
        return simulate_2_datacenter_spot_gpu(args.nodes)
    elif args.case == '3':
        return simulate_3_multi_universities(args.nodes)
    elif args.case == '4':
        return simulate_4_regional_geo_distributed(args.nodes)
    elif args.case == '5':
        return simulate_5_worldwide_geo_distributed(args.nodes)
    elif args.case == '5_2':
        return simulate_5_2_worldwide_geo_distributed(args.nodes)
    elif args.case == '6_1':
        return simulate_6_1_debug_homogeneous_tc(args.nodes)
    elif args.case == '6_2':
        return simulate_6_2_debug_pipeline(args.nodes)
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
    parser.add_argument('--case', type=str, default='5', metavar='R',
                        help='which case to generate.')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for this IP')
    parser.add_argument('--nodes', type=int, default=64, metavar='R',
                        help='Total number of nodes')
    args = parser.parse_args()
    generate_tc_scripts(args)


if __name__ == '__main__':
    main()
