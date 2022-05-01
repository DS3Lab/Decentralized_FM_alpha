import argparse
from generate_sim_com_matrices import *


private_ip = [
    "172.31.20.114",
    "172.31.25.206",
    "172.31.29.201",
    "172.31.21.45",
    "172.31.21.7",
    "172.31.29.9",
    "172.31.30.51",
    "172.31.27.115"
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
        script.write("sudo tc qdisc add dev ens3 root handle 1: prio\n")
        # setup pair-wise delay and bandwidth qdisc
        for i in range(len(private_ip)):
            if i != args.rank:
                limit_pkts = bandwidth[args.rank][i] * 22500 * delay[args.rank][i]
                script.write("sudo tc qdisc add dev ens3 parent 1:1 classid: 1:{}: netem delay {}ms rate {}Gbit limit {}\n"
                             .format(i+1, delay[args.rank][i], bandwidth[args.rank][i], limit_pkts))
        for i in range(len(private_ip)):
            if i != args.rank:
                script.write("sudo tc filter add dev ens3 parent 1:0 protocol ip prio 1 u32 match ip src {}/32 flowid 1:{}\n"
                             .format(private_ip[i], i+1))


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
