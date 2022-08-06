hostnames=[
    "ip-172-31-1-5",
    "ip-172-31-2-192",
    "ip-172-31-38-170",
    "ip-172-31-34-76"
]

shuffle_ranks = [3, 1, 0, 2]


def shuffle_hostnames_case2():
    assert (len(hostnames) == 40)
    with open("./ds_hostnames_shuffled", 'w') as output:
        for rank in shuffle_ranks:
            if rank < 8:
                output.write(hostnames[rank] + ' slots=4\n')
            else:
                output.write(hostnames[rank] + ' slots=1\n')


def shuffle_hostnames_case345():
    assert(len(hostnames) == 64)
    with open("./ds_hostnames_shuffled", 'w') as output:
        for rank in shuffle_ranks:
            output.write(hostnames[rank] + ' slots=1\n')


shuffle_hostnames_case345()
