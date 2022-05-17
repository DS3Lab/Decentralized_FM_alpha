import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, axes = plt.subplots(nrows=1, ncols=5, sharey=True,
                         figsize=(15, 3), tight_layout=True)

for case_idx in range(5):
    with open('random_scheduler_' + str(case_idx) + '.npy', 'rb') as f:
        random_scheduler_results = np.load(f)

    with open('hybrid_scheduler_' + str(case_idx) + '.npy', 'rb') as f:
        hybrid_scheduler_results = np.load(f)

    with open('our_scheduler_' + str(case_idx) + '.npy', 'rb') as f:
        our_scheduler_results = np.load(f)

    # assert(len(random_scheduler_results) == len(hybrid_scheduler_results))
    # assert(len(random_scheduler_results) == len(our_scheduler_results))

    data = []
    for i in range(5000):
        data.append([i, random_scheduler_results[i], 'Random'])
        data.append([i, hybrid_scheduler_results[i], 'Hybrid'])
        data.append([i, our_scheduler_results[i], 'Our'])

    df = pd.DataFrame(data, columns=['trial', 'cost', 'scheduler'])
    ax = sns.lineplot(ax=axes[case_idx], hue_order=['Our', 'Hybrid', 'Random'], linewidth=3,
                      palette=['tab:green', 'tab:orange', 'tab:blue'],
                      data=df, x="trial", y="cost", hue='scheduler')
    ax.lines[2].set_linestyle("--")

    ax.get_legend().set_title(None)
    handles, labels = ax.get_legend_handles_labels()
    handles[2].set_linestyle('--')
    handles = [handles[2], handles[1], handles[0]]
    labels = [labels[2], labels[1], labels[0]]
    if case_idx == 4:
        ax.legend(handles, labels, loc='lower right')
    else:
        ax.legend(handles, labels)

    ax.set_xlabel('Trials')
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    if case_idx == 0:
        ax.set_ylabel('Cost (ms)')

plt.savefig("convergence.eps", dpi=1000)
