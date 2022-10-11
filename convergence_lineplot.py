import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 4))
plt.subplots_adjust(wspace=0.3, hspace=0.1)

for case_idx in range(5):
    iterations = 15000 if case_idx == 1 else 1500
    random_scheduler_results = np.zeros(shape=(iterations))
    hybrid_scheduler_results = np.zeros(shape=(iterations))
    our_scheduler_results = np.zeros(shape=(iterations))
    for repetition in range(3):
        with open('data/random_scheduler_' + str(case_idx) + '_' + str(repetition) + '.npy', 'rb') as f:
            random_scheduler_results += np.load(f)[:iterations] / 3

        with open('data/hybrid_scheduler_' + str(case_idx) + '_' + str(repetition) + '.npy', 'rb') as f:
            hybrid_scheduler_results += np.load(f)[:iterations] / 3

        with open('data/our_scheduler_' + str(case_idx) + '_' + str(repetition) + '.npy', 'rb') as f:
            our_scheduler_results += np.load(f)[:iterations] / 3

    assert(len(random_scheduler_results) == len(hybrid_scheduler_results))
    assert(len(random_scheduler_results) == len(our_scheduler_results))

    data = []
    for i in range(iterations):
        data.append([i, random_scheduler_results[i], 'Random'])
        data.append([i, hybrid_scheduler_results[i], 'Hybrid'])
        data.append([i, our_scheduler_results[i], 'Ours'])

    df = pd.DataFrame(data, columns=['trial', 'cost', 'scheduler'])
    ax = sns.lineplot(ax=axes[0, case_idx], hue_order=['Ours', 'Hybrid', 'Random'], linewidth=2,
                      palette=['tab:green', 'tab:orange', 'tab:blue'],
                      data=df, x="trial", y="cost", hue='scheduler')
    ax.lines[2].set_linestyle("--")

    ax.set_xlabel(None)
    ax.set_ylabel(None)
    if case_idx == 0:
        ax.set(ylim=(0, 25))
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_yticklabels([0, 5, 10, 15, 20, 25], fontsize=13)
    elif case_idx == 1:
        ax.set(ylim=(0, 80))
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_yticklabels([0, 20, 40, 60, 80], fontsize=13)
    elif case_idx == 2:
        ax.set(ylim=(0, 80))
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_yticklabels([0, 20, 40, 60, 80], fontsize=13)
    elif case_idx == 3:
        ax.set(ylim=(0, 80))
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_yticklabels([0, 20, 40, 60, 80], fontsize=13)
    elif case_idx == 4:
        ax.set(ylim=(0, 200))
        ax.set_yticks([0, 50, 100, 150, 200])
        ax.set_yticklabels([0, 50, 100, 150, 200], fontsize=13)

    if case_idx == 1:
        ax.set_xticks([0, 5000, 10000, 15000])
        ax.set_xticklabels([0, 5000, 10000, 15000], fontsize=13)
    else:
        ax.set_xticks([0, 500, 1000, 1500])
        ax.set_xticklabels([0, 500, 1000, 1500], fontsize=13)

    if case_idx == 3:
        ax.get_legend().set_title(None)
        handles, labels = ax.get_legend_handles_labels()
        handles[2].set_linestyle('--')
        handles = [handles[2], handles[1], handles[0]]
        labels = [labels[2], labels[1], labels[0]]
        ax.legend(handles, labels, ncol=3, handletextpad=0.3,
                  loc='lower center', bbox_to_anchor=(-0.85, -0.9), fontsize=13)
    else:
        ax.get_legend().remove()
    axes[1, case_idx].remove()

plt.savefig("convergence.pdf", dpi=1000)
