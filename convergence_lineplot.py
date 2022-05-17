import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


fig, axes = plt.subplots(nrows=1, ncols=5, sharey=True, figsize=(15, 4))

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
        data.append([i, random_scheduler_results[i], 'random'])
        data.append([i, hybrid_scheduler_results[i], 'hybrid'])
        data.append([i, our_scheduler_results[i], 'our'])

    df = pd.DataFrame(data, columns=['trial', 'cost', 'scheduler'])
    ax = sns.lineplot(ax=axes[case_idx], data=df,
                      x="trial", y="cost", hue='scheduler')
    ax.get_legend().set_title(None)
    ax.set_xlabel('Trial')
    if case_idx == 0:
        ax.set_ylabel('Cost (ms)')
plt.savefig("convergence_lineplot.pdf", dpi=1000)
