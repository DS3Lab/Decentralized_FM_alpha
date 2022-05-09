from turtle import title
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import config


simulate_cases = [
    config.simulate_0_datacenter,
    config.simulate_1_datacenter_spot_gpu,
    config.simulate_2_multi_universities,
    config.simulate_3_regional_geo_distributed,
    config.simulate_4_worldwide_geo_distributed
]

fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True,
                         sharey=True, figsize=(6, 12))
sns.set_theme()

for i, simulate_case in enumerate(simulate_cases):
    allocation = np.zeros(shape=(64, 3)).astype('int')
    for node_id in range(64):
        allocation[node_id, 0] = node_id
        if i != 1:
            allocation[node_id, 1] = node_id // 8
            allocation[node_id, 2] = node_id % 8
        else:
            if node_id < 32:
                allocation[node_id, 1] = node_id // 4
                allocation[node_id, 2] = node_id % 4
            else:
                allocation[node_id, 1] = (node_id - 32) // 4
                allocation[node_id, 2] = (node_id - 32) % 4 + 3

    df = pd.DataFrame(data=allocation, columns=[
                      'node_id', 'pipeline_id', 'stage_id'])

    ax = sns.scatterplot(ax=axes[i], data=df, x="node_id", y="pipeline_id",
                         hue="stage_id", s=10, legend="full")
    ax.set(ylim=(-1, 8))
    if i == 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5), title="stage_id",
                  ncol=4, fancybox=False, shadow=False)
    else:
        ax.get_legend().remove()

plt.savefig("scheduling_pointplot.eps", dpi=300)
