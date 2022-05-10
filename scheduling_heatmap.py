import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True,
                         sharey=True, figsize=(5, 15), gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.45]})
sns.set_theme()

bounds = list(np.arange(0, 25, 0.5))
bounds.extend(list(np.arange(25, 301, 5)))
delay_cmap = plt.get_cmap('coolwarm', len(bounds))
delay_norm = mcolors.BoundaryNorm(bounds, len(bounds))

bounds = list(np.arange(0, 1.5, 0.05))
bounds.extend(list(np.arange(2, 10, 1)))
bounds.extend(list(np.arange(10, 110, 10)))
bandwidth_cmap = plt.get_cmap('coolwarm', len(bounds))
bandwidth_norm = mcolors.BoundaryNorm(bounds, len(bounds))

for i, simulate_case in enumerate(simulate_cases):
    peer_delay, peer_bandwidth, regions = simulate_case()
    ax = sns.heatmap(ax=axes[i, 0], data=pd.DataFrame(peer_delay, index=list(range(config.nodes)), columns=list(range(config.nodes))), vmin=0, vmax=300, cmap=delay_cmap,
                     norm=delay_norm, cbar=True if i == 4 else False, cbar_kws={'location': 'bottom', 'ticks': [0, 20, 50, 100, 200, 300]}, xticklabels=8, yticklabels=8, linewidths=0.001, square=True)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    ax = sns.heatmap(ax=axes[i, 1], data=pd.DataFrame(peer_bandwidth, index=list(range(config.nodes)), columns=list(range(config.nodes))), vmin=0, vmax=100, cmap=bandwidth_cmap,
                     norm=bandwidth_norm, cbar=True if i == 4 else False, cbar_kws={'location': 'bottom', 'ticks': [0.0, 0.5, 1.0, 2.0, 10, 100]}, xticklabels=8, yticklabels=8, linewidths=0.001, square=True)


plt.savefig("scheduling_heatmap.eps", dpi=1000)
