import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd

import config
from performance_pointplot import plot_performance

simulate_cases = [
    config.simulate_0_datacenter,
    config.simulate_1_datacenter_spot_gpu,
    config.simulate_2_multi_universities,
    config.simulate_3_regional_geo_distributed,
    config.simulate_4_worldwide_geo_distributed
]
bounds = list(np.arange(0, 25, 0.5))
bounds.extend(list(np.arange(25, 301, 5)))
delay_cmap = plt.get_cmap('RdYlGn_r', len(bounds))
delay_norm = mcolors.BoundaryNorm(bounds, len(bounds))

bounds = list(np.arange(0, 1.5, 0.05))
bounds.extend(list(np.arange(2, 10, 1)))
bounds.extend(list(np.arange(10, 110, 10)))
bandwidth_cmap = plt.get_cmap('RdYlGn', len(bounds))
bandwidth_norm = mcolors.BoundaryNorm(bounds, len(bounds))


def plot_colorbar(subfig=None):
    cax = subfig.add_axes([0.11, 0.056, 0.38, 0.01])
    cbar = mpl.colorbar.ColorbarBase(ax=cax, norm=delay_norm, cmap=delay_cmap,
                                     orientation='horizontal', ticklocation='top', ticks=[0, 10, 20, 100, 200, 300])
    cbar.set_ticklabels([0, 10, 20, 100, 200, 300])
    cbar.outline.set_visible(False)

    cax = subfig.add_axes([0.54, 0.056, 0.38, 0.01])
    cbar = mpl.colorbar.ColorbarBase(ax=cax, norm=bandwidth_norm, cmap=bandwidth_cmap,
                                     orientation='horizontal', ticklocation='top', ticks=[0.0, 0.5, 1.0, 2.0, 10, 100])
    cbar.set_ticklabels([0, 0.5, 1, 2, 10, 100])
    cbar.outline.set_visible(False)


def plot_heatmap(subfig=None):
    axes = subfig.subplots(nrows=5, ncols=2, sharex=True, sharey=True)
    for i, simulate_case in enumerate(simulate_cases):
        peer_delay, peer_bandwidth, regions = simulate_case()
        ax = sns.heatmap(ax=axes[i, 0], data=pd.DataFrame(peer_delay, index=list(range(config.nodes)), columns=list(range(
            config.nodes))), vmin=0, vmax=300, cmap=delay_cmap, norm=delay_norm, cbar=False,  xticklabels=8, yticklabels=8, linewidths=0.001, square=True)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        ax = sns.heatmap(ax=axes[i, 1], data=pd.DataFrame(peer_bandwidth, index=list(range(config.nodes)), columns=list(range(config.nodes))), vmin=0, vmax=100, cmap=bandwidth_cmap,
                         norm=bandwidth_norm, cbar=False, xticklabels=8, yticklabels=8, linewidths=0.001, square=True)


fig = plt.figure(figsize=(10, 17))
subfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=[1.4, 1])
plt.subplots_adjust(hspace=0.4)
plot_colorbar(subfig=subfigs[0])
plot_heatmap(subfig=subfigs[0])
plot_performance(subfig=subfigs[1])
plt.plot([0, 1], [0.751, 0.751], color='black', linestyle='--', linewidth=2,
         transform=plt.gcf().transFigure, clip_on=False)
plt.plot([0, 1], [0.589, 0.589], color='black', linestyle='--', linewidth=2,
         transform=plt.gcf().transFigure, clip_on=False)
plt.plot([0, 1], [0.427, 0.427], color='black', linestyle='--', linewidth=2,
         transform=plt.gcf().transFigure, clip_on=False)
plt.plot([0, 1], [0.265, 0.265], color='black', linestyle='--', linewidth=2,
         transform=plt.gcf().transFigure, clip_on=False)
plt.savefig("combination_plot.eps", dpi=1000)
