import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Map size
fig = plt.figure(figsize=(7, 3))
plt.title(None)

# Declare map projection, size and resolution
map = Basemap(projection='merc',
              llcrnrlat=-20,
              urcrnrlat=80,
              llcrnrlon=-180,
              urcrnrlon=180,
              lat_ts=20,
              resolution='l')
map.drawlsmask(land_color='floralwhite',
               ocean_color='powderblue',
               resolution='l')
map.drawcountries(linewidth=0.1, color='thistle')

pipelines_names = [['Oregon', 'Ohio', 'Seoul', 'Tokyo'],
                   ['Frankfurt', 'London', 'Ireland', 'Virginia']]
pipelines_longitudes = [[-120.55, -82.90, 127.02, 139.69],
                        [8.68, -0.11, -6.26, -78.65]]
pipelines_latitudes = [[43.80, 40.41, 37.53, 35.68],
                       [50.11, 51.50, 53.35, 37.43]]
'''
stages_colors = ['coral', 'green', 'royalblue', 'deeppink']

for stage_idx in range(4):
    x, y = map(pipelines_longitudes[0][stage_idx],
               pipelines_latitudes[0][stage_idx])
    plt.text(x, y, pipelines_names[0][stage_idx], fontsize=5,
             fontweight='bold', ha='left', va='bottom', color=stages_colors[stage_idx])

    x, y = map(pipelines_longitudes[1][stage_idx],
               pipelines_latitudes[1][stage_idx])
    plt.text(x, y, pipelines_names[1][stage_idx], fontsize=5,
             fontweight='bold', ha='left', va='bottom', color=stages_colors[stage_idx])
'''

plt.savefig("world_map.eps", dpi=1000)
