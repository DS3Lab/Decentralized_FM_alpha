import numpy as np


'''
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

fig = plt.figure(figsize=(7, 3))
plt.title(None)

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

plt.savefig("world_map.eps", dpi=1000)
'''


import folium

world_map = folium.Map()

pipelines_names = [['Oregon', 'Ohio', 'Seoul', 'Tokyo'],
                   ['Frankfurt', 'London', 'Ireland', 'Virginia']]
pipelines_longitudes = [[-120.55, -82.90, 127.02, 139.69],
                        [8.68, -0.11, -6.26, -78.65]]
pipelines_latitudes = [[43.80, 40.41, 37.53, 35.68],
                       [50.11, 51.50, 53.35, 37.43]]
stages_colors = ['orange', 'darkgreen', 'darkblue', 'darkpurple']

for stage_idx in range(4):
    folium.Marker(
        location=[pipelines_latitudes[0][stage_idx],
                  pipelines_longitudes[0][stage_idx]],
        popup=pipelines_names[0][stage_idx],
        icon=folium.Icon(prefix='fa', icon='circle', tooltip=pipelines_names[0][stage_idx],
                         color=stages_colors[stage_idx], icon_color='white'),
    ).add_to(world_map)
    folium.Marker(
        location=[pipelines_latitudes[1][stage_idx],
                  pipelines_longitudes[1][stage_idx]],
        popup=pipelines_names[1][stage_idx],
        icon=folium.Icon(prefix='fa', icon='circle',
                         color=stages_colors[stage_idx], icon_color='white'),
    ).add_to(world_map)

world_map.save("world_map.html")
