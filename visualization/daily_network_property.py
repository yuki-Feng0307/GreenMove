import os
import pickle
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# replace with your own directory path
directory = 'daily_network_exp_8-4_geometry/'
edge_counts = []
edge_commuter_ratios = []
edge_distance = []
flow = []


# Get all .pkl files in the directory and sort them by filename
pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
pkl_files.sort()

for filename in pkl_files:
    # '2014-02-03.pkl", '2014-02-12.pkl', '2014-02-13.pkl', '2014-02-14.pkl', '2014-02-15.pkl', '2014-02-16.pkl': dates with incomplete data
    if filename == '2014-02-04.pkl':
        edge_counts.append(None)
        edge_commuter_ratios.append(None)
        edge_distance.append(None)
        flow.append(None)
    if filename == '2014-02-17.pkl':
        for _ in range(5):
            edge_counts.append(None)
            edge_commuter_ratios.append(None)
            edge_distance.append(None)
            flow.append(None)

    else:
        file_path = os.path.join(directory, filename)

        with open(file_path, 'rb') as file:
            network = pickle.load(file)

        # edge numbers
        num_edges = len(network.edges)
        edge_counts.append(num_edges)

        # commuter_ratio
        commuter_ratios = [data.get('commuter_ratio', np.nan) for u, v, data in network.edges(data=True)]
        # mean commuter ratio
        if commuter_ratios:
            average_commuter_ratio = np.nanmean(commuter_ratios)
        else:
            average_commuter_ratio = np.nan
        # add to list
        edge_commuter_ratios.append(average_commuter_ratio)

        # distance
        distances = [data.get('distance', np.nan) for u, v, data in network.edges(data=True)]
        # mean distance
        if distances:
            average_distance = np.nanmean(distances)
        else:
            average_distance = np.nan
        # add to list
        edge_distance.append(average_distance)

        # flow
        flows = [data.get('flow', np.nan) for u, v, data in network.edges(data=True)]
        # flow sum
        if flows:
            sum_flows = np.nansum(commuter_ratios)
        else:
            sum_flows = np.nan
        # add to list
        flow.append(sum_flows)


'''number of edges'''
fig,ax = plt.subplots(figsize=(5, 2.5))
# plt.plot(edge_counts, linestyle='-', linewidth = 2, color='#EBA5D9')
plt.plot(edge_counts, linestyle='-', linewidth = 2, color='#E8B4DD')
# plt.xlabel('File Index')
plt.ylabel('Number of Edges', fontproperties=my_font)
xticks = [0, 31, 59, 90]
xticklabels = ['Jan', 'Feb', 'Mar', 'Apr']
plt.xticks(ticks=xticks, labels=xticklabels, fontproperties=my_font)
# for tick in xticks:
#     plt.axvline(x=tick, color='black', linestyle='--', linewidth = 0.5)
# Starting from first Monday, draw a vertical line every 7 steps
start_index = 5
interval = 7
for i in range(start_index, len(edge_commuter_ratios), interval):
    plt.axvline(x=i, color='black', linestyle='--', linewidth = 0.5)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
for label in ax.get_yticklabels():
    label.set_fontproperties(my_font)
# ax.set_ylim(bottom=0)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('./fig/daily_edge_number_8-4.png', dpi=300)
plt.show()


'''mean commuter ratio'''
fig,ax = plt.subplots(figsize=(5, 2.5))
# plt.plot(edge_commuter_ratios, linestyle='-', linewidth = 2, color='#16C0BF')
plt.plot(edge_commuter_ratios, linestyle='-', linewidth = 2, color='#A6AEE8')
# plt.xlabel('File Index')
plt.ylabel('Mean\nCommuter Ratio', fontproperties=my_font)
xticks = [0, 31, 59, 90]
xticklabels = ['Jan', 'Feb', 'Mar', 'Apr']
plt.xticks(ticks=xticks, labels=xticklabels, fontproperties=my_font)
# Starting from first Monday, draw a vertical line every 7 steps
start_index = 5
interval = 7
for i in range(start_index, len(edge_commuter_ratios), interval):
    plt.axvline(x=i, color='black', linestyle='--', linewidth = 0.5)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_ylim(bottom=0)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('./fig/daily_commuter_ratio_8-4.png', dpi=300)
plt.show()


'''mean distance'''
fig,ax = plt.subplots(figsize=(5, 2.5))
# plt.plot(edge_distance, linestyle='-', linewidth = 2, color='#613197')
plt.plot(edge_distance, linestyle='-', linewidth = 2, color='#74D0CD')
# plt.xlabel('File Index')
plt.ylabel('Mean Distance', fontproperties=my_font)
xticks = [0, 31, 59, 90]
xticklabels = ['Jan', 'Feb', 'Mar', 'Apr']
plt.xticks(ticks=xticks, labels=xticklabels, fontproperties=my_font)
# Starting from first Monday, draw a vertical line every 7 steps
start_index = 5
interval = 7
for i in range(start_index, len(edge_commuter_ratios), interval):
    plt.axvline(x=i, color='black', linestyle='--', linewidth = 0.5)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_ylim(bottom=0)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('./fig/daily_distance_8-4.png', dpi=300)
plt.show()


'''sum flow'''
fig,ax = plt.subplots(figsize=(5, 2.5))
plt.plot(flow, linestyle='-', linewidth = 2, color='#026B93')
# plt.xlabel('File Index')
plt.ylabel('Flow', fontproperties=my_font)
xticks = [0, 31, 59, 90]
xticklabels = ['Jan', 'Feb', 'Mar', 'Apr']
plt.xticks(ticks=xticks, labels=xticklabels, fontproperties=my_font)
# Starting from first Monday, draw a vertical line every 7 steps
start_index = 5
interval = 7
for i in range(start_index, len(edge_commuter_ratios), interval):
    plt.axvline(x=i, color='black', linestyle='--', linewidth = 0.5)
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax.set_ylim(bottom=0)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('./fig/daily_flow_8-4.png', dpi=300)
plt.show()
