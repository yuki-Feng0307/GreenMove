import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats

# replace with your own path
with open('4month_network_exp_8-4_geometry.pkl', "rb") as f:
    G = pickle.load(f)

# Extract nodes of type 'grid' and 'park'
grid_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'poly']
park_nodes = [node for node, data in G.nodes(data=True) if data['type'] == 'park']

'''poly'''
# Calculate degrees for 'grid' nodes
grid_degrees = [G.degree[node] for node in grid_nodes]
grid_degree_counts = np.bincount(grid_degrees)
grid_degree_freq = grid_degree_counts / len(grid_nodes)

'''park'''
# Calculate degrees for 'park' nodes
park_degrees = [(node, G.degree(node)) for node in park_nodes]
sorted_degrees = sorted(park_degrees, key=lambda x: x[1], reverse=True)
park_nodes, park_degrees = zip(*sorted_degrees)



'''Fig'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
for ax in [ax1, ax2]:
    ax.set_aspect('auto')  
    ax.set_box_aspect(0.7)

ax1.scatter(np.log10(np.nonzero(grid_degree_freq)[0]), np.log10(grid_degree_freq[np.nonzero(grid_degree_freq)]), marker='o', color='#D6D6D6', alpha=0.6)
# plt.plot(x_fit, y_fit, color='gray', label=f'$y = Cx^{{-{alpha:.2f}}}$')
ax1.set_xlabel('Polygon Degree', fontproperties=my_font)
ax1.set_ylabel('Frequency', fontproperties=my_font)
ax1.set_xticks(np.arange(0, 3), [f'$10^{{{i}}}$' for i in range(3)], fontproperties=my_font)
ax1.set_yticks(np.arange(-5, 0), [f'$10^{{{i}}}$' for i in range(-5, 0)], fontproperties=my_font)

indices = list(range(len(park_nodes)))
ax2.scatter(indices, park_degrees, color='#ACBF9F')
ax2.set_xlabel('Park Nodes', fontproperties=my_font)
ax2.set_ylabel('Degree', fontproperties=my_font)
ax2.set_xticks([0, len(park_nodes)-1])

ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.yaxis.get_major_formatter().set_scientific(True)
ax2.yaxis.get_major_formatter().set_powerlimits((-3, 4))

plt.tight_layout()
plt.savefig('./fig/Degree_distribution.png', dpi=300)
plt.show()