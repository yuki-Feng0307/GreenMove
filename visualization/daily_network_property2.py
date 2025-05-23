import os
import pickle
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import matplotlib.cm as cm



# replace with your own directory path
directory = 'daily_network_exp_8-4_geometry/'

edge_counts = []
edge_commuter_ratios = []
edge_distance = []
flow = []
weather_data = []  # daily weather


pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
pkl_files.sort()

for filename in pkl_files:
    if filename == '2014-02-04.pkl':
        edge_counts.append(None)
        edge_commuter_ratios.append(None)
        edge_distance.append(None)
        flow.append(None)
        weather_data.append(None)
    if filename == '2014-02-17.pkl':
        for _ in range(5):
            edge_counts.append(None)
            edge_commuter_ratios.append(None)
            edge_distance.append(None)
            flow.append(None)
            weather_data.append(None)
    else:
        file_path = os.path.join(directory, filename)

        with open(file_path, 'rb') as file:
            network = pickle.load(file)
            weather = pickle.load(file)

        weather_data.append(weather)

        # edge numbers
        num_edges = len(network.edges)
        edge_counts.append(num_edges)

        # commuter_ratio
        commuter_ratios = [data.get('commuter_ratio', np.nan) for u, v, data in network.edges(data=True)]
        if commuter_ratios:
            average_commuter_ratio = np.nanmean(commuter_ratios)
        else:
            average_commuter_ratio = np.nan
        edge_commuter_ratios.append(average_commuter_ratio)

        # distance
        distances = [data.get('distance', np.nan) for u, v, data in network.edges(data=True)]
        if distances:
            average_distance = np.nanmean(distances)
        else:
            average_distance = np.nan
        edge_distance.append(average_distance)

        # flow
        flows = [data.get('flow', np.nan) for u, v, data in network.edges(data=True)]
        if flows:
            sum_flows = np.nansum(flows)
        else:
            sum_flows = np.nan
        flow.append(sum_flows)

''' rain or not '''
rainy_flow = []
non_rainy_flow = []
non_rainy_temperatures = []
rainy_temperatures = []

for i, weather in enumerate(weather_data):
    if weather is not None:
        precipitation = weather['precipitation_sum']
        temperature = weather['temperature_2m_max']

        if precipitation == 0:
            # 不下雨天
            non_rainy_flow.append(flow[i])
            non_rainy_temperatures.append(temperature)
        else:
            # 下雨天
            rainy_flow.append(flow[i])
            rainy_temperatures.append(temperature)





''' Rainy vs Low Non-rainy '''

avg_rainy_flow = np.nanmean(rainy_flow) if rainy_flow else np.nan
avg_non_rainy_flow = np.nanmean(non_rainy_flow) if non_rainy_flow else np.nan


# Boxplot
plt.figure(figsize=(3.5, 3))
box_colors = ['#CA7D75', '#483B81']


box = plt.boxplot([rainy_flow, non_rainy_flow],
                  labels=['Rainy', 'Non-Rainy'],
                  patch_artist=True,
                  boxprops=dict(facecolor=box_colors[0], color='black', linewidth=2),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5),
                  flierprops=dict(markerfacecolor=box_colors[0], marker='o', markersize=6, linestyle='none'),
                  medianprops=dict(color='black', linewidth=1.5),
                  widths=0.7)


plt.setp(box['boxes'][1], color='black', linewidth=2, facecolor=box_colors[1])
if len(box['fliers']) > 1:
    plt.setp(box['fliers'][1], markerfacecolor=box_colors[1], markeredgecolor='black')

plt.xlabel('Weather Category', fontproperties=my_font)
plt.ylabel('Flow', fontproperties=my_font)

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 4))
plt.gca().yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig('rainy_nonrainy_compare.png', dpi=300)
plt.show()






# four types
high_temp_rainy_flow = []
low_temp_rainy_flow = []
high_temp_non_rainy_flow = []
low_temp_non_rainy_flow = []

for i, weather in enumerate(weather_data):
    if weather is not None:
        precipitation = weather['precipitation_sum']
        temperature = weather['temperature_2m_mean']

        if precipitation > 0:
            if temperature > 10:
                high_temp_rainy_flow.append(flow[i])
            else:
                low_temp_rainy_flow.append(flow[i])
        else:
            if temperature > 10:
                high_temp_non_rainy_flow.append(flow[i])
            else:
                low_temp_non_rainy_flow.append(flow[i])

# mean flow
def compute_average(flow_data):
    return np.nanmean(flow_data) if flow_data else np.nan

avg_high_temp_rainy = compute_average(high_temp_rainy_flow)
avg_low_temp_rainy = compute_average(low_temp_rainy_flow)
avg_high_temp_non_rainy = compute_average(high_temp_non_rainy_flow)
avg_low_temp_non_rainy = compute_average(low_temp_non_rainy_flow)





''' High Temp vs Low Temp '''

def temp_boxplot(data, ):
    # Boxplot
    plt.figure(figsize=(3.5, 3))

    box_colors = ['#FF6347', '#1E90FF']
    box = plt.boxplot(data,
                      labels=['Above\n10°C', 'Below\n10°C'],
                      patch_artist=True,
                      boxprops=dict(facecolor=box_colors[0], color='black', linewidth=2),
                      whiskerprops=dict(color='black', linewidth=1.5),
                      capprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(markerfacecolor=box_colors[0], marker='o', markersize=6, linestyle='none'),
                      medianprops=dict(color='black', linewidth=1.5),  # 设置中位数线为黑色
                      widths=0.7)  # 调整箱体的宽度

    # 设置低温箱子的颜色
    plt.setp(box['boxes'][1], color='black', linewidth=2, facecolor=box_colors[1])
    if len(box['fliers']) > 1:  # 确保有离群点
        plt.setp(box['fliers'][1], markerfacecolor=box_colors[1], markeredgecolor='black')

    # plt.title('Flow Distribution for High Temp vs Low Temp', fontsize=15)
    plt.xlabel('Mean Temperature', fontproperties=my_font)
    plt.ylabel('Flow', fontproperties=my_font)

    # 设置科学计数法显示
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 4))  # 设置显示范围为小于10的数字为科学计数法
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig('high_low_temp_compare.png', dpi=300)
    plt.show()

high_temp_flow = high_temp_rainy_flow + high_temp_non_rainy_flow
low_temp_flow = low_temp_rainy_flow + low_temp_non_rainy_flow


data_non_rainy = [high_temp_non_rainy_flow, low_temp_non_rainy_flow]
temp_boxplot(data_non_rainy)




# barplot

months = ['Jan', 'Feb', 'Mar', 'Apr']

def plot_monthly(data, color, feature):

    data = [np.nan if x is None else x for x in data]

    monthly_data_avg = []
    month_start_index = [0, 31, 59, 90]

    for i in range(4):
        start_idx = month_start_index[i]
        if i < 3:
            end_idx = month_start_index[i + 1]
        else:
            end_idx = len(data)
        month_data_avg = np.nanmean(data[start_idx:end_idx])
        monthly_data_avg.append(month_data_avg)

    fig, ax = plt.subplots(figsize=(4, 3))
    bars = ax.bar(months, monthly_data_avg, color=color)

    ax.plot(months, monthly_data_avg, marker='v', color='black', linestyle='-', linewidth=2)
    ax.set_ylabel(f'Average {feature}', fontproperties=my_font)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))

    plt.tight_layout()
    plt.savefig(f'monthly_average_{feature}.png', dpi=300)
    plt.show()


plot_monthly(flow, '#026B93', 'flow')
plot_monthly(edge_distance, '#74D0CD', 'distance')


