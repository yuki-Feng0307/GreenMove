import os
import pickle
import numpy as np
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict
from datetime import datetime
import networkx as nx
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday



class ChineseHolidaysCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('元旦', month=1, day=1),
        Holiday('春节假期第一天', month=1, day=31),
        Holiday('春节假期第二天', month=2, day=1),
        Holiday('春节假期第三天', month=2, day=2),
        Holiday('春节假期第四天', month=2, day=3),
        Holiday('春节假期第五天', month=2, day=4),
        Holiday('春节假期第六天', month=2, day=5),
        Holiday('春节假期第七天', month=2, day=6),
        Holiday('清明节第一天', month=4, day=5),
        Holiday('清明节第二天', month=4, day=6),
        Holiday('清明节第三天', month=4, day=7),

    ]
chinese_calendar = ChineseHolidaysCalendar()

def is_holiday(date, calendar):
    
    first_day_of_month = date.replace(day=1)
    last_day_of_month = date.replace(day=(pd.Timestamp(date.year, date.month, 1) + pd.offsets.MonthEnd(0)).day)

    holidays = calendar.holidays(start=first_day_of_month, end=last_day_of_month)
    return date in holidays

def is_weekend(date):
    
    return date.weekday() >= 5 

def is_tiaoxiu(date):
    date1 = datetime(2014, 1, 26)
    date2 = datetime(2014, 2, 8)
    return date == date1 or date == date2


# replace with your own path
base_path = 'daily_segment_network_exp_8-4_geometry/'

# edges
edge_counts_weekday = defaultdict(lambda: defaultdict(float))
edge_counts_weekend = defaultdict(lambda: defaultdict(float))
# commuter_ratios
edge_commuter_ratios_weekday = defaultdict(lambda: defaultdict(float))
edge_commuter_ratios_weekend = defaultdict(lambda: defaultdict(float))
# edge_distance
edge_distance_weekday = defaultdict(lambda: defaultdict(float))
edge_distance_weekend = defaultdict(lambda: defaultdict(float))
# flow
flow_weekday = defaultdict(lambda: defaultdict(float))
flow_weekend = defaultdict(lambda: defaultdict(float))

weekday_count = 0
weekend_count = 0

filter_file_names = ['2014-02-12', '2014-02-13', '2014-02-14', '2014-02-15', '2014-02-16']

for folder_name in os.listdir(base_path):
    if folder_name not in filter_file_names:
        folder_path = os.path.join(base_path, folder_name)

        try:
            date = datetime.strptime(folder_name, '%Y-%m-%d')
        except ValueError:
            # skip invalid date
            continue

        if (is_weekend(date) and not is_tiaoxiu(date)) or is_holiday(date, chinese_calendar):
            weekend_count += 1
            for segment in ['morning', 'noon', 'afternoon', 'evening']:
                pkl_path = os.path.join(folder_path, f'{segment}.pkl')
                if not os.path.exists(pkl_path):
                    print(f"{pkl_path} does not exist")
                    continue

                with open(pkl_path, 'rb') as f:
                    graph = pickle.load(f)

                # edge_counts
                edge_count = len(graph.edges())
                edge_counts_weekend[date.strftime('%Y-%m')][segment] += edge_count

                # commuter_ratio
                commuter_ratios = [data.get('commuter_ratio', np.nan) for u, v, data in graph.edges(data=True)]
                if commuter_ratios:
                    average_commuter_ratio = np.nanmean(commuter_ratios)
                else:
                    average_commuter_ratio = np.nan
                edge_commuter_ratios_weekend[date.strftime('%Y-%m')][segment] += average_commuter_ratio

                # distance
                distances = [data.get('distance', np.nan) for u, v, data in graph.edges(data=True)]
                if distances:
                    average_distance = np.nanmean(distances)
                else:
                    average_distance = np.nan
                edge_distance_weekend[date.strftime('%Y-%m')][segment] += average_distance

                # flow
                flows = [data.get('flow', np.nan) for u, v, data in graph.edges(data=True)]
                if flows:
                    sum_flows = np.nansum(flows)
                else:
                    sum_flows = np.nan
                flow_weekend[date.strftime('%Y-%m')][segment] += sum_flows


        else:
            weekday_count += 1
            for segment in ['morning', 'noon', 'afternoon', 'evening']:
                pkl_path = os.path.join(folder_path, f'{segment}.pkl')
                if not os.path.exists(pkl_path):
                    print(f"{pkl_path} does not exist")
                    continue

                with open(pkl_path, 'rb') as f:
                    graph = pickle.load(f)

                # edge_counts
                edge_count = len(graph.edges())
                
                edge_counts_weekday[date.strftime('%Y-%m')][segment] += edge_count

                # commuter_ratio
                commuter_ratios = [data.get('commuter_ratio', np.nan) for u, v, data in graph.edges(data=True)]
                if commuter_ratios:
                    average_commuter_ratio = np.nanmean(commuter_ratios)
                else:
                    average_commuter_ratio = np.nan
                edge_commuter_ratios_weekday[date.strftime('%Y-%m')][segment] += average_commuter_ratio

                # distance
                distances = [data.get('distance', np.nan) for u, v, data in graph.edges(data=True)]
                if distances:
                    average_distance = np.nanmean(distances)
                else:
                    average_distance = np.nan
                edge_distance_weekday[date.strftime('%Y-%m')][segment] += average_distance

                # flow
                flows = [data.get('flow', np.nan) for u, v, data in graph.edges(data=True)]
                if flows:
                    sum_flows = np.nansum(flows)
                else:
                    sum_flows = np.nan
                flow_weekday[date.strftime('%Y-%m')][segment] += sum_flows




def caculate_daily_mean(d,count):
    for month_year, segments in d.items():
        for segment, total_edge_count in segments.items():
            d[month_year][segment] = total_edge_count / count
    return d

edge_counts_weekend = caculate_daily_mean(edge_counts_weekend,weekend_count)
edge_counts_weekday = caculate_daily_mean(edge_counts_weekday,weekday_count)

edge_commuter_ratios_weekend = caculate_daily_mean(edge_commuter_ratios_weekend,weekend_count)
edge_commuter_ratios_weekday = caculate_daily_mean(edge_commuter_ratios_weekday,weekday_count)

edge_distance_weekend = caculate_daily_mean(edge_distance_weekend,weekend_count)
edge_distance_weekday = caculate_daily_mean(edge_distance_weekday,weekday_count)

flow_weekend = caculate_daily_mean(flow_weekend,weekend_count)
flow_weekday = caculate_daily_mean(flow_weekday,weekday_count)



def plot_line(edge_counts_weekday, edge_counts_weekend, title, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
  
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    ax[0].set_title('Weekdays', fontproperties=my_font, size=15)
    # ax[0].set_xlabel('Time Period')
    ax[0].set_ylabel(title, fontproperties=my_font, size=15)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(['MOR', 'MID', 'AFT', 'EVE'], fontproperties=my_font, size=15)
    ax[0].grid(True, linestyle='--', linewidth=0.5)

    # colors = ['#B3D4BB', '#9B7BAB', '#E6B1AF', '#DEDBE2']
    # colors = ['#647A9C', '#3E3D45', '#B4CABD', '#495732']
    colors = ['#647A9C', '#3E3D45', '#B4CABD', '#cccccc']
    for i, (month, counts) in enumerate(edge_counts_weekday.items()):
        time_periods = ['morning', 'noon', 'afternoon', 'evening']
        values = [counts[period] for period in time_periods]
        ax[0].plot(time_periods, values, label=month, color=colors[i], linewidth=3)

    # ax[0].legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax[1].set_title('Holidays', fontproperties=my_font, size=15)
    # ax[1].set_xlabel('Time Period')
    ax[1].set_ylabel(title, fontproperties=my_font, size=15)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(['MOR', 'MID', 'AFT', 'EVE'], fontproperties=my_font, size=15)
    ax[1].grid(True, linestyle='--', linewidth=0.5)

    for i, (month, counts) in enumerate(edge_counts_weekend.items()):
        time_periods = ['morning', 'noon', 'afternoon', 'evening']
        values = [counts[period] for period in time_periods]
        ax[1].plot(time_periods, values, label=month, color=colors[i], linewidth=3)

    # ax[1].legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def caculate_perc_cumsum(edge_counts_weekday):
    months = list(edge_counts_weekday.keys())
    total_counts = {month: sum(edge_counts_weekday[month].values()) for month in months}

    morning_perc = []
    noon_perc = []
    afternoon_perc = []
    evening_perc = []
    for month in months:
        morning_perc.append(edge_counts_weekday[month]['morning'] / total_counts[month] * 100)
        noon_perc.append(edge_counts_weekday[month]['noon'] / total_counts[month] * 100)
        afternoon_perc.append(edge_counts_weekday[month]['afternoon'] / total_counts[month] * 100)
        evening_perc.append(edge_counts_weekday[month]['evening'] / total_counts[month] * 100)

    morning_cumsum = np.array(morning_perc)
    noon_cumsum = morning_cumsum + np.array(noon_perc)
    afternoon_cumsum = noon_cumsum + np.array(afternoon_perc)
    evening_cumsum = afternoon_cumsum + np.array(evening_perc)

    return morning_cumsum,noon_cumsum,afternoon_cumsum,evening_cumsum


def plot_fill(edge_counts_weekday, edge_counts_weekend, title, save_path):
    months = list(edge_counts_weekday.keys())

    weekday_morning,weekday_noon,weekday_afternoon,weekday_evening = caculate_perc_cumsum(edge_counts_weekday)
    weekend_morning, weekend_noon, weekend_afternoon, weekend_evening = caculate_perc_cumsum(edge_counts_weekend)
    print(weekday_morning)
    print(weekday_noon)
    print(weekday_afternoon)
    print(weekday_evening)
    print(weekend_morning)
    print(weekend_noon)
    print(weekend_afternoon)
    print(weekend_evening)
    print('---------------')

    fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


    ax[0].set_title('Weekdays', fontproperties=my_font, size=15)
    # ax[0].set_xlabel('Time Period')
    ax[0].set_ylabel(title, fontproperties=my_font, size=15)
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr'], fontproperties=my_font, size=15)
    # ax[0].grid(True, linestyle='--', linewidth=0.5)

    # colors = ['#B3D4BB', '#9B7BAB', '#E6B1AF', '#DEDBE2']
    colors = ['#cccccc', '#250e3f', '#806590', '#dea3c4']

    ax[0].fill_between(months, weekday_morning, color=colors[0], label='Morning')
    ax[0].fill_between(months, weekday_noon, weekday_morning, color=colors[1], label='Noon')
    ax[0].fill_between(months, weekday_afternoon, weekday_noon, color=colors[2], label='Afternoon')
    ax[0].fill_between(months, weekday_evening, weekday_afternoon, color=colors[3], label='Evening')


    ax[1].set_title('Holidays', fontproperties=my_font, size=15)
    # ax[1].set_xlabel('Time Period')
    ax[1].set_ylabel(title, fontproperties=my_font, size=15)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr'], fontproperties=my_font, size=15)
    # ax[1].grid(True, linestyle='--', linewidth=0.5)

    ax[1].fill_between(months, weekend_morning, color=colors[0], label='Morning')
    ax[1].fill_between(months, weekend_noon, weekend_morning, color=colors[1], label='Noon')
    ax[1].fill_between(months, weekend_afternoon, weekend_noon, color=colors[2], label='Afternoon')
    ax[1].fill_between(months, weekend_evening, weekend_afternoon, color=colors[3], label='Evening')

    # ax[1].legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()



plot_line(edge_counts_weekday, edge_counts_weekend,'Mean Number of Edges',
                 './fig/segment_edge_count_8-4.png')
plot_line(edge_commuter_ratios_weekday, edge_commuter_ratios_weekend,'Mean Commuter Ratio',
                 './fig/segment_commuter_ratio_8-4.png')
plot_line(edge_distance_weekday, edge_distance_weekend,'Mean Distance(km)',
                 './fig/segment_distance_8-4.png')
plot_line(flow_weekday, flow_weekend,'Flow',
                 './fig/segment_flow_8-4.png')

plot_fill(edge_counts_weekday, edge_counts_weekend,'Mean Number of Edges(%)',
                 './fig//segment_fill_edge_count_8-4.png')
plot_fill(edge_commuter_ratios_weekday, edge_commuter_ratios_weekend,'Mean Commuter Ratio(%)',
                 './fig//segment_fill_commuter_ratio_8-4.png')
plot_fill(edge_distance_weekday, edge_distance_weekend,'Mean Distance(%)',
                 './fig/segment_fill_distance_8-4.png')
plot_fill(flow_weekday, flow_weekend,'Flow(%)',
                 './fig/segment_fill_flow_8-4.png')
