import matplotlib.pyplot as plt
import seaborn
import pandas as pd

from input import language_dict


# seaborn.set(style='ticks')
seaborn.set_palette("rocket")

small_size = 20
medium_size = 25
bigger_size = 30

plt.rc('font', size=small_size)
plt.rc('axes', titlesize=bigger_size)
plt.rc('axes', labelsize=medium_size)
plt.rc('xtick', labelsize=medium_size)
plt.rc('ytick', labelsize=medium_size)
plt.rc('legend', fontsize=medium_size)
plt.rc('figure', titlesize=bigger_size)
plt.rc('legend', title_fontsize=medium_size)


def format_ax(ax, format_xaxis=None, format_yaxis=None, rotation=90):
    def format_axis_comma_func(value, _):
        return '{:,.0f}'.format(value)

    def format_axis_percent_func(value, _):
        return '{:,.0f}%'.format(value * 100)

    def format_axis_million_func(value, _):
        return '{:,.0f}M'.format(value / 1000000)

    def format_axis_energy_func(value, _):
        return '{:,.2f}$/kWh'.format(value)

    if format_yaxis == 'comma':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_axis_comma_func))
    elif format_yaxis == 'percent':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_axis_percent_func))
    elif format_yaxis == 'million':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_axis_million_func))
    elif format_yaxis == '$/kWh':
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_axis_energy_func))
    else:
        pass

    if format_xaxis == 'comma':
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis_comma_func))
    elif format_xaxis == 'percent':
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis_percent_func))
    elif format_xaxis == 'million':
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis_million_func))
    elif format_xaxis == '$/kWh':
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_axis_energy_func))
    else:
        pass

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)

    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
    return ax


def plot_stock_buildings(stock, label='Energy performance', option=0):
    """Make bar stacked plot for stock_dict based on label.

    Example: plot_stock_buildings(stock_remained_seg)
    """
    fig, ax = plt.subplots(1, 1, figsize=[12.8, 9.6])
    list_year = list(stock.keys())
    df = stock[list_year[0]].groupby(label).sum()
    for yr in list_year[1:]:
        df = pd.concat((df, stock[yr].groupby(label).sum()), axis=1)
    df.columns = list_year
    df = df.T
    df.plot.bar(ax=ax, stacked=True, color=[language_dict['color'][key] for key in df.columns])
    format_ax(ax, format_yaxis='million')
    plt.show()

