import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import os
from project.parse_input import *
from project.buildings_class import HousingStock, HousingStockConstructed


# seaborn.set(style='ticks')
seaborn.set_palette("rocket")

small_size = 20
medium_size = 25
bigger_size = 30
fig_height = 12.8
fig_width = 9.6

plt.rc('font', size=small_size)
plt.rc('axes', titlesize=bigger_size)
plt.rc('axes', labelsize=medium_size)
plt.rc('xtick', labelsize=medium_size)
plt.rc('ytick', labelsize=medium_size)
plt.rc('legend', fontsize=medium_size)
plt.rc('figure', titlesize=bigger_size)
plt.rc('legend', title_fontsize=medium_size)


def format_ax(ax, format_xaxis=None, format_yaxis=None, rotation=90, template=False):
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

    if template is True:
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


def plot_stock_buildings(stock, level):
    """
    """
    fig, ax = plt.subplots(1, 1, figsize=[fig_height, fig_width])
    df = stock.groupby(level).sum()
    list_year = df.columns

    for yr in list_year:
        df = pd.concat((df, stock[yr].groupby(level).sum()), axis=1)
    df.columns = list_year
    df = df.T
    df.plot.bar(ax=ax, stacked=True, color=[colors_dict[key] for key in df.columns])
    format_ax(ax, format_yaxis='million')
    plt.show()


def subplots_level_stock(stock, level):
    # TODO: million or not as y_axis + x_axis for stock_constructed

    df = stock.groupby(level).sum()

    nb_subplots = len(df.index)
    fig, ax = plt.subplots(1, nb_subplots, sharey=True, figsize=[fig_height, fig_width])
    k = 0
    for _, row in df.iterrows():
        format_ax(ax[k], format_yaxis='million')
        ax[k].set_title(row.name)
        row.plot(ax=ax[k])
        ax[k].tick_params(axis='x', labelrotation=90)
        k += 1
    # add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Buildings number')
    plt.show()


if __name__ == '__main__':
    name_folder = '20210512_172839'
    name_folder = os.path.join(folder['output'], name_folder)
    name_file = os.path.join(name_folder, 'stock_segmented.pkl')
    stock_seg = pd.read_pickle(name_file)

    energy_price = forecast2myopic(energy_prices_dict['energy_price_forecast'], calibration_year)
    year = 2020

    buildings = HousingStock(stock_seg.loc[:, year], levels_dict,
                             label2area=dict_label['label2area'],
                             label2horizon=dict_label['label2horizon'],
                             label2discount=dict_label['label2discount'],
                             label2income=dict_label['label2income'],
                             label2consumption=dict_label['label2consumption'])
    segments_construction = buildings.to_segments_construction(
        ['Energy performance', 'Heating energy', 'Income class', 'Income class owner'], {})
    io_share_seg = buildings.to_io_share_seg()

    buildings_constructed = HousingStockConstructed(pd.Series(dtype='float64', index=segments_construction),
                                                    levels_dict_construction, calibration_year,
                                                    dict_parameters['Flow needed'],
                                                    param_share_multi_family=dict_parameters['Factor share multi-family'],
                                                    os_share_ht=dict_result['Occupancy status share housing type'],
                                                    io_share_seg=io_share_seg,
                                                    stock_area_existing_seg=None,
                                                    label2area=dict_label['label2area_construction'],
                                                    label2horizon=dict_label['label2horizon_construction'],
                                                    label2discount=dict_label['label2discount_construction'],
                                                    label2income=dict_label['label2income'],
                                                    label2consumption=dict_label['label2consumption_construction'])
    buildings_constructed.year = 2019
    buildings_constructed.flow_constructed = 2000
    flow_constructed_seg = buildings_constructed.to_flow_constructed_seg(energy_price,
                                                                         cost_construction=cost_construction)

    def to_info(buildings, yr):
        d = buildings.to_consumption_actual(energy_price)
        temp = dict()
        for key, df in d.items():
            if isinstance(df, pd.DataFrame):
                temp[key] = df.loc[:, yr]
            else:
                temp[key] = df
        df = pd.DataFrame(temp)
        energy_cost = HousingStock.energy_consumption2cost(d['Consumption-actual'], energy_price).loc[:, yr]
        energy_lcc = buildings.to_energy_lcc(energy_price, consumption='actual')
        energy_final_lcc = buildings.to_energy_lcc_final(energy_price, ['Energy performance'], consumption='actual').stack()


    to_info(buildings, 2018)


    name_file = os.path.join(name_folder, 'stock_construction_segmented.pkl')
    stock_construction_seg = pd.read_pickle(name_file)
    # plot_stock_buildings(stock_seg)
    level = 'Energy performance'
    subplots_level_stock(stock_seg, level)
    subplots_level_stock(stock_construction_seg, level)

    name_file = os.path.join(name_folder, 'stock_knowledge_energy_performance.pkl')
    stock_knowledge = pd.read_pickle(name_file)
    subplots_level_stock(stock_knowledge, 'Energy performance final')

    name_file = os.path.join(name_folder, 'stock_knowledge_construction.pkl')
    stock_knowledge_construction = pd.read_pickle(name_file)
    subplots_level_stock(stock_knowledge_construction, 'Energy performance final')

    name_file = os.path.join(name_folder, 'demography.pkl')
    demography = pd.read_pickle(name_file)
    # demography.plot()
    # plt.show()

    print('break')
    print('break')
