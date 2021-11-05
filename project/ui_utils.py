import numpy as np
import pandas as pd
from math import floor, ceil
from collections import defaultdict
from itertools import product
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def reverse_nested_dict(data_dict):
    """
    {(outerKey, innerKey): values for outerKey, innerDict in item_dict['val'].items() for
              innerKey, values in innerDict.items()}
    """
    flipped = defaultdict(dict)
    for key, val in data_dict.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    return dict(flipped)


def simple_plot(x, y, xlabel, ylabel, format_x=None, format_y=None, save=None):
    """Make pretty simple Line2D plot.
    
    Parameters
    ----------
    x: list-like
    y: list-like
    x_label: str
    y_label: str
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_tick_params(which=u'both', length=0)
    
    if format_y is not None:
        if format_y == 'percent':
            format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
        elif format_y == 'million':
            format_y = lambda y, _: '{:,.0f}M'.format(y / 1000000)      
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

    if format_x is not None:
        if format_x == 'percent':
            format_x = lambda x, _: '{:,.0f}%'.format(x * 100)
        elif format_x == 'million':
            format_x = lambda x, _: '{:,.0f}M'.format(x / 1000000)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    
    plt.show()
    
    if save is not None:
        plt.savefig(save)
    
    
def simple_pd_plot(df, xlabel, ylabel, colors=None, format_x=None, format_y=None, save=None, figsize='big', scatter_list=None):
    """Make pretty simple Line2D plot.
    
    Parameters
    ----------
    df: pd.DataFrame
    x_label: str
    y_label: str
    """
    if figsize == 'big':
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    else:
        fig, ax = plt.subplots(1, 1)
    if colors is None:
        df.plot(ax=ax)
    else:
        df.plot(ax=ax, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.yaxis.set_tick_params(which=u'both', length=0)
    
    if format_y is not None:
        if format_y == 'percent':
            format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
        elif format_y == 'million':
            format_y = lambda y, _: '{:,.0f}M'.format(y / 10**6)
        elif format_y == 'billion':
            format_y = lambda y, _: '{:,.0f}B'.format(y / 10**9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

    if format_x is not None:
        if format_x == 'percent':
            format_x = lambda x, _: '{:,.0f}%'.format(x * 100)
        elif format_x == 'million':
            format_x = lambda x, _: '{:,.0f}M'.format(x / 1000000)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
        
        
    if scatter_list is not None:
        ax.scatter(scatter_list[0], scatter_list[1])

    try:
        ax.get_legend().remove()
        fig.legend(frameon=False)
    except AttributeError:
        pass

    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def economic_subplots(df, suptitle, format_axtitle=lambda x: x, format_val=lambda x: '{:.0f}'.format(x), n_columns=3):
    """Plot a line for each index in a subplot.

    Parameters
    ----------
    df: pd.DataFrame
        columns must be years
    suptitle: str
    format_axtitle: function, optional
    format_val: function, optional
    n_columns: int, default 3
    """
    n_axes = int(len(df.index))
    n_rows = ceil(n_axes / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6), sharex=True, sharey=True)
    fig.suptitle(suptitle, fontsize=20, fontweight='bold')
    for k in range(n_rows * n_columns):
        row = floor(k / n_columns)
        column = k % n_columns
        if n_rows == 1:
            ax = axes[column]
        else:
            ax = axes[row, column]
        try:
            test = df.iloc[k, :]
            df.T.dropna().plot(ax=ax, color='lightgray', linewidth=0.7)
            ds = df.iloc[k, :]
            ds.dropna().plot(ax=ax, color='black', linewidth=2)
            first_val = ds[ds.first_valid_index()]
            last_val = ds[ds.last_valid_index()]
            ax.annotate(format_val(first_val), xy=(0, first_val), xytext=(-18, 0), color='black',
                        xycoords=ax.get_yaxis_transform(), textcoords='offset points',
                        size=10, va='center')
            ax.annotate(format_val(last_val), xy=(1, last_val), xytext=(-6, 0), color='black',
                        xycoords=ax.get_yaxis_transform(), textcoords='offset points',
                        size=10, va='center')
            ax.xaxis.set_tick_params(which=u'both', length=0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.figure.autofmt_xdate()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(format_axtitle(df.index[k]), fontweight='bold', fontsize=10, pad=-1.6)
            ax.get_legend().remove()
        except IndexError:
            ax.axis('off')

    plt.show()

    
def scenario_grouped_subplots(df_dict, suptitle='', n_columns=3, format_y=lambda y, _: y, rotation=0, nbins=None, save=None):
    """
    Plot a line for each index in a subplot.

    Parameters
    ----------
    df_dict: dict
        df_dict values are pd.DataFrame (index=years, columns=scenario)
    suptitle: str, optional
    format_y: function, optional
    n_columns: int, default 3
    """
    list_keys = list(df_dict.keys())

    sns.set_palette(sns.color_palette('husl', df_dict[list_keys[0]].shape[1]))

    n_axes = int(len(list_keys))
    n_rows = ceil(n_axes / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6), sharex=True, sharey=True)
    fig.suptitle(suptitle, fontsize=20, fontweight='bold')
    for k in range(n_rows * n_columns):

        row = floor(k / n_columns)
        column = k % n_columns
        if n_rows == 1:
            ax = axes[column]
        else:
            ax = axes[row, column]
        try:
            key = list_keys[k]
            df_dict[key].sort_index().plot(ax=ax, linewidth=1)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
            ax.xaxis.set_tick_params(which=u'both', length=0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if nbins is not None:
                plt.locator_params(axis='x', nbins=nbins)
            
            ax.yaxis.set_tick_params(which=u'both', length=0)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

            # ax.get_yaxis().set_visible(False)
            if isinstance(key, tuple):
                ax.set_title('{}-{}'.format(key[0], key[1]), fontweight='bold', fontsize=10, pad=-1.6)
            else:
                ax.set_title(key, fontweight='bold', fontsize=10, pad=-1.6)
            if k == 0:
                handles, labels = ax.get_legend_handles_labels()
                labels = [l.replace('_', ' ') for l in labels]
            ax.get_legend().remove()
        except IndexError:
            ax.axis('off')

    fig.legend(handles, labels, loc='upper right', frameon=False)
    # plt.legend(frameon=False)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def distribution_scatter(df, column_x, column_y, dict_color, level='Energy performance',
                         xlabel=None, ylabel=None, ax_title=None,
                         format_x=lambda x, _: '{:,.0f}'.format(x), format_y=lambda y, _: '{:,.0f}'.format(y)):
    """Plot df values by df.index.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    scatter_x = df[column_x].to_numpy()
    scatter_y = df[column_y].to_numpy()
    group = df.index.get_level_values(level).to_numpy()
    for g in np.unique(group):
        ix = np.where(group == g)
        ax.scatter(scatter_x[ix], scatter_y[ix], c=dict_color[g], label=g, s=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
        ax.xaxis.set_tick_params(which=u'both', length=0)
        ax.yaxis.set_tick_params(which=u'both', length=0)

        if ax_title is not None:
            ax.set_title(ax_title, fontweight='bold')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    leg = plt.legend(title=level)
    leg.get_frame().set_linewidth(0.0)


def economic_boxplots(df, xlabel=None, ylabel=None, ax_title=None,
                      format_x=lambda x, _: '{:,.0f}'.format(x), format_y=lambda y, _: '{:,.0f}'.format(y)):

    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
    df.boxplot(ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.yaxis.set_tick_params(which=u'both', length=0)
    if ax_title is not None:
        ax.set_title(ax_title, fontweight='bold')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def table_plots(df, level_x=0, level_y=1, suptitle='', format_val=lambda x: '{:.0f}'.format(x)):
    """Organized subplots with level_x and level_x values as subplot coordinate.
    
    Parameters
    ----------
    df: pd.DataFrame
        2 levels MultiIndex 
    level_x: str
    level_y: str
    suptitle: str, optional
    format_val: function, optional
    """
    
    index = df.index.get_level_values(level_x).unique()
    columns = df.index.get_level_values(level_y).unique()
    coord = list(product(range(0, len(index)), range(0, len(columns))))

    fig, axes = plt.subplots(len(index), len(columns), figsize=(12.8, 9.6), sharex='col', sharey='row')
    fig.suptitle(suptitle, fontsize=20, fontweight='bold')
    fig.subplots_adjust(top=0.95)

    for row, col in coord:
        ax = axes[row, col]

        if col == 0:
            ax.set_ylabel(index[row], labelpad=5, fontdict=dict(weight='bold'))
            ax.yaxis.set_label_position('left')
        if row == len(index) - 1:
            ax.set_xlabel(columns[col], labelpad=5, fontdict=dict(weight='bold'))
            ax.xaxis.set_label_position('bottom')

        try:
            df.loc[index[row], columns[col]].plot(ax=ax, color='black', linewidth=2)
            df.T.plot(ax=ax, color='lightgray', linewidth=0.5)
            df.loc[index[row], columns[col]].plot(ax=ax, color='black', linewidth=2)
            ax.get_legend().remove()
            
            # ax.get_yaxis().set_visible(False)
            ax.set_yticklabels([])

            first_val = df.loc[index[row], columns[col]].iloc[0]
            last_val = df.loc[index[row], columns[col]].iloc[-1]
            
            ax.annotate(format_val(first_val), xy=(0, first_val), xytext=(-12, 0), color='black',
                        xycoords=ax.get_yaxis_transform(), textcoords='offset points',
                        size=8, va='center')
            ax.annotate(format_val(last_val), xy=(1, last_val), xytext=(0, 0), color='black',
                        xycoords=ax.get_yaxis_transform(), textcoords='offset points',
                        size=8, va='center')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.xaxis.set_tick_params(which=u'both', length=0)
            ax.yaxis.set_tick_params(which=u'both', length=0)
            
        except:
            ax.axis('off')
            continue
            

def table_plots_scenarios(dict_df, suptitle='', format_y=lambda y, _: y):
    """Organized subplots key[0], key[1] values as subplot coordinate.
    
    Parameters
    ----------
    dict_df: dict
        2 levels MultiIndex 

    suptitle: str, optional
    format_y: function, optional
    """
    index = sorted(list(set([k[0] for k in dict_df.keys()])), reverse=False)
    columns = sorted(list(set([k[1] for k in dict_df.keys()])), reverse=False)
    coord = list(product(range(0, len(index)), range(0, len(columns))))

    fig, axes = plt.subplots(len(index), len(columns), figsize=(12.8, 9.6), sharex='col', sharey=True)
    fig.suptitle(suptitle, fontsize=20, fontweight='bold')
    fig.subplots_adjust(top=0.95)

    for row, col in coord:
        ax = axes[row, col]

        if col == 0:
            ax.set_ylabel(index[row], labelpad=5, fontdict=dict(weight='bold'))
            ax.yaxis.set_label_position('left')
        if row == len(index) - 1:
            ax.set_xlabel(columns[col], labelpad=5, fontdict=dict(weight='bold'))
            ax.xaxis.set_label_position('bottom')

        try:
            df = dict_df[(index[row], columns[col])]
            df.sort_index().plot(ax=ax, linewidth=1)
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()

            # ax.get_yaxis().set_visible(False)
            # ax.set_yticklabels([])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.xaxis.set_tick_params(which=u'both', length=0)
            ax.yaxis.set_tick_params(which=u'both', length=0)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))


        except KeyError:
            ax.axis('off')
            continue

    fig.legend(handles, labels, loc='upper right', frameon=False)
    plt.show()


def stock_attributes_subplots(stock, dict_order={}, suptitle='Buildings stock', option='percent', dict_color=None,
                              n_columns=3, sharey=False):
    """
    Plots building stock by attribute.

    Parameters
    ----------
    stock
    dict_order
    suptitle
    option
    dict_color
    n_columns
    sharey

    Returns
    -------

    """
    labels = list(stock.index.names)
    stock_total = stock.sum()

    n_axes = int(len(stock.index.names))
    n_rows = ceil(n_axes / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6), sharey=sharey)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')

    for k in range(n_rows * n_columns):

        try:
            label = labels[k]
        except IndexError:
            ax.remove()
            break

        stock_label = stock.groupby(label).sum()
        if label in dict_order.keys():
            stock_label = stock_label.loc[dict_order[label]]

        if option == 'percent':
            stock_label = stock_label / stock_total

        row = floor(k / n_columns)
        column = k % n_columns
        if n_rows == 1:
            ax = axes[column]
        else:
            ax = axes[row, column]

        if dict_color is not None:
            stock_label.plot.bar(ax=ax, color=[dict_color[key] for key in stock_label.index])
        else:
            stock_label.plot.bar(ax=ax)

        ax.xaxis.set_tick_params(which=u'both', length=0)
        ax.xaxis.label.set_size(12)

        if option == 'percent':
            format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
        elif option == 'million':
            format_y = lambda y, _: '{:,.0f}M'.format(y / 1000000)
        else:
            raise

        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
        ax.yaxis.set_tick_params(which=u'both', length=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)


def comparison_stock_attribute(stock1, stock2, attribute, dict_order={}, suptitle='Buildings stocks', option='percent',
                               dict_color=None, width=0.3):
    """
    Make bar plot for 2 BuildingStocks attribute to compare them graphically.

    Parameters
    ----------
    stock1: pd.Series
    stock2: pd.Series
    attribute: str
        Level name of both stocks.
    dict_order: dict, optional
    suptitle: str, optional
    option: str, {'percent', 'million'}
    dict_color: dict, optional
    width: float, default 0.3
    """

    fig, ax = plt.subplots(figsize=(12.8, 9.6))

    stock1_total = stock1.sum()
    stock2_total = stock2.sum()

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')

    stock_attribute1 = stock1.groupby(attribute).sum()
    stock_attribute2 = stock2.groupby(attribute).sum()

    if attribute in dict_order.keys():
        stock_attribute1 = stock_attribute1.loc[dict_order[attribute]]
        stock_attribute2 = stock_attribute2.loc[dict_order[attribute]]

    if option == 'percent':
        stock_attribute1 = stock_attribute1 / stock1_total
        stock_attribute2 = stock_attribute2 / stock2_total

    if dict_color is not None:
        stock_attribute1.plot.bar(ax=ax, color=[dict_color[key] for key in stock_attribute1.index], position=0,
                                  width=width)
        stock_attribute2.plot.bar(ax=ax, color=[dict_color[key] for key in stock_attribute2.index], position=1,
                                  width=width, hatch='/////')

    else:
        stock_attribute1.plot.bar(ax=ax, position=0, width=width)
        stock_attribute2.plot.bar(ax=ax, position=1, width=width, hatch='/////')

    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.xaxis.label.set_size(12)

    if option == 'percent':
        format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
    elif option == 'million':
        format_y = lambda y, _: '{:,.0f}M'.format(y / 1000000)
    else:
        raise

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
    ax.yaxis.set_tick_params(which=u'both', length=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.legend(loc='best', frameon=False)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)


def comparison_stock_attributes(stock1, stock2, dict_order={}, suptitle='Buildings stock', option='percent',
                                dict_color=None, width=0.1, n_columns=3, sharey=True):
    """
    Plot bar plot figure that compare 2 BuildingStocks attributes by attributes.

    Parameters
    ----------
    stock1: pd.Series
    stock2: pd.Series
    dict_order: dict
    suptitle: str
    option: str, {'percent', 'million'}
    dict_color: dict
    width: float
    n_columns: int
    sharey: bool
    """

    attributes = list(stock1.index.names)

    n_axes = int(len(stock1.index.names))
    n_rows = ceil(n_axes / n_columns)
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(12.8, 9.6), sharey=sharey)

    stock1_total = stock1.sum()
    stock2_total = stock2.sum()

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20, fontweight='bold')

    for k in range(n_rows * n_columns):

        try:
            attribute = attributes[k]
        except IndexError:
            ax.remove()
            break

        stock_attribute1 = stock1.groupby(attribute).sum()
        stock_attribute2 = stock2.groupby(attribute).sum()

        if attribute in dict_order.keys():
            stock_attribute1 = stock_attribute1.loc[dict_order[attribute]]
            stock_attribute2 = stock_attribute2.loc[dict_order[attribute]]

        if option == 'percent':
            stock_attribute1 = stock_attribute1 / stock1_total
            stock_attribute2 = stock_attribute2 / stock2_total

        row = floor(k / n_columns)
        column = k % n_columns
        if n_rows == 1:
            ax = axes[column]
        else:
            ax = axes[row, column]

        if dict_color is not None:
            stock_attribute1.plot.bar(ax=ax, color=[dict_color[key] for key in stock_attribute1.index], position=0,
                                      width=width)
            stock_attribute2.plot.bar(ax=ax, color=[dict_color[key] for key in stock_attribute2.index], position=1,
                                      width=width, hatch='/////')

        else:
            stock_attribute1.plot.bar(ax=ax, position=0, width=width)
            stock_attribute2.plot.bar(ax=ax, position=1, width=width, hatch='/////')

        ax.xaxis.set_tick_params(which=u'both', length=0)
        ax.xaxis.label.set_size(12)

        if option == 'percent':
            format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
        elif option == 'million':
            format_y = lambda y, _: '{:,.0f}M'.format(y / 1000000)
        else:
            raise

        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))
        ax.yaxis.set_tick_params(which=u'both', length=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        if k == 0:
            ax.legend(loc='best', frameon=False)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)


def grouped_scenarios(output_dict, level, func='sum', weight=None):
    """
    Parameters
    ----------
    output_dict: dict
        {scenario: pd.DataFrame(index=segments, column=years)}
    level: str
    func: str, {'sum', 'mean', 'weighted_mean'}, default 'sum'
    weight: dict, optional
    
    Returns
    -------
    dict
        {group: pd.DataFrame(index=years, column=scenarios)}
        
    Example
    -------
    >>> index = pd.MultiIndex.from_tuples(list(product(['A', 'B'], ['foo', 'bar'])))
    >>> output_dict = {'reference': pd.DataFrame(np.random.randint(0, 100, size=(3, 2)), index=index, columns=[2018, 2019])}
    """
    output = dict()
    for key in output_dict.keys():
        if func == 'sum':
            output[key] = output_dict[key].groupby(level).sum().T.to_dict()
        elif func == 'mean':
            output[key] = output_dict[key].groupby(level).mean().T.to_dict()
        elif func == 'weighted_mean':
            output[key] = ((output_dict[key] * weight[key]).groupby(level).sum() / weight[key].groupby(level).sum()).T.to_dict()

    output = reverse_nested_dict(output)
    return {k: pd.DataFrame(output[k]) for k in output.keys()}


def uncertainty_area_plot(data, idx_ref, title, xlabel, ylabel, leg=False, version='simple',
                          format_y=None, format_x=None):
    """Plot multi scenarios and uncertainty area between lower value and higher value of scenarios.

    Parameters
    ----------
    data: pd.DataFrame
        columns represent one scenario
    idx_ref
    title: str
    leg: bool, default True
    """

    data_min = data.min(axis=1)
    data_max = data.max(axis=1)
    data_ref = data.loc[:, idx_ref]

    fig, ax = plt.subplots(1, 1, figsize=[12.8, 9.6])
    fig.suptitle(title, fontweight='bold')
    fig.subplots_adjust(top=0.85)

    if version != 'simple':
        data.plot(ax=ax, linewidth=0.9)
    data_ref.plot(ax=ax, linewidth=1.8, c='black')
    plt.fill_between(data_min.index, data_min.values, data_max.values, alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlabel(xlabel, loc='right')
    ax.xaxis.set_tick_params(which=u'both', length=0)

    ax.set_ylabel(ylabel, loc='top')
    ax.yaxis.set_tick_params(which=u'both', length=0)

    if format_y is not None:
        if format_y == 'percent':
            format_y = lambda y, _: '{:,.0f}%'.format(y * 100)
        elif format_y == 'million':
            format_y = lambda y, _: '{:,.0f}M'.format(y / 10 ** 6)
        elif format_y == 'billion':
            format_y = lambda y, _: '{:,.0f}B'.format(y / 10 ** 9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_y))

    if format_x is not None:
        if format_x == 'percent':
            format_x = lambda x, _: '{:,.0f}%'.format(x * 100)
        elif format_x == 'million':
            format_x = lambda x, _: '{:,.0f}M'.format(x / 1000000)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_x))

    if version == 'simple' and leg is True:
        legend = ax.legend(loc='lower left', prop={'size': 10})
        plt.show()

    if version != 'simple':
        if leg is True:
            legend_title = ''
            legend = ax.legend(loc='lower left', prop={'size': 10}, title=legend_title)
            legend.get_title().set_fontsize('10')
            plt.show()
        else:
            handles_labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            fig, ax = plt.subplots(1, 1, figsize=[12.8, 9.6])
            # add the legend from the previous axes
            ax.legend(*handles_labels, loc='center')
            # hide the axes frame and the x/y labels
            ax.axis('off')
            plt.show()


def policies_stacked_plot(df, save=None):
    """Make pretty simple Line2D plot.

    Parameters
    ----------
    df: pd.DataFrame
    save: str, optional
    """
    fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))

    df.plot.area(ax=ax)

    ax.set_ylabel('Total cost (Billions €)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_tick_params(which=u'both', length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.yaxis.set_tick_params(which=u'both', length=0)

    try:
        ax.get_legend().remove()
        fig.legend(frameon=False)
    except AttributeError:
        pass

    if save is not None:
        fig.savefig(save)
