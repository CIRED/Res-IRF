import numpy as np
import pandas as pd
from math import floor, ceil
from collections import defaultdict

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
    return flipped


def economic_subplots(df, suptitle, format_axtitle=lambda x: x, format_val=lambda x: '{:.0f}'.format(x), n_columns=3):
    """Plot a line for each index in a subplot.

    Parameters
    ----------
    df: pd.DataFrame
    columns must be years

    suptitle: str

    format_axtitle: function, optional

    format_val: function, optional

    n_columns: int, optional
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
            df.T.plot(ax=ax, color='lightgray', linewidth=0.7)
            df.iloc[k, :].plot(ax=ax, color='black', linewidth=2)
            first_val = df.iloc[k, 0]
            last_val = df.iloc[k, -1]
            ax.annotate(format_val(first_val), xy=(0, first_val), xytext=(-18, 0), color='black',
                        xycoords=ax.get_yaxis_transform(), textcoords='offset points',
                        size=10, va='center')
            ax.annotate(format_val(last_val), xy=(1, last_val), xytext=(-6, 0), color='black',
                        xycoords=ax.get_yaxis_transform(), textcoords='offset points',
                        size=10, va='center')
            ax.xaxis.set_tick_params(which=u'both', length=0)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(format_axtitle(df.index[k]), fontweight='bold', fontsize=10, pad=-1.6)
            ax.get_legend().remove()
        except IndexError:
            ax.axis('off')

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


def stock_attributes_subplots(stock, dict_order={}, suptitle='Buildings stock', option='percent', dict_color=None,
                              n_columns=3, sharey=False):
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


def table_plots(df, level_x, level_y, suptitle='', format_val=lambda x: '{:.0f}'.format(x)):
    """Organized subplots with level_x and level_x values as subplot coordinate.
    
    Parameters
    ----------
    df: pd.DataFrame
        MultiIndex 
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