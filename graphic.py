import matplotlib.pyplot as plt
import seaborn
import pandas as pd


from input import language_dict


dict_color = language_dict['dict_color']

seaborn.set(style='ticks')
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


def format_ax(ax, format_xaxis=None, format_yaxis=None, legend=True, ylim=False, option=0):
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

    if option == 0:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)
    elif option == 1:
        # set the x-spine
        ax.spines['left'].set_position('zero')

        # turn off the right spine/ticks
        ax.spines['right'].set_color('none')
        ax.yaxis.tick_left()

        # set the y-spine
        ax.spines['bottom'].set_position('zero')

        # turn off the top spine/ticks
        ax.spines['top'].set_color('none')
        ax.xaxis.tick_bottom()

    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    if isinstance(ylim, tuple):
        ax.set_ylim(*ylim)

    if legend:
        ax.legend(loc='best')

    return ax


def make_plot(df_dict, title, ylabel=None, format_xaxis=None, format_yaxis=None, legend=False, color=True,
              legendtitle='', ylim=False):
    fig, ax = plt.subplots(1, 1, figsize=[12.8, 9.6])

    linestyle_list = ['solid', 'dotted', 'dashed', 'dashed', 'dashed', 'dashed'] * 10
    marker_list = [None, None, None, "o", ".", ","] * 10
    linewidth = 2
    markersize = 3
    k = 0
    for _, df in df_dict.items():

        if color == True:
            df.plot(ax=ax, title=title, ylabel=ylabel, legend=legend, color=[dict_color[key] for key in df.columns],
                    linestyle=linestyle_list[k], linewidth=linewidth, marker=marker_list[k], markersize=markersize)
        else:
            df.plot(ax=ax, title=title, ylabel=ylabel, legend=legend, linestyle=linestyle_list[k],
                    marker=marker_list[k], linewidth=linewidth, markersize=markersize)

        k += 1

    format_ax(ax, format_xaxis=format_xaxis, format_yaxis=format_yaxis, legend=legend, ylim=ylim)
    plt.legend(title=legendtitle)
    plt.show()

    return None


def make_stacked_barplot(dsp, labels, source=None, format_yaxis='million', rotation=0, legend=True, title=None,
                         cumsum=False, ylim=False):
    fig, ax = plt.subplots(1, 1, figsize=[12.8, 9.6])

    if title == None:
        title = 'Description du parc de logements {} par {} et {}'.format(source, labels[0], labels[1])

    fig.suptitle(title)
    fig.tight_layout()

    ds = dsp.groupby(labels).sum().unstack()
    if cumsum == True:
        ds = ds.fillna(0).cumsum(axis=0)
    # ds = english_to_french(ds)
    ds.plot.bar(ax=ax, stacked=True, color=[dict_color[key] for key in ds.columns])

    ax.set_ylabel(dsp.name, loc='top', rotation='horizontal')
    ax = format_ax(ax, format_yaxis=format_yaxis, legend=legend, ylim=ylim)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation)
    plt.show()
    return ax


def graph_parc(dsp, dsp2=None):
    width = 0.2
    format_yaxis = 'percent'

    legend = True
    fig, axes = plt.subplots(2, 2, figsize=[12.8, 9.6])
    """fig.suptitle('Description of the housing stock: {} // Total housing stock: {:,.0f}'.format(dsp.name, dsp.sum()))
    if isinstance(dsp2, pd.Series):
        fig.suptitle('Description of the housing stocks: [////]:: {} &  {} '.format(dsp2.name, dsp.name))"""

    fig.tight_layout()
    labels = list(dsp.index.names)
    labels.remove('DPE')

    for k, ax in enumerate(axes.flat):
        try:
            label = labels[k]
        except IndexError:
            ax.remove()
            break
        ds = dsp.groupby(label).sum()
        if isinstance(dsp2, pd.Series):
            ds2 = dsp2.groupby(label).sum()

        if format_yaxis == 'percent':
            ds = ds / dsp.groupby(label).sum().sum()
            if isinstance(dsp2, pd.Series):
                ds2 = ds2 / dsp2.groupby(label).sum().sum()

        ds.plot.bar(ax=ax, position=0, width=width, color=[dict_color[key] for key in ds.index])

        if isinstance(dsp2, pd.Series):
            ds2.plot.bar(ax=ax, position=1, width=width, hatch='/////', color=[dict_color[key] for key in ds.index])

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

        if k > 0:
            legend = False
        ax = format_ax(ax, format_yaxis=format_yaxis, legend=legend)
        if legend:
            ax.legend(prop=dict(size=10))
        ax.xaxis.label.set_size(15)
        ax.tick_params(axis='both', which='major', labelsize=12)
    plt.show()


def graph_parc_label(dsp, dsp2=None):

    label = 'DPE'
    width = 0.2
    format_yaxis = 'percent'

    fig, axes = plt.subplots(1, 1, figsize=[12.8, 9.6])
    fig.tight_layout()

    ds = dsp.groupby(label).sum()
    if isinstance(dsp2, pd.Series):
        ds2 = dsp2.groupby(label).sum()
    if isinstance(dsp2, pd.Series):
        ds2 = dsp2.groupby(label).sum()

    if format_yaxis == 'percent':
        ds = ds / dsp.groupby(label).sum().sum()
        if isinstance(dsp2, pd.Series):
            ds2 = ds2 / dsp2.groupby(label).sum().sum()

    ax = ds.plot.bar(position=0, width=width, color=[dict_color[key] for key in ds.index])
    # ds.plot(ax=ax, linewidth=1, color='black')

    if isinstance(dsp2, pd.Series):
        ds2.plot.bar(ax=ax, position=1, width=width, hatch='/////', color=[dict_color[key] for key in ds.index])
        # ds2.plot(ax=ax, linewidth=1, linestyle='dashed', color='black')

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    ax = format_ax(ax, format_yaxis=format_yaxis, legend=True)
    plt.show()


"""
labels = ['Revenu ménages', "Statut d'occupation"]
make_stacked_barplot(dsp, labels, source='SDES-2018', trad=False)

labels = ['DPE', 'Energie de chauffage']
make_stacked_barplot(dsp, labels, source='SDES-2018', trad=False)

"""