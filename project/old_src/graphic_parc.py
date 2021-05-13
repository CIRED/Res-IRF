from project.graphic import *


def graph_parc(dsp, dsp2=None):
    width = 0.2
    format_yaxis = 'percent'

    legend = True
    fig, axes = plt.subplots(2, 2, figsize=[12.8, 9.6])

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


"""
labels = ['Revenu ménages', "Statut d'occupation"]
make_stacked_barplot(dsp, labels, source='SDES-2018', trad=False)

labels = ['DPE', 'Energie de chauffage']
make_stacked_barplot(dsp, labels, source='SDES-2018', trad=False)

"""

