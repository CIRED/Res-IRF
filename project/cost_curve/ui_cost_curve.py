import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cumulated_emission_cost_plot(data, graph, add_title=''):
    """
    Plot McKinsey curve.

    x = 'Cumulated potential emission saving (%)' or x = 'Cumulated dwelling number (%)'
    y = 'Carbon cost (euro/tCO2)'

    Parameters
    ----------
    data: pd.DataFrame
    graph: {'Cumulated potential emission saving (%)', 'Cumulated dwelling number (%)'}
    add_title: str, optional
    """

    fig, ax = plt.subplots(1, 1, figsize=[12.8, 9.6])
    title = "Courbe McKinsey\n {}".format(add_title)
    fig.suptitle(title, fontweight='bold')
    fig.subplots_adjust(top=0.85)

    if graph == 'Cumulated potential emission saving (%)':
        xlabel = "Potentiel de réduction\n % Émission secteur résidentiel 2018"
        x = data[graph]
    elif graph == 'Cumulated dwelling number (%)':
        xlabel = 'Potentiel\n Millions logements'
        x = data[graph]
    else:
        raise

    y = data['Carbon cost (euro/tCO2)']
    ax.step(x, y, linewidth=1)

    ax.set_xlabel(xlabel, loc='right')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    ax.xaxis.set_tick_params(which=u'both', length=0)

    ylabel = 'Coût\n €/tCO2'
    ax.set_ylabel(ylabel, loc='top', rotation='horizontal')
    ax.set_ylim(-1000, 1000)
    ax.yaxis.set_tick_params(which=u'both', length=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.show()


def cost_cumulated_emission_plots(scenarios_dict, addtitle=None, legtitle=None, save=None, xlim=(0, 2000),
                                  graph='Cumulated emission difference (%)'):
    """
    Cumulated potential reduction of emission function of CO2 cost (sorted from lower to higher cost)

    y = 'Cumulated CO2 Potential Emission (%/2018)' or x = 'Cumulated Housing number'
    x = 'CO2 cost (€/tCO2)'
    If X €/tCO2, how many emission is it possible to reduced?

    Parameters
    ----------
    scenarios_dict: dict
        dict that contains one pd.DataFrame by scenarios
    graph: {'Cumulated potential emission saving (%)', 'Cumulated emission difference (%)',
            'Cumulated dwelling number (%)'}
    addtitle: str
    legtitle: str
    """

    sns.set_palette('rocket', n_colors=len(scenarios_dict.keys()))

    fig, ax = plt.subplots(1, 1, figsize=[12.8, 9.6])

    k = 0
    for key in scenarios_dict.keys():
        data = scenarios_dict[key]
        x = data['Carbon cost (euro/tCO2)']
        y = data[graph]
        ax.plot(x, y)
        k += 1

    title = "Coût d'abattement CO2 social - Méthode bilan carbone \n {}".format(addtitle)
    fig.suptitle(title, fontweight='bold')
    fig.subplots_adjust(top=0.85)

    xlabel = "Coût\n Euro par tonne CO2"
    ax.set_xlabel(xlabel, loc='right')
    ax.xaxis.set_tick_params(which=u'both', length=0)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if graph in ['Cumulated potential emission saving (%)', 'Cumulated emission difference (%)']:
        ylabel = "Potentiel de réduction\n % Émission secteur résidentiel 2018"
    elif graph == 'Cumulated dwelling number (%)':
        ylabel = 'Potentiel\n Millions logements'
    else:
        raise

    ax.set_ylabel(ylabel, loc='top')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:,.0%}'.format(y)))
    ax.yaxis.set_tick_params(which=u'both', length=0)

    leg = [str(s).lower().replace('_', ' ').capitalize().split('.')[0] for s in scenarios_dict.keys()]
    ax.legend(leg, loc='best', prop={'size': 12}, title=legtitle, frameon=False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    else:
        plt.show()

