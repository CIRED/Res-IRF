import pandas as pd
import os
import pickle

from buildings import HousingStock
from utils import reindex_mi
from ui_utils import *


def reverse_dict(data):
    flipped = defaultdict(dict)
    for key, val in data.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    return dict(flipped)


def parse_dict(output):
    """Parse dict and returns pd.DataFrame.

    output = {'key1': {2018: pd.Series(), 2019: pd.Series()}, 'key2': {2019: pd.DataFrame(), 2020: pd.DataFrame()}}
    {'key1': pd.DataFrame(), 'key2': pd.DataFrame()}
    """
    new_output = {}
    for key, output_dict in output.items():
        if isinstance(output_dict, dict):
            new_output[key] = dict_pd2df(output_dict)
        elif isinstance(output_dict, pd.DataFrame):
            new_output[key] = output_dict
        elif isinstance(output_dict, pd.Series):
            new_output[key] = output_dict
    return new_output


def dict_pd2df(d_pd):
    """Function that takes dict with year as key and data as value and return DataFrame with year as column.

    If item is a pd.DataFrame will stack columns as row to be able to concatenate multiple years.

    Parameters
    ----------
    d_pd: dict
        {year: pd.DataFrame or pd.Series}


    Returns
    -------
    pd.DataFrame
    """

    def dict_df2df(d):
        return pd.DataFrame({yr: df.stack(df.columns.names) for yr, df in d.items()})

    def dict_ds2df(d):
        return pd.DataFrame(d)

    if d_pd != {}:
        element = list(d_pd.values())[0]
        if isinstance(element, pd.DataFrame):
            return dict_df2df(d_pd)
        if isinstance(element, pd.Series):
            return dict_ds2df(d_pd)


def gap_number(flow_renovation):
    """Calculate the number of label jump.

    Parameters
    ----------
    flow_renovation: pd.Series or pd.DataFrame
        Number of renovation indexed at least by energy performance initial and final

    Returns
    -------
    dict
        Keys being number of jump, and values number of transition.
    """
    dict_count = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    initial = [dict_count[i] for i in flow_renovation.index.get_level_values('Energy performance')]
    final = [dict_count[i] for i in flow_renovation.index.get_level_values('Energy performance final')]
    result = pd.Index([ini - final[n] for n, ini in enumerate(initial)])
    output = dict()
    for i in range(1, 7):
        output['{} label (Thousands)'.format(i)] = flow_renovation[result == i].sum() / 1000

    return output


def parse_subsidies(buildings, flow_renovation_label, area):
    """Parse buildings.policies_detailed and returns:
    1. dict with year as key and DataFrame aggregating all policies as value
    2. dict with policies as key and time DataFrame

    Parameters
    ----------
    buildings: HousingStockRenovated
    flow_renovation_label: pd.DataFrame
        index: segments + transition, columns: years, value: number of buildings renovating to final state
    area: pd.Series
    """

    area = reindex_mi(area, flow_renovation_label.index)
    flow_area_renovation_label = (flow_renovation_label.T * area).T

    d = buildings.subsidies_detailed[tuple(['Energy performance'])]

    # 1. dict with year as key and DataFrame aggregating all subsidies as value
    subsides_year = dict()
    nb_subsidies_year = dict()
    for yr, item in d.items():
        if yr == buildings.calibration_year:
            pass
        else:
            temp = []
            for name, df in item.items():
                temp += [df.stack(df.columns.names)]

            if temp != []:
                subsides_year[yr] = pd.concat(temp, axis=1)
                subsides_year[yr].columns = item.keys()
                # subsides_year[yr].replace(0, float('nan'), inplace=True)
                # subsides_year[yr].dropna(axis=0, how='any', inplace=True)
                subsides_year[yr].columns = [c.replace('_', ' ').capitalize() for c in subsides_year[yr].columns]

                df = subsides_year[yr].copy()
                df = df.reorder_levels(flow_area_renovation_label.index.names)
                df = df.reindex(flow_area_renovation_label.index)

                subsides_euro = (flow_area_renovation_label.loc[:, yr] * df.T).T
                subsides_euro.columns = [c.replace('euro/m2', 'euro') for c in list(subsides_euro.columns)]

                subsides_year[yr] = pd.concat((subsides_year[yr], subsides_euro), axis=1)

                bool_subsidies = df.copy()
                bool_subsidies[bool_subsidies < 0] = 0
                bool_subsidies[bool_subsidies > 0] = 1
                nb_subsidies_year[yr] = (flow_renovation_label.loc[:, yr] * bool_subsidies.T).T
                nb_subsidies_year[yr].columns = [c.replace(' (euro/m2)', ' (Thousands)') for c in
                                                 list(nb_subsidies_year[yr].columns)]
                nb_subsidies_year[yr] = nb_subsidies_year[yr].sum() / 10 ** 3

    # 2. dict with subsidies as key and time DataFrame
    subsides_dict = parse_dict(reverse_dict(d))

    for key in list(subsides_dict.keys()):
        subsides_dict[key.replace('euro/m2', 'euro')] = subsides_dict[key] * flow_area_renovation_label
    subsides_dict_copy = {key.replace('_', ' ').capitalize(): subsides_dict[key] for key in list(subsides_dict.keys())}

    # summary_subsidies contains total subsidies expenditures by year and number of beneficiaries
    summary_subsidies = dict()
    for year, df in subsides_year.items():
        summary_subsidies[year] = df.loc[:, [c for c in df.columns if '(euro)' in c]].sum()
    summary_subsidies = pd.DataFrame(summary_subsidies).T

    summary_subsidies = pd.concat((summary_subsidies, pd.DataFrame(nb_subsidies_year).T), axis=1)

    return summary_subsidies, subsides_dict_copy, subsides_year


def parse_output(output, buildings, buildings_constructed, energy_prices, energy_taxes, energy_taxes_detailed,
                 co2_emission, coefficient, folder_output, lbd_output=False, output_detailed=False):
    """Format Res-IRF output to return understandable data.

    Main output are segmented in 2 categories (stock, and transition flow).
    1. Stock - image of the stock in year y.
        - Housing number
        - Conventional energy consumption (kWh)
        - Conventional energy consumption (kWh/m2)
        - Actual energy consumption (kWh)
        - Actual energy consumption (kWh/M2)
        - Energy cost (euro/m2)
        - Energy tax cost (euro/m2)
        - Heating intensity (%)
        - Renovation rate (%)

    Stock can be:
    - fully aggregated --> macro
    - aggregated by level

    2. Transition flow (index = segments + ['Energy performance final', 'Heating energy final'])
        - Number of transitions
        - Investment cost (euro)
        - Investment cost (euro/m2)
        - Subsidies used (euro)
        - Subsidies used (euro/m2)


        - NB: energy saving and emission saving are not necessary

    Others output:
    1. Knowledge (is defined by final state)

    Returns
    -------
    dict
        Stock
        keys are years, and values pd.DataFrame (index: segments, columns: data)
        Example: {2018: pd.DataFrame(index=segments, columns=['Housing Number', 'Energy consumption', ...])
    dict
        Transition flow
        keys are years, and values pd.DataFrame (index: segments + final state, columns: data)
        Example: {2018: pd.DataFrame(index=segments + ['Energy performance final', 'Heating energy final'],
        columns=['Transition number', 'Capex', 'Subsidies'])
    """

    # 1. Stock
    object_dict = {'Renovation': buildings, 'Construction': buildings_constructed}
    buildings_constructed.to_consumption_actual(energy_prices)
    output_stock = dict()
    for name, building in object_dict.items():
        output_stock['Stock' + ' - {}'.format(name)] = pd.DataFrame(building.stock_dict)
        output_stock['Stock (m2)' + ' - {}'.format(name)] = (
                output_stock['Stock' + ' - {}'.format(name)].T * building.area).T
        output_stock['Consumption conventional (kWh/m2)' + ' - {}'.format(name)] = building.consumption_conventional
        output_stock['Consumption conventional (kWh)' + ' - {}'.format(name)] = (
                building.consumption_conventional * output_stock['Stock (m2)' + ' - {}'.format(name)].T).T
        output_stock['Consumption actual (kWh/m2)' + ' - {}'.format(name)] = building.consumption_actual
        output_stock['Consumption actual (kWh)' + ' - {}'.format(name)] = building.consumption_actual * output_stock[
            'Stock (m2)' + ' - {}'.format(name)]
        output_stock['Budget share (%)' + ' - {}'.format(name)] = building.budget_share
        output_stock['Heating intensity (%)' + ' - {}'.format(name)] = building.heating_intensity
        output_stock['Emission (gCO2/m2)' + ' - {}'.format(name)] = HousingStock.mul_consumption(
            output_stock['Consumption actual (kWh/m2)' + ' - {}'.format(name)], co2_emission)
        output_stock['Emission (gCO2)' + ' - {}'.format(name)] = HousingStock.mul_consumption(
            output_stock['Consumption actual (kWh)' + ' - {}'.format(name)], co2_emission)

        if 'total_taxes' in output.keys():
            output_stock['Taxes cost (euro/m2)' + ' - {}'.format(name)] = HousingStock.mul_consumption(
                building.consumption_actual,
                output['Energy taxes (euro/kWh)' + ' - {}'.format(name)])
        else:
            output_stock['Taxes cost (euro/m2)' + ' - {}'.format(name)] = building.consumption_actual * 0
        output_stock['Taxes cost (euro)' + ' - {}'.format(name)] = output_stock[
                                                                       'Taxes cost (euro/m2)' + ' - {}'.format(name)] * \
                                                                   output_stock['Stock (m2)' + ' - {}'.format(name)]

    for key in output_stock.keys():
        if isinstance(output_stock[key], pd.DataFrame):
            output_stock[key].dropna(axis=1, how='all', inplace=True)

    # aggregated new and existing data
    temp = ['Stock', 'Stock (m2)', 'Consumption conventional (kWh/m2)', 'Consumption conventional (kWh)',
            'Consumption actual (kWh/m2)', 'Consumption actual (kWh)', 'Budget share (%)', 'Heating intensity (%)',
            'Emission (gCO2/m2)', 'Emission (gCO2)']
    for keys in temp:
        output_stock['{} - Construction'.format(keys)] = output_stock['{} - Construction'.format(keys)].reorder_levels(
            output_stock['{} - Renovation'.format(keys)].index.names)
        output_stock[keys] = pd.concat(
            (output_stock['{} - Renovation'.format(keys)], output_stock['{} - Construction'.format(keys)]), axis=0)

    # only for existing buildings
    output_stock['NPV (euro/m2) - Renovation'] = dict_pd2df(buildings.npv[('Energy performance',)])
    output_stock['Renovation rate (%) - Renovation'] = dict_pd2df(
        buildings.renovation_rate_dict[('Energy performance',)])

    # 2 Transitions
    output_flow_transition = dict()
    flow_renovation = dict_pd2df(buildings.flow_renovation_label_energy_dict)
    output_flow_transition['Flow transition'] = flow_renovation
    output_flow_transition['Flow transition (m2)'] = (
                flow_renovation.T * reindex_mi(buildings.area, flow_renovation.index)).T

    output_stock['Renovation'] = flow_renovation.groupby(
        output_stock['Renovation rate (%) - Renovation'].index.names).sum()

    # final just represent energy performance transition and doesn't consider new heating energy
    energy_lcc_final = reindex_mi(dict_pd2df(buildings.energy_lcc_final[('Energy performance',)]['conventional']),
                                  flow_renovation.index)
    output_flow_transition['Energy cost final (euro/m2)'] = energy_lcc_final

    energy_lcc = reindex_mi(dict_pd2df(buildings.energy_lcc[('Energy performance',)]['conventional']),
                            flow_renovation.index)
    output_flow_transition['Energy cost initial (euro/m2)'] = energy_lcc

    energy_lcc_saving = energy_lcc - energy_lcc_final
    output_flow_transition['Energy cost saving (euro/m2)'] = energy_lcc_saving

    # investment and subsides
    capex_ep = reindex_mi(dict_pd2df(buildings.capex[('Energy performance',)]), flow_renovation.index)
    capex_he = reindex_mi(dict_pd2df(buildings.capex[('Heating energy',)]), flow_renovation.index)
    output_flow_transition['Capex wo/ intangible (euro/m2)'] = capex_ep + capex_he
    output_flow_transition['Capex wo/ intangible energy performance (euro/m2)'] = capex_ep
    output_flow_transition['Capex wo/ intangible (euro)'] = output_flow_transition['Flow transition (m2)'] * \
                                                            output_flow_transition['Capex wo/ intangible (euro/m2)']

    output_flow_transition['Capex wo/ intangible energy performance (euro)'] = output_flow_transition[
                                                                                      'Flow transition (m2)'] * \
                                                                                  output_flow_transition[
                                                                                      'Capex wo/ intangible energy performance (euro/m2)']

    capex_ep = reindex_mi(dict_pd2df(buildings.capex_intangible[('Energy performance',)]), flow_renovation.index)
    output_flow_transition['Capex intangible (euro/m2)'] = capex_ep

    temp = output['Cost intangible'][buildings.calibration_year].groupby('Energy performance').mean().loc[
        ['G', 'F', 'E', 'D', 'C'], ['F', 'E', 'D', 'C', 'B', 'A']]
    temp.to_csv(os.path.join(folder_output, 'cost_intangible_ini.csv'))

    capex_ep = reindex_mi(dict_pd2df(buildings.capex_total[('Energy performance',)]), flow_renovation.index)
    capex_he = reindex_mi(dict_pd2df(buildings.capex_total[('Heating energy',)]), flow_renovation.index)
    output_flow_transition['Capex energy performance (euro/m2)'] = capex_ep

    output_flow_transition['Capex w/ intangible (euro/m2)'] = capex_ep + capex_he
    output_flow_transition['Capex w/ intangible (euro)'] = output_flow_transition['Flow transition (m2)'] * \
                                                           output_flow_transition['Capex w/ intangible (euro/m2)']

    if buildings.subsidies_total[('Energy performance',)] != {}:
        subsidies_ep = reindex_mi(dict_pd2df(buildings.subsidies_total[('Energy performance',)]), flow_renovation.index)
    else:
        subsidies_ep = 0 * capex_ep
    if buildings.subsidies_total[('Heating energy',)] != {}:
        subsidies_he = reindex_mi(dict_pd2df(buildings.subsidies_total[('Heating energy',)]), flow_renovation.index)
    else:
        subsidies_he = 0 * capex_he
    output_flow_transition['Subsidies (euro/m2)'] = subsidies_ep + subsidies_he
    output_flow_transition['Subsidies (euro)'] = output_flow_transition['Flow transition (m2)'] * \
                                                 output_flow_transition['Subsidies (euro/m2)']

    flow_renovation_label = dict_pd2df(buildings.flow_renovation_label_dict)
    # area = reindex_mi(buildings.attributes2area, flow_renovation_label.index)
    # flow_area_renovation_label = (flow_renovation_label.T * area).T

    df = (output_stock['Consumption actual (kWh)'].groupby('Heating energy').sum().T * coefficient).T
    result = dict()
    for key, item in energy_taxes_detailed.items():
        key = key.replace('_', ' ').capitalize()
        tax = (df * item).dropna(axis=1).sum()
        result['{} (euro)'.format(key)] = tax
    summary_taxes = pd.DataFrame(result)

    summary_policies = - summary_taxes

    if buildings.subsidies_total[('Energy performance',)] != {}:
        summary_subsidies, output_subsides_year, output_subsides = parse_subsidies(buildings, flow_renovation_label,
                                                                                   buildings.attributes2area)

        summary_policies = pd.concat((summary_policies, summary_subsidies), axis=1)
        cols = [c for c in summary_policies.columns if 'euro' in c]
        policies_stacked_plot(summary_policies.loc[buildings.calibration_year + 1:, cols] / 10 ** 9,
                              save=os.path.join(folder_output, 'summary_policies.png'))

        if False:
            pickle.dump(output_subsides_year, open(os.path.join(folder_output, 'output_subsides_year.pkl'), 'wb'))
            pickle.dump(output_subsides, open(os.path.join(folder_output, 'output_subsides.pkl'), 'wb'))

    detailed = dict()
    detailed['Emission (MtCO2)'] = output_stock['Emission (gCO2)'].sum(axis=0) / 10 ** 12

    detailed['Consumption conventional (TWh)'] = (
                                                         output_stock['Consumption conventional (kWh)'].groupby(
                                                             'Heating energy').sum().T * coefficient).T.sum(
        axis=0) / 10 ** 9

    df = (output_stock['Consumption actual (kWh)'].groupby('Heating energy').sum().T * coefficient).T
    energy_expenditure = (df * energy_prices).dropna(axis=1).sum() / 10 ** 9
    taxes_expenditure = (df * energy_taxes).dropna(axis=1).sum() / 10 ** 9

    detailed['Consumption actual (TWh)'] = df.sum() / 10 ** 9
    for energy in buildings.attributes_values['Heating energy']:
        detailed['Consumption {} (TWh)'.format(energy)] = df.loc[energy, :] / 10 ** 9

    detailed['Heating intensity (%)'] = (output_stock['Heating intensity (%)'] * output_stock['Stock']).sum(axis=0) / \
                                        output_stock['Stock'].sum(axis=0)
    df = (output_stock['Heating intensity (%)'] * output_stock['Stock']).groupby('Income class').sum() / output_stock[
        'Stock'].groupby('Income class').sum()
    for income in buildings.attributes_values['Income class']:
        detailed['Heating intensity {} (%)'.format(income)] = df.loc[income, :]

    detailed['Stock (Thousands)'] = output_stock['Stock'].sum(axis=0) / 10 ** 3
    for label in buildings.total_attributes_values['Energy performance'] + \
                 buildings_constructed.total_attributes_values['Energy performance']:
        detailed['Stock {} (Thousands)'.format(label)] = output_stock['Stock'].groupby('Energy performance').sum().loc[
                                                         label, :] / 1000
    detailed['Flow renovation (Thousands)'] = output_flow_transition['Flow transition'].sum(axis=0) / 10 ** 3

    detailed.update(gap_number(flow_renovation))
    df = flow_renovation.groupby(['Occupancy status', 'Housing type']).sum()
    for dm in df.index:
        detailed['Renovation {} - {} (Thousands)'.format(dm[0], dm[1])] = df.loc[dm, :] / 10 ** 3

    for attribute in ['Income class owner', 'Energy performance', 'Heating energy']:

        df = flow_renovation.groupby([attribute]).sum()
        for idx in df.index:
            detailed['Renovation {} (Thousands)'.format(idx)] = df.loc[idx, :] / 10 ** 3

    temp = buildings.rate_renovation_ini
    temp = temp.groupby(['Occupancy status', 'Housing type']).mean()
    temp.name = 2012
    temp = pd.concat((temp, dict_pd2df(buildings.renovation_rate_dm)), axis=1)
    for dm in temp.index:
        detailed['Renovation rate {} (%)'.format(dm)] = temp.loc[dm, :]

    detailed['Annual renovation expenditure (Billions euro)'] = output_flow_transition[
                                                                    'Capex wo/ intangible (euro)'].sum(axis=0) / 10 ** 9
    detailed['Annual energy expenditure (Billions euro)'] = energy_expenditure
    detailed['Annual energy taxes expenditure (Billions euro)'] = taxes_expenditure
    detailed['Annual subsidies (Billions euro)'] = output_flow_transition['Subsidies (euro)'].sum(axis=0) / 10 ** 9

    detailed['Energy poverty (Thousands)'] = output_stock['Stock'][output_stock['Budget share (%)'] > 0.1].sum(
        axis=0) / 10 ** 3
    detailed['Share energy poverty (%)'] = output_stock['Stock'][output_stock['Budget share (%)'] > 0.1].sum(axis=0) / \
                                           output_stock['Stock'].sum(axis=0)

    detailed = pd.DataFrame(detailed).dropna(how='all', axis=0).T

    detailed = pd.concat((detailed, summary_policies.T), axis=0)
    # detailed.index = detailed.index.astype('int64')

    if lbd_output:
        temp = pd.DataFrame(buildings.stock_knowledge_ep_dict) / 10 ** 6
        temp.index = ['Stock experience {} Mm2'.format(i) for i in temp.index]
        detailed = pd.concat((detailed, temp), axis=0)

        temp = pd.DataFrame(buildings.knowledge_dict)
        temp.index = ['Knowledge {}'.format(i) for i in temp.index]
        detailed = pd.concat((detailed, temp), axis=0)

        temp = HousingStock.lbd(pd.DataFrame(buildings.knowledge_dict), -0.1)
        temp.index = ['Lbd factor {}'.format(i) for i in temp.index]
        detailed = pd.concat((detailed, temp), axis=0)

    detailed.to_csv(os.path.join(folder_output, 'detailed.csv'))

    stock = pd.DataFrame(buildings.stock_dict)
    stock_constructed = pd.DataFrame(buildings_constructed.stock_dict).reorder_levels(stock.index.names)
    stock = pd.concat((stock, stock_constructed), axis=0)
    stock.to_csv(os.path.join(folder_output, 'stock.csv'))


    if output_detailed:
        pickle.dump(output_flow_transition, open(os.path.join(folder_output, 'output_transition.pkl'), 'wb'))
    pickle.dump(output_stock, open(os.path.join(folder_output, 'output_stock.pkl'), 'wb'))


def quick_graphs(folder_output, output_detailed):
    """Returns main comparison graphs.

    - What could be the evolution of actual energy consumption?
    - How many renovation (of at least 1 EPC) could there be by 2050?
    - What could be the evolution of stock energy performance by 2050?
    - What could be the heating intensity by 2050?
    - How many energy poverty households could there be by 2050?

    Parameters
    ----------
    folder_output : str
    output_detailed : bool
    """

    scenarios = [f for f in os.listdir(folder_output) if f not in ['log.txt', '.DS_Store']]
    folders = {scenario: os.path.join(folder_output, scenario) for scenario in scenarios}
    sns.set_palette(sns.color_palette('husl', len(scenarios)))

    folder_output = os.path.join(folder_output, 'img')
    os.mkdir(folder_output)

    detailed = {
        scenario.replace('_', ' '): pd.read_csv(os.path.join(folders[scenario], 'detailed.csv'), index_col=[0]).T for
        scenario in
        scenarios}
    detailed = reverse_nested_dict(detailed)
    detailed = {key: pd.DataFrame(item) for key, item in detailed.items()}

    paths = {scenario: os.path.join(folders[scenario], 'output_stock.pkl') for scenario in scenarios}
    output_stock = {scenario: pickle.load(open(paths[scenario], 'rb')) for scenario in paths}
    output_stock = reverse_nested_dict(output_stock)

    simple_pd_plot(detailed['Consumption actual (TWh)'], 'Years', 'Consumption actual (TWh)',
                   save=os.path.join(folder_output, 'consumption_actual.png'))

    simple_pd_plot(detailed['Heating intensity (%)'], 'Years', 'Heating intensity (%)', format_y='percent',
                   save=os.path.join(folder_output, 'heating_intensity.png'))

    simple_pd_plot(detailed['Flow renovation (Thousands)'], 'Years', 'Flow renovation (Thousands)',
                   save=os.path.join(folder_output, 'flow_renovation.png'))

    scenario_grouped_subplots(grouped_scenarios(output_stock['Renovation'], 'Energy performance'),
                              suptitle='Evolution flow renovation (Thousands)',
                              format_y=lambda y, _: '{:,.0f}'.format(y / 10 ** 3), n_columns=7, rotation=90, nbins=4,
                              save=os.path.join(folder_output, 'renovation_performance.png'))

    scenario_grouped_subplots(grouped_scenarios(output_stock['Renovation'], 'Heating energy'),
                              suptitle='Evolution flow renovation (Thousands)',
                              format_y=lambda y, _: '{:,.0f}'.format(y / 10 ** 3), n_columns=4, rotation=90, nbins=4,
                              save=os.path.join(folder_output, 'renovation_energy.png'))

    scenario_grouped_subplots(grouped_scenarios(output_stock['Renovation'], 'Income class owner'),
                              suptitle='Evolution flow renovation (Thousands)',
                              format_y=lambda y, _: '{:,.0f}'.format(y / 10 ** 3), n_columns=5, rotation=90, nbins=4,
                              save=os.path.join(folder_output, 'renovation_income_class.png'))

    scenario_grouped_subplots(grouped_scenarios(output_stock['Renovation'], ['Occupancy status', 'Housing type']),
                              suptitle='Evolution flow renovation (Thousands)',
                              format_y=lambda y, _: '{:,.0f}'.format(y / 10 ** 3), n_columns=3, rotation=90, nbins=4,
                              save=os.path.join(folder_output, 'renovation_decision_maker.png'))

    simple_pd_plot(detailed['Share energy poverty (%)'], 'Years', 'Share energy poverty (%)',
                   save=os.path.join(folder_output, 'energy_poverty.png'))

    scenario_grouped_subplots(grouped_scenarios(output_stock['Stock - Renovation'], 'Energy performance'),
                              suptitle='Evolution buildings stock (Millions)',
                              format_y=lambda y, _: '{:,.0f}'.format(y / 10 ** 6), n_columns=7, rotation=90, nbins=4,
                              save=os.path.join(folder_output, 'stock_performance.png'))

    weight = dict()
    for key, item in output_stock['Stock - Renovation'].items():
        weight[key] = item[item.index.get_level_values('Energy performance') != 'A']
        weight[key].columns = [c + 1 for c in weight[key].columns]
        weight[key] = weight[key].iloc[:, :-1]

    scenario_grouped_subplots(
        grouped_scenarios(output_stock['Renovation rate (%) - Renovation'], 'Energy performance', func='weighted_mean',
                          weight=weight),
        suptitle='Renovation rate (%)',
        format_y=lambda y, _: '{:.1%}'.format(y), n_columns=7, rotation=90, nbins=4,
        save=os.path.join(folder_output, 'renovation_rate_performance.png'))

    scenario_grouped_subplots(
        grouped_scenarios(output_stock['Renovation rate (%) - Renovation'], 'Heating energy', func='weighted_mean',
                          weight=weight),
        suptitle='Renovation rate (%)',
        format_y=lambda y, _: '{:.1%}'.format(y), n_columns=4, rotation=90, nbins=4,
        save=os.path.join(folder_output, 'renovation_rate_energy.png'))

    scenario_grouped_subplots(
        grouped_scenarios(output_stock['Renovation rate (%) - Renovation'], 'Income class owner', func='weighted_mean',
                          weight=weight),
        suptitle='Renovation rate (%)',
        format_y=lambda y, _: '{:.1%}'.format(y), n_columns=5, rotation=90, nbins=4,
        save=os.path.join(folder_output, 'renovation_rate_income_class.png'))

    scenario_grouped_subplots(
        grouped_scenarios(output_stock['Renovation rate (%) - Renovation'], ['Occupancy status', 'Housing type'],
                          func='weighted_mean', weight=weight),
        suptitle='Renovation rate (%)',
        format_y=lambda y, _: '{:.1%}'.format(y), n_columns=3, rotation=90, nbins=4,
        save=os.path.join(folder_output, 'renovation_rate_decision_maker.png'))

    if output_detailed is False:
        for _, path in paths.items():
            os.remove(path)
