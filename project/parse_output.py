import pandas as pd
import os
import pickle
from buildings import HousingStock
from utils import reindex_mi
from parse_input import co2_content_data, energy_prices_dict
from collections import defaultdict


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


def parse_subsidies(buildings, flow_renovation):
    """Parse buildings.policies_detailed and returns:
    1. dict with year as key and DataFrame aggregating all policies as value
    2. dict with policies as key and time DataFrame

    Parameters
    ----------
    buildings: HousingStockRenovated
    flow_renovation: pd.DataFrame
        index: segments + transition, columns: years, value: number of buildings renovating to final state
    """

    d = buildings.policies_detailed[tuple(['Energy performance'])]

    # 1. dict with year as key and DataFrame aggregating all policies as value
    subsides_year = dict()
    for yr, item in d.items():
        temp = []
        for name, df in item.items():
            temp += [df.stack(df.columns.names)]
        subsides_year[yr] = pd.concat(temp, axis=1)
        subsides_year[yr].columns = item.keys()
        subsides_year[yr].replace(0, float('nan'), inplace=True)
        subsides_year[yr].dropna(axis=0, how='any', inplace=True)
        subsides_year[yr].columns = [c.replace('_', ' ').capitalize() for c in subsides_year[yr].columns]

        df = subsides_year[yr].copy()
        df = df.reorder_levels(flow_renovation.index.names)
        subsides_euro = (flow_renovation.loc[:, yr] * df.T).T
        subsides_euro.columns = [c.replace('€/m2', '€') for c in list(subsides_euro.columns)]

        subsides_year[yr] = pd.concat((subsides_year[yr], subsides_euro), axis=1)

    # 2. dict with policies as key and time DataFrame
    subsides_dict = parse_dict(reverse_dict(d))

    for key in list(subsides_dict.keys()):
        subsides_dict[key.replace('€/m2', '€')] = subsides_dict[key] * flow_renovation
    subsides_dict_copy = {key.replace('_', ' ').capitalize(): subsides_dict[key] for key in list(subsides_dict.keys())}

    summary_subsidies = dict()
    for year, df in subsides_year.items():
        summary_subsidies[year] = df.loc[:, [c for c in df.columns if '(€)' in c]].sum()
    summary_subsidies = pd.DataFrame(summary_subsidies).T

    return summary_subsidies, subsides_dict_copy, subsides_year


def parse_output(output, buildings, buildings_constructed, folder_output):
    """Parse Res-IRF output to return understandable data.

    Main output are segmented in 2 categories (stock, and transition flow).
    1. Stock - image of the stock in year y.
        - Housing number
        - Conventional energy consumption (kWh)
        - Conventional energy consumption (kWh/m2)
        - Actual energy consumption (kWh)
        - Actual energy consumption (kWh/M2)
        - Energy cost (€/m2)
        - Energy tax cost (€/m2)
        - Heating intensity (%)
        - Renovation rate (%)

    Stock can be:
    - fully aggregated --> macro
    - aggregated by level

    2. Transition flow (index = segments + ['Energy performance final', 'Heating energy final'])
        - Number of transitions
        - Investment cost (€)
        - Investment cost (€/m2)
        - Subsidies used (€)
        - Subsidies used (€/m2)


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

    # 1. stock
    object_dict = {'Renovation': buildings, 'Construction': buildings_constructed}
    buildings_constructed.to_consumption_actual(energy_prices_dict['energy_price_forecast'])
    output_stock = dict()
    for name, building in object_dict.items():
        output_stock['Stock' + ' - {}'.format(name)] = pd.DataFrame(building.stock_seg_dict)
        output_stock['Stock (m2)' + ' - {}'.format(name)] = (
                    output_stock['Stock' + ' - {}'.format(name)].T * building.area).T
        output_stock['Consumption conventional (kWh/m2)' + ' - {}'.format(name)] = building.consumption_conventional
        output_stock['Consumption conventional (kWh)' + ' - {}'.format(name)] = (
                    building.consumption_conventional * output_stock['Stock (m2)' + ' - {}'.format(name)].T).T
        output_stock['Consumption actual (kWh/m2)' + ' - {}'.format(name)] = building.consumption_actual
        output_stock['Consumption actual (kWh)' + ' - {}'.format(name)] = building.consumption_actual * output_stock[
            'Stock (m2)' + ' - {}'.format(name)]
        output_stock['Budget share (%)' + ' - {}'.format(name)] = building.budget_share
        output_stock['Use intensity (%)' + ' - {}'.format(name)] = building.heating_intensity
        output_stock['Emission (gCO2/m2)' + ' - {}'.format(name)] = HousingStock.mul_consumption(
            output_stock['Consumption actual (kWh/m2)' + ' - {}'.format(name)],
            co2_content_data)
        output_stock['Emission (gCO2)' + ' - {}'.format(name)] = HousingStock.mul_consumption(
            output_stock['Consumption actual (kWh)' + ' - {}'.format(name)],
            co2_content_data)

        if 'total_taxes' in output.keys():
            output_stock['Taxes cost (€/m2)' + ' - {}'.format(name)] = HousingStock.mul_consumption(
                building.consumption_actual,
                output['Energy taxes (€/kWh)' + ' - {}'.format(name)])
        else:
            output_stock['Taxes cost (€/m2)' + ' - {}'.format(name)] = building.consumption_actual * 0
        output_stock['Taxes cost (€)' + ' - {}'.format(name)] = output_stock[
                                                                    'Taxes cost (€/m2)' + ' - {}'.format(name)] * \
                                                                output_stock['Stock (m2)' + ' - {}'.format(name)]

    for key in output_stock.keys():
        if isinstance(output_stock[key], pd.DataFrame):
            output_stock[key].dropna(axis=1, how='all', inplace=True)

    # concatenate data
    temp = ['Stock', 'Stock (m2)', 'Consumption conventional (kWh/m2)', 'Consumption conventional (kWh)',
            'Consumption actual (kWh/m2)', 'Consumption actual (kWh)', 'Budget share (%)', 'Use intensity (%)',
            'Emission (gCO2/m2)', 'Emission (gCO2)']
    for keys in temp:
        output_stock['{} - Construction'.format(keys)] = output_stock['{} - Construction'.format(keys)].reorder_levels(
        output_stock['{} - Renovation'.format(keys)].index.names)
        output_stock[keys] = pd.concat(
            (output_stock['{} - Renovation'.format(keys)], output_stock['{} - Construction'.format(keys)]), axis=0)

    # only for existing buildings
    output_stock['NPV (€/m2) - Renovation'] = dict_pd2df(buildings.npv[('Energy performance',)])
    output_stock['Renovation rate (%) - Renovation'] = dict_pd2df(buildings.renovation_rate_dict[('Energy performance',)])

    # 2 Transitions
    output_flow_transition = dict()
    flow_renovation = dict_pd2df(buildings.flow_renovation_label_energy_dict)
    output_flow_transition['Flow transition'] = flow_renovation
    output_flow_transition['Flow transition (m2)'] = (flow_renovation.T * reindex_mi(buildings.area, flow_renovation.index)).T

    # investment and subsides
    capex_ep = reindex_mi(dict_pd2df(buildings.capex_total[('Energy performance',)]), flow_renovation.index)
    capex_he = reindex_mi(dict_pd2df(buildings.capex_total[('Heating energy',)]), flow_renovation.index)
    output_flow_transition['Capex (€/m2)'] = capex_ep + capex_he
    output_flow_transition['Capex (€)'] = output_flow_transition['Flow transition (m2)'] * output_flow_transition['Capex (€/m2)']

    if buildings.policies_total[('Energy performance',)] != {}:
        subsidies_ep = reindex_mi(dict_pd2df(buildings.policies_total[('Energy performance',)]), flow_renovation.index)
    else:
        subsidies_ep = 0 * capex_ep
    if buildings.policies_total[('Heating energy',)] != {}:
        subsidies_he = reindex_mi(dict_pd2df(buildings.policies_total[('Heating energy',)]), flow_renovation.index)
    else:
        subsidies_he = 0 * capex_he
    output_flow_transition['Subsidies (€/m2)'] = subsidies_ep + subsidies_he
    output_flow_transition['Subsidies (€)'] = output_flow_transition['Flow transition (m2)'] * output_flow_transition['Subsidies (€/m2)']

    flow_renovation_label = dict_pd2df(buildings.flow_renovation_label_dict)
    area = reindex_mi(buildings.label2area, flow_renovation_label.index)
    flow_renovation_label = (flow_renovation_label.T * area).T
    if buildings.policies_total[('Energy performance',)] != {}:
        summary_subsidies, output_subsides_year, output_subsides = parse_subsidies(buildings, flow_renovation_label)
        pickle.dump(output_subsides_year, open(os.path.join(folder_output, 'output_subsides_year.pkl'), 'wb'))
        pickle.dump(output_subsides, open(os.path.join(folder_output, 'output_subsides.pkl'), 'wb'))

    # 3. Quick summary
    summary = dict()
    summary['Stock'] = output_stock['Stock'].sum(axis=0)
    summary['Consumption conventional (kWh)'] = output_stock['Consumption conventional (kWh)'].sum(axis=0)
    summary['Consumption actual (kWh)'] = output_stock['Consumption actual (kWh)'].sum(axis=0)
    summary['Emission (gCO2)'] = output_stock['Emission (gCO2)'].sum(axis=0)
    summary['Use intensity renovation (%)'] = (output_stock['Use intensity (%)'] * output_stock['Stock']).sum(axis=0) / output_stock['Stock'].sum(axis=0)
    summary['Renovation rate renovation (%)'] = (output_stock['Renovation rate (%) - Renovation'] * output_stock['Stock - Renovation']).sum(axis=0) / output_stock['Stock - Renovation'].sum(axis=0)
    summary['Flow transition renovation'] = output_flow_transition['Flow transition'].sum(axis=0)
    summary['Capex renovation (€)'] = output_flow_transition['Capex (€)'].sum(axis=0)
    summary['Subsidies renovation (€)'] = output_flow_transition['Subsidies (€)'].sum(axis=0)
    summary = pd.DataFrame(summary)
    summary.dropna(axis=0, thresh=4, inplace=True)

    # add number of F and G buildings
    summary['Stock FG'] = output_stock['Stock'].groupby('Energy performance').sum().loc[['F', 'G'], :]
    summary['Stock CDEFG'] = output_stock['Stock'].groupby('Energy performance').sum().loc[['C', 'D', 'E', 'F', 'G'], :]

    if buildings.policies_total[('Energy performance',)] != {}:
        summary = pd.concat((summary, summary_subsidies), axis=1)

    summary.to_csv(os.path.join(folder_output, 'summary.csv'))
    pickle.dump(output_flow_transition, open(os.path.join(folder_output, 'output_transition.pkl'), 'wb'))
    pickle.dump(output_stock, open(os.path.join(folder_output, 'output_stock.pkl'), 'wb'))



