import pandas as pd
import os
import pickle
from buildings import HousingStock
from utils import reindex_mi
from project.parse_input import colors_dict, co2_content_data, energy_prices_dict


def parse_dict(output):
    """Parse dict and returns pd.DataFrame.

    output = {'key1': {2018: pd.Series(), 2019: pd.Series()}, 'key2': {2019: pd.DataFrame(), 2020: pd.DataFrame()}}
    >>> parse_dict(output)
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
    """Function that takes dict with year as key and return DataFrame with year as column.

    If item is a pd.DataFrame will stack columns as row to be able to concatenate multiple years.

    Parameters
    ----------
    d_pd: dict


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


def to_policies_detailed(buildings, folder_detailed):
    """Parse buildings.policies_detailed and returns DataFrame.

    Parameters
    ----------
    buildings: HousingStockRenovated
    folder_detailed: path
        ok

    """
    d = buildings.policies_detailed[tuple(['Energy performance'])]
    result_subsidies = {}
    for yr, item in d.items():
        temp = []
        for name, df in item.items():
            temp += [df.stack(df.columns.names)]
        result_subsidies[yr] = pd.concat(temp, axis=1)
        result_subsidies[yr].columns = item.keys()
        result_subsidies[yr].replace(0, float('nan'), inplace=True)
        result_subsidies[yr].dropna(axis=0, how='any', inplace=True)
        # result_subsidies[yr].index.names = list(df.index.names) + list(df.columns.names)

    name_file = os.path.join(folder_detailed, 'policies_detailed.pkl')
    with open(name_file, 'wb') as file:
        pickle.dump(result_subsidies, file)
    return result_subsidies


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
    output_stock = dict()
    for name, building in object_dict.items():
        print(name)
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

    # concatenate data
    temp = ['Stock', 'Stock (m2)', 'Consumption conventional (kWh/m2)', 'Consumption conventional (kWh)',
            'Consumption actual (kWh/m2)', 'Consumption actual (kWh)', 'Budget share (%)', 'Use intensity (%)',
            'Emission (gCO2/m2)', 'Emission (gCO2)']
    for keys in temp:
        output_stock[keys] = output_stock['{} - Construction'.format(keys)].reorder_levels(
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

    # 3. Quick summary
    summary = dict()
    summary['Stock renovation'] = output_stock['Stock'].sum(axis=0)
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

    summary.to_csv(os.path.join(folder_output, 'summary.csv'))
    pickle.dump(output_flow_transition, open('output_transition.pkl', 'wb'))
    pickle.dump(output_stock, open('output_transition.pkl', 'wb'))

