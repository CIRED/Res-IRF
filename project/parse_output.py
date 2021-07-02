import pandas as pd
import os
import pickle
from buildings import HousingStock
from utils import reindex_mi, ds_mul_df, get_levels_values
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


def to_detailed_output(buildings, buildings_constructed, output, folder_output):
    """Parsing output concerning financial output: capex, energy cost, subsidies.

    Parameters
    ----------
    buildings: HousingStockRenovated
        buildings after a script
    buildings_constructed: HousingStockConstructed
    folder_output: path
    """

    def capex2cash_flows(capex_ds, columns):
        """Returns DataFrame from Series with value ofr the first year then 0.

        Parameters
        ----------
        capex_ds: pd.Series
            capex
        columns: list
            list of years used as column for the new df

        Returns
        -------
        pd.DataFrame

        Example
        -------
        capex_ds = pd.Series([100, 1000, 10000])
        >>> capex2cash_flows(capex_ds, [2018, 2019, 2020])
        pd.DataFrame([100, 1000, 10000] * 3, columns=[2018, 2019, 2020])
        """
        capex_df = capex_ds.to_frame()
        capex_df.columns = [columns[0]]
        capex_df = capex_df.reindex(columns, axis=1).fillna(0)
        return capex_df

    folder_detailed = os.path.join(folder_output, 'detailed')
    os.mkdir(folder_detailed)

    folder_pkl = os.path.join(folder_detailed, 'pkl')
    os.mkdir(folder_pkl)

    var_dict = {'consumption_actual': buildings.consumption_actual,
                'consumption_conventional': buildings.consumption_conventional,
                'consumption_final': buildings.consumption_final,
                'flow_renovation_label_energy_dict': buildings.flow_renovation_label_energy_dict,
                'energy_cash_flows': buildings.energy_cash_flows,
                'energy_lcc': buildings.energy_lcc,
                'energy_lcc_final': buildings.energy_lcc_final,
                'policies_detailed': buildings.policies_detailed,
                'policies_total': buildings.policies_total,
                'capex_total': buildings.capex_total,
                'energy_saving': buildings.energy_saving,
                'emission_saving': buildings.emission_saving,
                'energy_saving_lc': buildings.energy_saving_lc,
                'emission_saving_lc': buildings.emission_saving_lc
                }

    # TODO: transfrom with dict_pd2df before pkl

    parsed_dict = parse_dict(var_dict)

    # pkl all var_dict
    for key, item in parsed_dict.items():
        name_file = os.path.join(folder_pkl, '{}.pkl'.format(key))
        with open(name_file, 'wb') as file:
            pickle.dump(item, file)

    flow_renovation = dict_pd2df(var_dict['flow_renovation_label_energy_dict'])
    capex_ep = dict_pd2df(var_dict['capex_total'][('Energy performance',)].copy())
    capex_he = dict_pd2df(var_dict['capex_total'][('Heating energy',)].copy())
    subsidies_ep = dict_pd2df(var_dict['policies_total'][('Energy performance',)].copy())
    # subsidies_he = dict_pd2df(var_dict['policies_total'][('Heating energy',)].copy())

    consumption_actual = var_dict['consumption_actual']
    consumption_conventional = var_dict['consumption_conventional']

    energy_cash_flows_actual = var_dict['energy_cash_flows']['actual']
    energy_cash_flows_conventional = var_dict['energy_cash_flows']['conventional']

    # horizon and discount rate --> decision-maker // energy_lcc depends on transition by the invest horizon
    energy_lcc = dict_pd2df(var_dict['energy_lcc'][('Energy performance',)]['conventional']).copy()

    idx_full = flow_renovation.index
    transition = ['Energy performance', 'Heating energy']
    consumption_actual_final = HousingStock.initial2final(consumption_actual, idx_full, transition)
    consumption_actual_re = reindex_mi(consumption_actual, idx_full, consumption_actual.index.names)
    energy_saving = consumption_actual_re - consumption_actual_final
    energy_saving_disc = HousingStock.to_discounted(energy_saving, 0.04)

    emission_initial = HousingStock.mul_consumption(consumption_actual_re, co2_content_data)
    emission_final = HousingStock.mul_consumption(consumption_actual_final, co2_content_data, option='final')
    emission_saving = emission_initial - emission_final
    emission_saving_disc = HousingStock.to_discounted(emission_saving, 0.04)

    energy_lcc_final = HousingStock.initial2final(energy_lcc, idx_full, transition)
    energy_lcc_re = reindex_mi(energy_lcc, idx_full, energy_lcc.index.names)
    energy_lcc_saving = energy_lcc_re - energy_lcc_final

    capex_ep_re = reindex_mi(capex_ep, flow_renovation.index, capex_ep.index.names)
    capex_he_re = reindex_mi(capex_he, flow_renovation.index, capex_he.index.names)
    if subsidies_ep is not None:
        subsidies_ep_re = reindex_mi(subsidies_ep, flow_renovation.index, subsidies_ep.index.names)
        # subsidies_he_re = reindex_mi(subsidies_he, flow_renovation.index, subsidies_he.index.names)
        to_policies_detailed(buildings, folder_detailed)
    else:
        subsidies_ep_re = capex_ep_re * 0

    # total consumption and emission of buildings
    consumption = pd.DataFrame(buildings.stock_seg_dict) * buildings.consumption_actual
    consumption = ds_mul_df(buildings.to_area(), consumption)
    consumption_construction = pd.DataFrame(
        buildings_constructed.stock_constructed_seg_dict) * buildings_constructed.to_consumption_new_stock(
        energy_prices_dict['energy_price_forecast'])
    consumption_construction = ds_mul_df(buildings_constructed.to_area(segments=consumption_construction.index),
                                         consumption_construction)
    consumption_total = consumption.sum(axis=0) + consumption_construction.sum(axis=0)

    if 'total_taxes' in output.keys():
        taxes = HousingStock.mul_consumption(consumption, output['total_taxes'])
        taxes_construction = HousingStock.mul_consumption(consumption_construction, output['total_taxes'])
    else:
        taxes = consumption * 0
        taxes_construction = consumption_construction * 0
    taxes_total = taxes.sum(axis=0) + taxes_construction.sum(axis=0)

    emission = HousingStock.mul_consumption(consumption, co2_content_data)
    emission_construction = HousingStock.mul_consumption(consumption_construction, co2_content_data)
    emission_total = emission.sum(axis=0) + emission_construction.sum(axis=0)
    financials_dict = dict()
    list_years = flow_renovation.columns
    for year in list_years:
        horizon = 30
        yrs = range(year, year + horizon, 1)

        # gCO2 --> tCO2
        emission_saving_yr = HousingStock.to_summed(emission_saving_disc, year, horizon)
        # kWh --> MWh
        energy_saving_yr = HousingStock.to_summed(energy_saving_disc, year, horizon)
        financials_unit = pd.concat((flow_renovation.loc[:, year],
                                     energy_lcc_re.loc[:, year],
                                     energy_lcc_final.loc[:, year],
                                     energy_lcc_saving.loc[:, year],
                                     capex_ep_re.loc[:, year],
                                     subsidies_ep_re.loc[:, year],
                                     capex_he_re.loc[:, year],
                                     emission_saving_yr,
                                     energy_saving_yr), axis=1)
        financials_unit.columns = ['Flow renovation',
                                   'LCC energy initial',
                                   'LCC energy final',
                                   'LCC energy saving',
                                   'Capex envelope',
                                   'Subsidies envelope',
                                   'Capex switch-fuel',
                                   'Emission saving',
                                   'Energy saving']
        financials_unit['LCC energy saving'] = financials_unit['LCC energy initial'] - financials_unit[
            'LCC energy final']
        financials_unit['Total capex'] = financials_unit['Capex envelope'] + financials_unit['Capex switch-fuel'] - \
                                         financials_unit['Subsidies envelope']
        financials_unit['Total subsidies'] = financials_unit['Subsidies envelope']
        financials_unit['NPV'] = financials_unit['LCC energy saving'] - financials_unit['Total capex']
        financials_unit['Investment macro'] = financials_unit['Total capex'] * financials_unit['Flow renovation']
        financials_unit['Subsidies macro'] = financials_unit['Subsidies envelope'] * financials_unit['Flow renovation']
        financials_unit['Private investment macro'] = financials_unit['Investment macro'] - financials_unit[
            'Subsidies macro']
        financials_unit['CO2 investment cost'] = (financials_unit['Investment macro'] / financials_unit[
            'Emission saving'])
        financials_unit['CO2 private cost'] = (financials_unit['NPV'] / financials_unit['Emission saving'])

        area = buildings.to_area()
        area_reindex = reindex_mi(area, flow_renovation.index, area.index.names)
        financials_euro = ds_mul_df(area_reindex, financials_unit)
        financials_euro['Flow renovation'] = financials_unit['Flow renovation']

        # 'G and F buildings': flow_renovation.loc[:, year].
        # percentage of parc at least B
        # number of energy percariy
        financials_dict[year] = {'Consumption': consumption.loc[:, year].sum(),
                                 'Consumption construction': consumption_construction.loc[:, year].sum(),
                                 'Consumption total': consumption_total.loc[year],
                                 'Emission': emission.loc[:, year].sum(),
                                 'Emission construction': emission_construction.loc[:, year].sum(),
                                 'Emission total': emission_total.loc[year],
                                 'Total number of renovations': flow_renovation.loc[:, year].sum(),
                                 'Number of G & F buildings':
                                     buildings.stock_seg_dict[year].groupby('Energy performance').sum().loc[
                                         ['G', 'F']].sum(),
                                 'Investment macro': financials_euro['Investment macro'].sum(),
                                 'Private investment macro': financials_euro['Private investment macro'].sum(),
                                 'Subsidies macro': financials_euro['Subsidies macro'].sum(),
                                 'Energy taxes': taxes.loc[:, year].sum(),
                                 'Energy taxes construction': taxes_construction.loc[:, year].sum(),
                                 'Energy taxes total': taxes_total.loc[year],
                                 'Emission saving': (emission_saving_yr * area_reindex).sum(),
                                 'Energy saving': (energy_saving_yr * area_reindex).sum()
                                 }
        energy_cash_flow = energy_cash_flows_conventional.loc[:, yrs]
        energy_cash_flow_final = HousingStock.initial2final(energy_cash_flow, idx_full, transition)
        energy_cash_flow_re = reindex_mi(energy_cash_flow, idx_full, energy_cash_flow.index.names)
        energy_cash_flow_saving = energy_cash_flow_re - energy_cash_flow_final

        total_capex = capex2cash_flows(financials_unit['Total capex'], energy_cash_flow_saving.columns)
        total_subsidies = capex2cash_flows(financials_unit['Total subsidies'], energy_cash_flow_saving.columns)
        cash_flow = energy_cash_flow_saving - total_capex - total_subsidies

        financials_euro.to_csv(os.path.join(folder_detailed, 'result_euro_{}.csv'.format(year)))
        financials_unit.to_csv(os.path.join(folder_detailed, 'result_unit_{}.csv'.format(year)))

    pd.DataFrame(financials_dict).to_csv(os.path.join(folder_output, 'financials_dict.csv'))


def parse_output(output, buildings, buildings_constructed, folder_output, logging):
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

    logging.debug('Parsing output')
    # output['Stock knowledge energy performance'] = pd.DataFrame(buildings._stock_knowledge_ep_dict)
    # output['Stock knowledge construction'] = pd.DataFrame(buildings_constructed._stock_knowledge_construction_dict)

    # 1. Renovation
    # 1.1 Stock
    output_stock = dict()
    output_stock['Stock'] = pd.DataFrame(buildings._stock_seg_dict)
    output_stock['Stock (m2)'] = (output_stock['Stock'].T * buildings.area).T
    output_stock['Consumption conventional (kWh/m2)'] = buildings.consumption_conventional
    output_stock['Consumption conventional (kWh)'] = (buildings.consumption_conventional * output_stock['Stock (m2)'].T).T
    output_stock['Consumption actual (kWh/m2)'] = buildings.consumption_actual
    output_stock['Consumption actual (kWh)'] = buildings.consumption_actual * output_stock['Stock (m2)']
    output_stock['Budget share (%)'] = buildings.budget_share
    output_stock['Use intensity (%)'] = buildings.heating_intensity
    output_stock['Emission (gCO2/m2)'] = HousingStock.mul_consumption(output_stock['Consumption actual (kWh/m2)'], co2_content_data)
    output_stock['Emission (gCO2)'] = HousingStock.mul_consumption(output_stock['Consumption actual (kWh)'], co2_content_data)

    if 'total_taxes' in output.keys():
        output_stock['Taxes cost (€/m2)'] = HousingStock.mul_consumption(buildings.consumption_actual, output['total_taxes'])
    else:
        output_stock['Taxes cost (€/m2)'] = buildings.consumption_actual * 0
    output_stock['Taxes cost (€)'] = output_stock['Taxes cost (€/m2)'] * output_stock['Stock (m2)']

    output_stock['NPV (€/m2'] = dict_pd2df(buildings.npv[('Energy performance',)])
    output_stock['Renovation rate (%)'] = dict_pd2df(buildings.renovation_rate_dict[('Energy performance',)])

    # 1.2 Transitions
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
    summary['Consumption conventional renovation (kWh)'] = output_stock['Consumption conventional (kWh)'].sum(axis=0)
    summary['Consumption actual renovation (kWh)'] = output_stock['Consumption actual (kWh)'].sum(axis=0)
    summary['Emission (gCO2)'] = output_stock['Emission (gCO2)'].sum(axis=0)
    summary['Use intensity renovation (%)'] = (output_stock['Use intensity (%)'] * output_stock['Stock']).sum(axis=0) / output_stock['Stock'].sum(axis=0)
    summary['Renovation rate renovation (%)'] = (output_stock['Renovation rate (%)'] * output_stock['Stock']).sum(axis=0) / output_stock['Stock'].sum(axis=0)

    summary['Flow transition renovation'] = output_flow_transition['Flow transition'].sum(axis=0)
    summary['Capex renovation (€)'] = output_flow_transition['Capex (€)'].sum(axis=0)
    summary['Subsidies renovation (€)'] = output_flow_transition['Subsidies (€)'].sum(axis=0)

    summary = pd.DataFrame(summary)
    summary.dropna(axis=0, thresh=2, inplace=True)
    # 2. Construction
    # 2.1 Stock
    output_cons_stock = dict()
    output_cons_stock['Stock'] = pd.DataFrame(buildings_constructed.stock_constructed_seg_dict)
    output_cons_stock['Stock (m2)'] = (output_cons_stock['Stock'].T * buildings_constructed.to_area(segments=output_cons_stock['Stock'].index)).T
    # output_cons_stock['Consumption conventional (kWh/m2)']
    # output_cons_stock['Consumption conventional (kWh)']
    output_cons_stock['Consumption actual (kWh/m2)'] = buildings_constructed.to_consumption_new_stock(
        energy_prices_dict['energy_price_forecast'])
    output_cons_stock['Consumption actual (kWh)'] = output_cons_stock['Consumption actual (kWh/m2)'] * output_cons_stock['Stock (m2)']
    output_cons_stock['Emission (gCO2/m2)'] = HousingStock.mul_consumption(
        output_cons_stock['Consumption actual (kWh/m2)'], co2_content_data)
    output_cons_stock['Emission (gCO2)'] = HousingStock.mul_consumption(output_cons_stock['Consumption actual (kWh)'],
                                                                        co2_content_data)



    folder_csv = os.path.join(folder_output, 'output_csv')
    os.mkdir(folder_csv)
    folder_pkl = os.path.join(folder_output, 'output_pkl')
    os.mkdir(folder_pkl)

    new_output = parse_dict(output)
    for key in new_output.keys():
        name_file = os.path.join(folder_csv, '{}.csv'.format(key.lower().replace(' ', '_')))
        logging.debug('Output to csv: {}'.format(name_file))
        new_output[key].to_csv(name_file, header=True)
        name_file = os.path.join(folder_pkl, '{}.pkl'.format(key.lower().replace(' ', '_')))
        new_output[key].to_pickle(name_file)

