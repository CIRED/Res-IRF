import pandas as pd
import os
import pickle
from buildings import HousingStock
from utils import reindex_mi, ds_mul_df, get_levels_values
from project.parse_input import colors_dict, co2_content_data


def concat_yearly_dict(output):
    """
    output = {'key1': {2018: pd.Series(), 2019: pd.Series()}, 'key2': {2019: pd.DataFrame(), 2020: pd.DataFrame()}}
    >>> concat_yearly_dict(output)
    {'key1': pd.DataFrame(), 'key2': pd.DataFrame()}
    """
    new_output = {}
    for key, output_dict in output.items():
        if isinstance(output_dict, dict):
            for yr, val in output_dict.items():
                if isinstance(val, pd.DataFrame):
                    new_item = {yr: itm.stack(itm.columns.names) for yr, itm in output_dict.items()}
                    new_output[key] = pd.DataFrame(new_item)
                elif isinstance(val, pd.Series):
                    new_output[key] = pd.DataFrame(output_dict)
                elif isinstance(val, float):
                    new_output[key] = pd.Series(output[key])
        elif isinstance(output_dict, pd.DataFrame):
            new_output[key] = output_dict
        elif isinstance(output_dict, pd.Series):
            new_output[key] = output_dict
    return new_output


def dict_pd2df(d_pd):
    """
    Function that takes dict with year as key and return DataFrame with year as column
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


def to_detailed_output(buildings, folder_output):

    def capex2cash_flows(capex_ds, columns):
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
                'subsidies_detailed': buildings.subsidies_detailed,
                'subsidies_total': buildings.subsidies_total,
                'capex_total': buildings.capex_total,
                'energy_saving': buildings.energy_saving,
                'emission_saving': buildings.emission_saving,
                'energy_saving_lc': buildings.energy_saving_lc,
                'emission_saving_lc': buildings.emission_saving_lc
                }

    flow_renovation = dict_pd2df(var_dict['flow_renovation_label_energy_dict'])
    capex_ep = dict_pd2df(var_dict['capex_total'][('Energy performance',)].copy())
    capex_he = dict_pd2df(var_dict['capex_total'][('Heating energy',)].copy())
    subsidies_ep = dict_pd2df(var_dict['subsidies_total'][('Energy performance',)].copy())
    # subsidies_he = dict_pd2df(var_dict['subsidies_total'][('Heating energy',)].copy())

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
    subsidies_ep_re = reindex_mi(subsidies_ep, flow_renovation.index, subsidies_ep.index.names)
    # subsidies_he_re = reindex_mi(subsidies_he, flow_renovation.index, subsidies_he.index.names)

    list_years = flow_renovation.columns
    for year in list_years:
        horizon = 30
        yrs = range(year, year + horizon, 1)
        emission_saving_yr = HousingStock.to_summed(emission_saving_disc, year, horizon)
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
        financials_unit['CO2 investment cost'] = financials_unit['Investment macro'] / financials_unit[
            'Emission saving']
        financials_unit['CO2 private cost'] = financials_unit['NPV'] / financials_unit['Emission saving']

        area = buildings.to_area()
        area_reindex = reindex_mi(area, flow_renovation.index, area.index.names)

        financials_euro = ds_mul_df(area_reindex, financials_unit)
        financials_euro['Flow renovation'] = financials_unit['Flow renovation']

        financials_dict = {'Investment macro': financials_euro['Investment macro'].sum(),
                           'Subsidies macro': financials_euro['Subsidies macro'].sum(),
                           'Private investment macro': financials_euro['Private investment macro'].sum(),
                           }

        energy_cash_flow = energy_cash_flows_conventional.loc[:, yrs]
        energy_cash_flow_final = HousingStock.initial2final(energy_cash_flow, idx_full, transition)
        energy_cash_flow_re = reindex_mi(energy_cash_flow, idx_full, energy_cash_flow.index.names)
        energy_cash_flow_saving = energy_cash_flow_re - energy_cash_flow_final

        total_capex = capex2cash_flows(financials_unit['Total capex'], energy_cash_flow_saving.columns)
        total_subsidies = capex2cash_flows(financials_unit['Total subsidies'], energy_cash_flow_saving.columns)
        cash_flow = energy_cash_flow_saving - total_capex - total_subsidies

        financials_euro.to_csv(os.path.join(folder_detailed, 'result_euro_{}'.format(year)))
        financials_unit.to_csv(os.path.join(folder_detailed, 'result_unit_{}'.format(year)))

        for key, item in var_dict.items():
            name_file = os.path.join(folder_output, 'detailed', '{}.pkl'.format(key))
            with open(name_file, 'wb') as file:
                pickle.dump(item, file)


def parse_output(output, buildings, buildings_constructed, logging, folder_output, detailed_output=True):

    logging.debug('Parsing output')
    output['Stock segmented'] = pd.DataFrame(buildings._stock_seg_dict)
    output['Stock knowledge energy performance'] = pd.DataFrame(buildings._stock_knowledge_ep_dict)
    # output['Stock construction segmented'] = pd.DataFrame(buildings_constructed._stock_constructed_seg_dict)
    # output['Stock knowledge construction'] = pd.DataFrame(buildings_constructed._stock_knowledge_construction_dict)

    folder_csv = os.path.join(folder_output, 'output_csv')
    os.mkdir(folder_csv)
    folder_pkl = os.path.join(folder_output, 'output_pkl')
    os.mkdir(folder_pkl)

    new_output = concat_yearly_dict(output)
    for key in new_output.keys():
        name_file = os.path.join(folder_csv, '{}.csv'.format(key.lower().replace(' ', '_')))
        logging.debug('Output to csv: {}'.format(name_file))
        new_output[key].to_csv(name_file, header=True)
        name_file = os.path.join(folder_pkl, '{}.pkl'.format(key.lower().replace(' ', '_')))
        new_output[key].to_pickle(name_file)

    if detailed_output:
        to_detailed_output(buildings, folder_output)

    """"'# can be done later
    def to_grouped(df, level):
        grouped_sum = df.groupby(level).sum()
        summed = grouped_sum.sum()
        summed.name = 'Total'
        result = pd.concat((grouped_sum.T, summed), axis=1).T
        return result

    # can be done later
    levels = ['Energy performance', 'Heating energy']
    val = 'Stock segmented'
    for lvl in levels:
        name_file = os.path.join(folder_output, '{}.csv'.format((val + '_' + lvl).lower().replace(' ', '_')))
        logging.debug('Output to csv: {}'.format(name_file))
        to_grouped(output[val], lvl).to_csv(name_file, header=True)

    keys = ['Population total', 'Population', 'Population housing', 'Flow needed']
    name_file = os.path.join(folder_output, 'demography.csv')
    temp = pd.concat([output[k] for k in keys], axis=1)
    temp.columns = keys
    temp.to_csv(name_file, header=True)
    name_file = os.path.join(folder_output, 'demography.pkl')
    temp.to_pickle(name_file)"""
