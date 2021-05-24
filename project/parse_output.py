import pandas as pd
import os
import pickle

# TODO: simplify to reduce memory
"""
flow_renovation = flow_renovation[flow_renovation > 1]
flow_renovation.dropna(axis=0, how='all', inplace=True)
"""


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


def parse_output(output, buildings, buildings_constructed, logging, folder_output, detailed_output=True):

    logging.debug('Parsing output')
    output['Stock segmented'] = pd.DataFrame(buildings._stock_seg_dict)
    output['Stock knowledge energy performance'] = pd.DataFrame(buildings._stock_knowledge_ep_dict)
    output['Stock construction segmented'] = pd.DataFrame(buildings_constructed._stock_constructed_seg_dict)
    output['Stock knowledge construction'] = pd.DataFrame(buildings_constructed._stock_knowledge_construction_dict)

    new_output = concat_yearly_dict(output)
    for key in new_output.keys():
        name_file = os.path.join(folder_output, '{}.csv'.format(key.lower().replace(' ', '_')))
        logging.debug('Output to csv: {}'.format(name_file))
        new_output[key].to_csv(name_file, header=True)
        name_file = os.path.join(folder_output, '{}.pkl'.format(key.lower().replace(' ', '_')))
        new_output[key].to_pickle(name_file)

    if detailed_output:
        os.mkdir(os.path.join(folder_output, 'detailed'))

        # very detailed var
        total_var_dict = {'consumption_info_dict': buildings.consumption_info_dict,
                          'energy_lcc_dict': buildings.energy_lcc_dict,
                          'energy_lcc_final_dict': buildings.energy_lcc_final_dict,
                          'lcc_final_dict': buildings.lcc_final_dict,
                          'pv_dict': buildings.pv_dict,
                          'npv_dict': buildings.npv_dict,
                          'market_share_dict': buildings.market_share_dict,
                          'renovation_rate_dict': buildings.renovation_rate_dict,
                          'flow_renovation_label_dict': buildings.flow_renovation_label_dict,
                          'flow_renovation_label_energy_dict': buildings.flow_renovation_label_energy_dict,
                          'flow_remained_dict': buildings.flow_remained_dict,
                          'flow_demolition_dict': buildings.flow_demolition_dict,
                          'subsidies_detailed_dict': buildings.subsidies_detailed_dict,
                          'subsidies_total_dict': buildings.subsidies_total_dict,
                          'capex_total_dict': buildings.capex_total_dict
                          }

        var_dict = {'consumption_info_dict': buildings.consumption_info_dict,
                    'flow_renovation_label_energy_dict': buildings.flow_renovation_label_energy_dict,
                    'energy_lcc_dict': buildings.energy_lcc_dict,
                    'energy_lcc_final_dict': buildings.energy_lcc_final_dict,
                    'subsidies_detailed_dict': buildings.subsidies_detailed_dict,
                    'subsidies_total_dict': buildings.subsidies_total_dict,
                    'capex_total_dict': buildings.capex_total_dict}

        """
        2 types de .pkl dict -->  
        """
        from utils import reindex_mi, ds_mul_df

        # TODO: function that takes dict with year as key and return DataFrame with year as column
        def transition_dict2dataframe(d):
            temp = {yr: df.stack(df.columns.names) for yr, df in d.items()}
            return pd.DataFrame(temp)

        def segment_dict2dataframe(d):
            return pd.DataFrame(d)

        # TODO: check flow_renovation < 0
        flow_renovation = transition_dict2dataframe(var_dict['flow_renovation_label_energy_dict'])
        capex_ep = transition_dict2dataframe(var_dict['capex_total_dict'][('Energy performance',)].copy())
        capex_he = transition_dict2dataframe(var_dict['capex_total_dict'][('Heating energy',)].copy())
        subsidies_ep = transition_dict2dataframe(var_dict['subsidies_total_dict'][('Energy performance',)].copy())
        subsidies_he = transition_dict2dataframe(var_dict['subsidies_total_dict'][('Heating energy',)].copy())

        # TODO: how to discount energy_lcc
        # horizon and discount rate --> decision-maker
        energy_lcc = segment_dict2dataframe(var_dict['energy_lcc_dict'][('Energy performance',)].copy())

        year = 2020
        from buildings import HousingStock
        stock_temp = pd.Series(0, energy_lcc.index)
        buildings_temp = HousingStock(stock_temp,
                                      buildings.levels_values,
                                      year=year,
                                      label2area=buildings.label2area,
                                      label2horizon=buildings.label2horizon,
                                      label2discount=buildings.label2discount,
                                      label2income=buildings.label2income,
                                      label2consumption=buildings.label2consumption)

        from project.parse_input import forecast2myopic, energy_prices_dict
        energy_prices = forecast2myopic(energy_prices_dict['energy_price_forecast'], year)

        energy_lcc_final = buildings_temp.to_energy_lcc_final(energy_prices,
                                                              transition=['Energy performance', 'Heating energy'],
                                                              consumption='conventional')
        energy_lcc_final = energy_lcc_final.stack(energy_lcc_final.columns.names)

        energy_lcc_reindex = reindex_mi(energy_lcc, flow_renovation.index, energy_lcc.index.names)
        energy_lcc_final_reindex = reindex_mi(energy_lcc_final, flow_renovation.index, energy_lcc_final.index.names)
        subsidies_ep_reindex = reindex_mi(subsidies_ep, flow_renovation.index, subsidies_ep.index.names)
        # if not DataFrame empty
        # subsidies_he_reindex = reindex_mi(subsidies_he, flow_renovation.index, subsidies_he.index.names)
        capex_ep_reindex = reindex_mi(capex_ep, flow_renovation.index, capex_ep.index.names)
        capex_he_reindex = reindex_mi(capex_he, flow_renovation.index, capex_he.index.names)

        financials_unit = pd.concat((flow_renovation.loc[:, year],
                                     energy_lcc_reindex.loc[:, year],
                                     energy_lcc_final_reindex,
                                     capex_ep_reindex.loc[:, year],
                                     subsidies_ep_reindex.loc[:, year],
                                     capex_he_reindex.loc[:, year]), axis=1)

        financials_unit.columns = ['Flow renovation',
                                   'LCC energy initial',
                                   'LCC energy final',
                                   'Capex envelope',
                                   'Subsidies envelope',
                                   'Capex switch-fuel']

        financials_unit['LCC energy saving'] = financials_unit['LCC energy initial'] - financials_unit['LCC energy final']
        financials_unit['Total capex'] = financials_unit['Capex envelope'] + financials_unit['Capex switch-fuel'] - financials_unit['Subsidies envelope']
        financials_unit['NPV'] = financials_unit['LCC energy saving'] - financials_unit['Total capex']
        financials_unit['Total investment'] = financials_unit['Total capex'] * financials_unit['Flow renovation']
        financials_unit['Total subsides'] = financials_unit['Subsidies envelope'] * financials_unit['Flow renovation']
        financials_unit['Total private investment'] = financials_unit['Total investment'] - financials_unit['Total subsides']
        area = buildings.to_area()
        area_reindex = reindex_mi(area, flow_renovation.index, area.index.names)

        financials_euro = ds_mul_df(area_reindex, financials_unit)
        financials_euro['Flow renovation'] = financials_unit['Flow renovation']

        from ui_utils import distribution_scatter
        from project.parse_input import colors_dict

        distribution_scatter(financials_unit, 'Total capex', 'Flow renovation', colors_dict,
                             xlabel='Total capex (€/m2)', ylabel='Renovation', level='Energy performance')

        var_construction_dict = {'consumption_info_dict': buildings.consumption_info_dict,
                                 'energy_lcc_dict': buildings_constructed.energy_lcc_dict,
                                 'energy_lcc_final_dict': buildings_constructed.energy_lcc_final_dict,
                                 'lcc_final_dict': buildings_constructed.lcc_final_dict,
                                 'pv_dict': buildings.pv_dict,
                                 'npv_dict': buildings.npv_dict,
                                 'market_share_dict': buildings.market_share_dict,
                                 }

        """consumption = 'conventional'
        if consumption == 'conventional':
            cash_flows = """

        # TODO:
        #  1 --> for conventional IRR and Payback
        #  2 --> for actual IRR and Payback
        #  3 --> energy saving on total investment duration
        #  4 --> emission saving on total investment duration
        #  5 --> CO2 cost
        #  5 --> McKinsey curve

        for key, item in var_dict.items():
            name_file = os.path.join(folder_output, 'detailed', '{}.pkl'.format(key))
            with open(name_file, 'wb') as file:
                pickle.dump(item, file)

    # can be done later
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
    temp.to_pickle(name_file)
