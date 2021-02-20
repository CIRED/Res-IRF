import pandas as pd
import numpy as np
import logging
import os

import time
from itertools import product

from definition import Housing

from input import language_dict, parameters_dict, index_year, folder, exogenous_dict, cost_dict, calibration_dict
from function import logistic
from func import *

# TODO: switch_fuel





def renovation_rate_func(data, all_segments, energy_price_df):
    logging.debug('Calculation of renovation rate for each segment')
    npv = pd.Series(index=all_segments, dtype='float64')
    renovation_rate = pd.Series(index=all_segments, dtype='float64')
    for segment in all_segments:
        discounted_energy_price = Housing(*segment).discounted_energy_prices(energy_price_df)
        npv.loc[segment] = discounted_energy_price - data[segment]

        renovation_rate.loc[segment] = logistic(npv.loc[segment] - parameters_dict['npv_min'],
                                                    a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                                                    r=parameters_dict['r'],
                                                    K=parameters_dict['rate_max'])
    return renovation_rate, npv


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug('Start Res-IRF')

    cost_transition_df = cost_dict['cost_inv']
    cost_switch_fuel_df = cost_dict['cost_switch_fuel']
    intangible_cost_data = pd.DataFrame(0, index=language_dict['energy_performance_list'],
                                        columns=language_dict['energy_performance_list'])

    name_file = os.path.join(folder['middle'], 'parc.pkl')
    logging.debug('Loading parc pickle file {}'.format(name_file))
    dsp = pd.read_pickle(name_file)
    dsp_idx = dsp.index.names
    dsp = dsp.reset_index().replace(language_dict['dict_replace']).set_index(dsp_idx).iloc[:, 0]
    dsp.index = dsp.index.set_names('Income class owner', 'DECILE_PB')
    dsp = dsp.reorder_levels(language_dict['properties_names'])

    logging.debug('Total number of housing in this study {:,}'.format(dsp.sum()))

    segments = pd.MultiIndex.from_tuples(dsp.index.tolist())
    """segments = dsp.droplevel('Income class owner').index[~dsp.droplevel('Income class owner').index.duplicated(keep='first')].tolist()
    segments = [i + ('income_owner', ) for i in segments]
    segments = segments[700:-700]
    segments = pd.MultiIndex.from_tuples(segments)"""

    energy_prices_df = exogenous_dict['energy_price_data']

    discount_factor = discount_factor_func(segments)
    energy_cost_df = energy_cost_func(segments, energy_prices_df)[3]
    energy_lcc_ds = energy_cost_df.sum(axis=1)
    energy_discount_lcc_ds = pd.DataFrame(discount_factor.values * energy_lcc_ds.values, index=segments)
    # energy_discount_lcc_ds.index.names = language_dict['properties_names']
    energy_discount_lcc_ds.index.names = language_dict['properties_names']

    energy_discount_lcc_ds.columns = ['Values']

    market_share_df, pv_df = market_share_func(energy_discount_lcc_ds, cost_transition_df, cost_switch_fuel_df)
    npv_df = pd.DataFrame(energy_discount_lcc_ds.values - pv_df.values, index=pv_df.index, columns=pv_df.columns)


    def func(ds):
        return logistic(ds - parameters_dict['npv_min'],
                 a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                 r=parameters_dict['r'],
                 K=parameters_dict['rate_max'])


    renovation_rate_df = npv_df.apply(func)


    print('pause')

    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
