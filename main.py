import pandas as pd
import numpy as np
import logging
import os
import time
from scipy.optimize import fsolve

from input import language_dict, parameters_dict, index_year, folder, exogenous_dict, cost_dict, calibration_dict
from function import logistic
from func import *



if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug('Start Res-IRF')

    cost_invest_df = cost_dict['cost_inv']
    cost_invest_df.replace({0: float('nan')}, inplace=True)
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
    energy_discount_lcc_ds.index.names = language_dict['properties_names']
    energy_discount_lcc_ds.columns = ['Values']

    intangible_cost = pd.DataFrame(0, index=cost_invest_df.index, columns=cost_invest_df.columns)

    logging.debug('Calculate life cycle cost for each possible transition')
    lcc_df = lcc_func(energy_discount_lcc_ds, cost_invest_df, cost_switch_fuel_df, intangible_cost)
    lcc_grouped_df = lcc_df.groupby(by='Energy performance', axis=0).mean().groupby(by='Energy performance', axis=1).mean()

    if False:
        # market_share_test = market_share_func(lcc_ds)
        # TODO: add **2 to intangible_cost
        def func(intangible_cost_np, lcc_ds, factor):
            intangible_cost_ds = pd.Series(intangible_cost_np, index=lcc_ds.index, name=lcc_ds.name)
            market_share_ds = market_share_func(lcc_ds + intangible_cost_ds**2)
            result0 = market_share_ds - calibration_dict['market_share'].loc[market_share_ds.name, market_share_ds.index]
            result1 = lcc_ds.sum() / (lcc_ds + intangible_cost_ds**2).sum() - factor
            result0.iloc[-1] = result1
            return result0

        logging.debug('Calibration of intangible cost')
        lcc_ds = lcc_grouped_df.iloc[0]
        lcc_ds = lcc_ds[lcc_ds.index < lcc_ds.name]
        x0 = lcc_ds.to_numpy() * 0
        root, info_dict, ier, message = fsolve(func, x0, args=(lcc_ds, 0.8), full_output=True)
        logging.debug(message)

    intangible_cost_ds = pd.Series(root, index=lcc_ds.index, name=lcc_ds.name)
    # checking if solution solve the system
    total_cost_ds = intangible_cost_ds + lcc_ds

    print(lcc_ds.sum() / (lcc_ds + intangible_cost_ds).sum())
    print(market_share_func(lcc_ds + intangible_cost_ds))
    print(calibration_dict['market_share'].loc[lcc_ds.name, :])

    logging.debug('Calculate market share for each possible transition')
    market_share_df = market_share_func(lcc_df)
    market_share_grouped_df = market_share_df.groupby(by='Energy performance', axis=0).mean().groupby(by='Energy performance', axis=1).mean()

    logging.debug('Calculate net present value for each segment')
    pv_df = market_share_df * lcc_df
    npv_df = pd.DataFrame(energy_discount_lcc_ds.values - pv_df.values, index=pv_df.index, columns=pv_df.columns)

    print('pause')

    """def func(intangible_cost, energy_discount_lcc_ds, factor):
        intangible_cost = pd.Series(intangible_cost, index=cost_transition_df.index)
        _, _, ms_grouped, lcc_grouped_df = market_share_func(energy_discount_lcc_ds, cost_transition_df, cost_switch_fuel_df, intangible_cost)
        result0 = ms_grouped - calibration_dict['market_share']
        result1 = lcc_grouped_df.sum(axis=1) / (lcc_grouped_df + intangible_cost).sum(axis=1) - factor
        result0.iloc[:, -1] = result1
        return result0"""




    # func(intangible_cost, energy_discount_lcc_ds, 0.1)

    # market_share_df_ini = calibration_dict['market_share'].loc[ms_grouped.index, ms_grouped.columns]


    print('pause')

    #


    def func(ds):
        return logistic(ds - parameters_dict['npv_min'],
                        a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                        r=parameters_dict['r'],
                        K=parameters_dict['rate_max'])


    # renovation_rate_df = npv_df.apply(func)

    print('pause')

    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
