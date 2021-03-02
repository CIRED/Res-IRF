import pandas as pd
import numpy as np
import logging
import os
import time
import pickle
from scipy.optimize import fsolve

from input import language_dict, parameters_dict, index_year, folder, exogenous_dict, cost_dict, calibration_dict
from func import *


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug('Start Res-IRF')

    name_file = os.path.join(folder['middle'], 'parameter_dict.pkl')
    logging.debug('Dumping parameter_dict pickle file {}'.format(name_file))
    with open(name_file, 'wb') as file:
        pickle.dump(parameters_dict, file)

    name_file = os.path.join(folder['middle'], 'language_dict.pkl')
    logging.debug('Dumping language_dict pickle file {}'.format(name_file))
    with open(name_file, 'wb') as file:
        pickle.dump(language_dict, file)

    cost_invest_df = cost_dict['cost_inv']
    cost_invest_df.replace({0: float('nan')}, inplace=True)
    cost_switch_fuel_df = cost_dict['cost_switch_fuel']
    intangible_cost_data = pd.DataFrame(0, index=language_dict['energy_performance_list'],
                                        columns=language_dict['energy_performance_list'])

    name_file = os.path.join(folder['middle'], 'parc.pkl')
    logging.debug('Loading parc pickle file {}'.format(name_file))
    dsp = pd.read_pickle(name_file)

    logging.debug('Total number of housing in this study {}'.format(dsp.sum()))

    segments = pd.MultiIndex.from_tuples(dsp.index.tolist())
    logging.debug('Total number of segments in this study {:,}'.format(len(segments)))

    energy_prices_df = exogenous_dict['energy_price_data']

    logging.debug('Calculate life cycle cost for each possible transition')
    discount_factor = discount_factor_func(segments)
    energy_cost_ts_df = energy_cost_func(segments, energy_prices_df)[3]
    energy_lcc_ds = energy_cost_ts_df.sum(axis=1)
    energy_discount_lcc_ds = pd.DataFrame(discount_factor.values * energy_lcc_ds.values, index=segments)
    energy_discount_lcc_ds.index.names = language_dict['properties_names']
    energy_discount_lcc_ds.columns = ['Values']

    intangible_cost = pd.DataFrame(0, index=cost_invest_df.index, columns=cost_invest_df.columns)

    logging.debug('Calculate life cycle cost for each possible transition')
    lcc_df = lcc_func(energy_discount_lcc_ds, cost_invest_df, cost_switch_fuel_df, intangible_cost)
    lcc_grouped_df = lcc_df.groupby(by='Energy performance', axis=0).mean().groupby(by='Energy performance', axis=1).mean()

    calibration = False
    if calibration == 'solver':
        # market_share_test = market_share_func(lcc_ds)
        def func(intangible_cost_np, lcc_ds, factor):
            intangible_cost_ds = pd.Series(intangible_cost_np, index=lcc_ds.index, name=lcc_ds.name)
            market_share_ds = market_share_func(lcc_ds + intangible_cost_ds**1)
            result0 = market_share_ds - calibration_dict['market_share'].loc[market_share_ds.name, market_share_ds.index]
            result1 = lcc_ds.apply(lambda x: x**-8).sum() / (lcc_ds + intangible_cost_ds**1).apply(lambda x: x**-8).sum() - factor
            result0.iloc[-1] = result1
            return result0


        logging.debug('Calibration of intangible cost')
        lcc_ds = lcc_grouped_df.iloc[0]
        lcc_ds = lcc_ds[lcc_ds.index < lcc_ds.name]

        market_share_ini = calibration_dict['market_share'].loc[lcc_ds.name, lcc_ds.index]
        factor = (lcc_ds.sum() * market_share_ini) / lcc_ds.apply(lambda x: x**-8)

        factor - ((market_share_ini * lcc_ds.sum()) / lcc_ds.apply(lambda x: x**-8))

        factor / (market_share_ini * lcc_ds.sum()) - 1 / lcc_ds.apply(lambda x: x**-8)

        (market_share_ini * lcc_ds.sum() / factor).apply(lambda x: x**-1/8) - lcc_ds

        x0 = lcc_ds.to_numpy() * 1
        factor = 0.1
        root, info_dict, ier, message = fsolve(func, x0, args=(lcc_ds, factor), full_output=True)
        logging.debug(message)

        # checking if solution solve the system
        intangible_cost_ds = pd.Series(root, index=lcc_ds.index, name=lcc_ds.name)
        print(lcc_ds.sum() / (lcc_ds + intangible_cost_ds).sum())
        print(market_share_func(lcc_ds + intangible_cost_ds))
        print(calibration_dict['market_share'].loc[lcc_ds.name, :])

    logging.debug('Calculate market share for each possible transition')
    market_share_df = market_share_func(lcc_df)
    market_share_grouped_df = market_share_df.groupby(by='Energy performance', axis=0).mean().groupby(by='Energy performance', axis=1).mean()

    logging.debug('Calculate net present value for each segment')
    pv_df = market_share_df * lcc_df
    # npv_df = pd.DataFrame(energy_discount_lcc_ds.values - pv_df.values, index=pv_df.index, columns=pv_df.columns)

    logging.debug('Calculate renovation rate for each segment')

    def func(ds):
        return logistic(ds - parameters_dict['npv_min'],
                        a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                        r=parameters_dict['r'],
                        K=parameters_dict['rate_max'])


    # renovation_rate_df = npv_df.apply(func)

    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
