import pandas as pd
import numpy as np
import logging
import os

import time
from itertools import product

from definition import Housing

from input import language_dict, parameters_dict, index_year, folder, exogenous_dict, cost_dict
from function import logistic

# TODO: switch_fuel


def life_cycle_cost(combination, energy_price_df, all_segments):
    """
    Calculate the life cycle cost of an investment for a specific initial combination, and its market share.
    combination: tuple that represents ("Occupancy status", "Housing type", "Energy performance", "Heating energy",
    "Income class", "Income class owner")
    """
    # TODO: surface_series, switch_fuel
    h = Housing(*combination)
    energy_performance_transition = [i for i in language_dict['energy_performance_list'] if i < h.energy_performance]

    life_cycle_cost_data = pd.DataFrame(index=all_segments, columns=language_dict['energy_performance_list'])
    market_share_data = pd.DataFrame(index=all_segments, columns=language_dict['energy_performance_list'])

    for i in energy_performance_transition:
        discounted_energy_price = h.discounted_energy_prices(energy_price_df)
        life_cycle_cost_data.loc[combination, i] = investment_cost_data.loc[h.energy_performance, i] + \
                                                   discounted_energy_price + \
                                                   intangible_cost_data.loc[h.energy_performance, i]

    total_cost = life_cycle_cost_data.loc[combination, :].apply(lambda x: x ** -1).sum()

    for i in energy_performance_transition:
        market_share_data.loc[combination, i] = life_cycle_cost_data.loc[combination, i] ** -1 / total_cost

    return life_cycle_cost_data, market_share_data


def iteration(energy_price_df, all_segments):
    life_cycle_cost_data = pd.DataFrame(index=all_segments, columns=language_dict['energy_performance_list'])
    market_share_data = pd.DataFrame(index=all_segments, columns=language_dict['energy_performance_list'])

    logging.debug('Calculation of life cycle cost and market share for each segment. Total number {:,}'.format(len(all_segments)))
    k = 0
    for combination in all_segments:
        if k % round(len(all_segments)/100) == 0:
            logging.debug('Loading {:.2f} %'.format(k/len(all_segments) * 100))
        lfc, ms = life_cycle_cost(combination, energy_price_df, all_segments)
        life_cycle_cost_data.update(lfc)
        market_share_data.update(ms)
        k += 1

    logging.debug('Calculation of renovation rate for each segment')
    npv = pd.Series(index=all_segments, dtype='float64')
    renovation_rate = pd.Series(index=all_segments, dtype='float64')
    for combination in all_segments:
        discounted_energy_price = Housing(*combination).discounted_energy_prices(energy_price_df)
        npv.loc[combination] = discounted_energy_price - \
                               (life_cycle_cost_data.loc[combination, :] * market_share_data.loc[combination, :]).sum()

        renovation_rate.loc[combination] = logistic(npv.loc[combination] - parameters_dict['npv_min'],
                                                    a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                                                    r=parameters_dict['r'],
                                                    K=parameters_dict['rate_max'])

    return renovation_rate, npv


if __name__ == '__main__':
    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug('Start Res-IRF')

    investment_cost_data = cost_dict['cost_inv']
    intangible_cost_data = pd.DataFrame(0, index=language_dict['energy_performance_list'],
                                        columns=language_dict['energy_performance_list'])

    # stock = pd.Series(index=all_combination_index)

    name_file = os.path.join(folder['middle'], 'parc.pkl')
    logging.debug('Loading parc pickle file {}'.format(name_file))
    dsp = pd.read_pickle(name_file)
    dsp_idx = dsp.index.names
    dsp = dsp.reset_index().replace(language_dict['dict_replace']).set_index(dsp_idx).iloc[:, 0]
    dsp.index = dsp.index.set_names('Income class owner', 'DECILE_PB')
    dsp = dsp.reorder_levels(language_dict['properties_names'])

    # all_segments = pd.MultiIndex.from_tuples(dsp.index.tolist())
    all_segments = dsp.droplevel('Income class owner').index[~dsp.droplevel('Income class owner').index.duplicated(keep='first')].tolist()
    all_segments = [i + ('income_owner', ) for i in all_segments]
    all_segments = all_segments[700:-700]
    all_segments = pd.MultiIndex.from_tuples(all_segments)
    logging.debug('Total number of housing in this study {:,}'.format(dsp.sum()))

    for year in index_year:
        logging.debug('Loading year {}'.format(year))
        # population = exogenous_dict['population_series'].loc[year]
        # national_income = exogenous_dict['national_income_series'].loc[year]
        renovation_rate, npv = iteration(exogenous_dict['energy_price_data'], all_segments)

        logging.debug("Time for one iteration: {} seconds.".format(time.time() - start))

        break

    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
