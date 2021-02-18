import pandas as pd
import numpy as np
import logging
import os

import time
from itertools import product

from definition import Housing

from input import language_dict, parameters_dict, index_year, folder, exogenous_dict
from function import logistic


def life_cycle_cost(combination, life_cycle_cost_data, market_share_data, energy_prices, surface=1):
    """
    Calculate the life cycle cost of an investment for a specific combination, and its market share.
    combination: tuple that represents ("Occupancy status", "Housing type", "Energy performance", "Income class",
    "Heating energy")
    """
    occupancy_status, housing_type, energy_performance, income_class, heating_energy = combination
    h = Housing(occupancy_status, housing_type, energy_performance, heating_energy, income_class)
    energy_performance_transition = [i for i in language_dict['energy_performance_list'] if i < energy_performance]

    for i in energy_performance_transition:
        _, _, _, energy_cost = h.energy_cost(energy_price=energy_prices.loc[heating_energy],
                                             surface=surface,
                                             energy_performance=i)
        life_cycle_cost_data.loc[combination, i] = investment_cost_data.loc[energy_performance, i] + \
                                                   h.discount_factor * energy_cost + \
                                                   intangible_cost_data.loc[energy_performance, i]

    total_cost = life_cycle_cost_data.loc[combination, :].apply(lambda x: x ** -1).sum()

    for i in energy_performance_transition:
        market_share_data.loc[combination, i] = life_cycle_cost_data.loc[combination, i] ** -1 / total_cost

    return life_cycle_cost_data, market_share_data


def iteration(energy_prices):
    life_cycle_cost_data = pd.DataFrame(index=language_dict['all_combination_index'],
                                        columns=language_dict['energy_performance_list'])
    market_share_data = pd.DataFrame(index=language_dict['all_combination_index'],
                                     columns=language_dict['energy_performance_list'])

    for combination in language_dict['all_combination_list']:
        life_cycle_cost_data, market_share_data = life_cycle_cost(combination,
                                                                  life_cycle_cost_data,
                                                                  market_share_data,
                                                                  energy_prices)

    npv = pd.Series(index=language_dict['all_combination_index'], dtype='float64')
    renovation_rate = pd.Series(index=language_dict['all_combination_index'], dtype='float64')

    for combination in language_dict['all_combination_list']:
        npv.loc[combination] = (life_cycle_cost_data.loc[combination, :] * market_share_data.loc[combination, :]).sum()

        renovation_rate.loc[combination] = logistic(npv.loc[combination] - parameters_dict['npv_min'],
                                                    a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                                                    r=parameters_dict['r'],
                                                    K=parameters_dict['rate_max'])

    return renovation_rate


if __name__ == '__main__':
    start = time.time()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    investment_cost_data = pd.DataFrame(100, index=language_dict['energy_performance_list'],
                                        columns=language_dict['energy_performance_list'])
    intangible_cost_data = pd.DataFrame(0, index=language_dict['energy_performance_list'],
                                        columns=language_dict['energy_performance_list'])

    # stock = pd.Series(index=all_combination_index)

    name_file = os.path.join(folder['middle'], 'parc.pkl')
    logging.debug('Loading parc pickle file {}'.format(name_file))
    dsp = pd.read_pickle(name_file)
    logging.debug('Total number of housing in this study {:,}'.format(dsp.sum()))

    for year in index_year:
        logging.debug('Loading year {}'.format(year))
        energy_prices = exogenous_dict['energy_price_data'].loc[year, :]
        population = exogenous_dict['population_series'].loc[year]
        national_income = exogenous_dict['national_income_series'].loc[year]
        renovation_rate = iteration(energy_prices)

        logging.debug("Time for one iteration: {} seconds.".format(time.time() - start))

        break

    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
