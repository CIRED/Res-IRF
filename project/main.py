#!/usr/bin/env python

"""
Res-IRF is a multi-agent building stock dynamic microsimulation model.

The Res-IRF model is a tool for **simulating energy consumption for space heating** in the French residential sector.
Its main characteristic is to integrate a detailed description of the energy performance of the dwelling stock with a
rich description of household behaviour. Res-IRF has been developed to improve the behavioural realism that integrated
models of energy demand typically lack.
"""

import os
import argparse
import time
import logging
import datetime
import json
import copy
from multiprocessing import Process

import pandas as pd

from parse_input import parse_building_stock, parse_exogenous_input, parse_parameters, parse_observed_data
from res_irf import res_irf
from parse_output import quick_graphs
from policy_indicators import run_indicators

__author__ = "Louis-Gaëtan Giraudet, Cyril Bourgeois, Frédéric Branger, François Chabrol, David Glotin, Céline Guivarch, Philippe Quirion, Lucas Vivier"
__copyright__ = "Copyright 2007 Free Software Foundation"
__credits__ = ["Louis-Gaëtan Giraudet", "Cyril Bourgeois", "Frédéric Branger", "François Chabrol", "David Glotin",
               "Céline Guivarch", "Philippe Quirion", "Lucas Vivier"]
__license__ = "GPL"
__version__ = "3.0"
__maintainer__ = "Lucas Vivier"
__email__ = "vivier@centre-cired.fr"
__status__ = "Production"


def model_launcher(path=None):
    """Set up folders and run Res-IRF based on config_files.

    Function enables multiprocessing run.

    Parameters
    ----------
    path: path to config_files, default None
    If None, path must be a python argument.

    Returns
    -------
    """
    folder = dict()
    folder['input'] = os.path.join(os.getcwd(), 'project', 'input')
    folder['output'] = os.path.join(os.getcwd(), 'project', 'output')
    if not os.path.isdir(folder['output']):
        os.mkdir(folder['output'])

    folder['intermediate'] = os.path.join(os.getcwd(), 'project', 'input/phebus/intermediate')
    if not os.path.isdir(folder['intermediate']):
        os.mkdir(folder['intermediate'])

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default=False, help='name scenarios')
    parser.add_argument('-o', '--output', default=False, help='detailed output')

    args = parser.parse_args()

    start = time.time()
    folder['output'] = os.path.join(folder['output'], datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
    if not os.path.isdir(folder['output']):
        os.mkdir(folder['output'])

    logging.basicConfig(filename=os.path.join(folder['output'], 'log.txt'),
                        filemode='a',
                        level=logging.DEBUG,
                        format='%(asctime)s - (%(lineno)s) - %(message)s')

    logging.getLogger('matplotlib.font_manager').disabled = True

    root_logger = logging.getLogger("")
    log_formatter = logging.Formatter('%(asctime)s - (%(lineno)s) - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel('DEBUG')
    root_logger.addHandler(console_handler)

    if path is None:
        path = args.name
    name_file = os.path.join(folder['input'], path)

    with open(name_file) as file:
        config_dict = json.load(file)

    if 'config' in config_dict.keys():
        config_runs = config_dict['config']
        del config_dict['config']
    else:
        config_runs = dict()
        config_runs['Policies indicators'] = False

    parameters = None
    processes_list = []
    for key, config in config_dict.items():

        calibration_year = config['stock_buildings']['year']

        stock_ini, attributes = parse_building_stock(config)
        parameters, summary_param = parse_parameters(folder['input'], config, stock_ini.sum())
        energy_prices, energy_taxes, cost_invest, cost_invest_construction, co2_tax, co2_emission, policies_parameters, summary_input, cost_switch_fuel_end = parse_exogenous_input(
            folder['input'], config)
        rate_renovation_ini, ms_renovation_ini, ms_construction_ini, income_tenants_construction = parse_observed_data(config)

        end_year = config['end']

        folder_scenario = copy.copy(folder)
        folder_scenario['output'] = os.path.join(folder['output'], key.replace(' ', '_'))
        os.mkdir(folder_scenario['output'])

        income = attributes['attributes2income'].T
        income.columns = ['Income {} (euro)'.format(c) for c in income.columns]
        summary_param = pd.concat((summary_param, income), axis=1)
        pd.concat((summary_input, summary_param), axis=1).T.loc[:, calibration_year:].to_csv(
            os.path.join(folder_scenario['output'], 'summary_input.csv'))

        processes_list += [Process(target=res_irf,
                                   args=(calibration_year, end_year, folder_scenario, config, parameters, 
                                         policies_parameters, attributes,
                                         energy_prices, energy_taxes, cost_invest, cost_invest_construction,
                                         cost_switch_fuel_end,
                                         stock_ini, co2_tax, co2_emission,
                                         rate_renovation_ini, ms_renovation_ini, ms_construction_ini,
                                         income_tenants_construction, logging, args.output))]

    for p in processes_list:
        p.start()
    for p in processes_list:
        p.join()

    logging.debug('Creating graphs')
    quick_graphs(folder['output'], args.output)

    if config_runs['Policies indicators']:
        logging.debug('Calculating policies indicators')
        CO2_value = pd.read_csv(os.path.join(folder['input'], 'policies', 'CO2_value.csv'), header=None, index_col=[0], squeeze=True)
        run_indicators(config_runs, folder['output'], CO2_value, parameters=parameters)

    # value_CO2 = pd.read_csv(os.path.join(folder['input'], 'value_CO2.csv'), header=None, index_col=[0], squeeze=True)

    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))


if __name__ == '__main__':
    model_launcher()
