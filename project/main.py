"""Res-IRF script

This script requires that 'pandas' be installed within the Python
environment you are running this script in.
"""
import os
import time
import logging
import datetime
import pandas as pd
from shutil import copyfile
from itertools import product

from project.buildings import HousingStock, HousingStockRenovated, HousingStockConstructed
from project.policies import EnergyTaxes, Subsidies, RegulatedLoan, RenovationObligation
from project.parse_output import parse_output


def res_irf(folder, scenario_dict, dict_parameters, dict_policies, levels_dict_construction, levels_dict,
            energy_prices_dict, cost_invest, cost_invest_construction,
            sources_dict, stock_ini_seg, co2_content_data, dict_label,
            rate_renovation_ini, ms_renovation_ini, dict_share, logging):
    """Res-IRF functions.

    Parameters
    ----------
    folder: dict
        path
    scenario_dict: dict
        dict with all scenarios
    dict_parameters: dict
        dict with all parameters
    dict_policies: dict
        dict with all policies parameters
    levels_dict_construction: dict
    levels_dict: dict
    energy_prices_dict: dict
    cost_invest: dict
    cost_invest_construction: dict
    sources_dict: dict
    stock_ini_seg: pd.Series
    co2_content_data: pd.Series
    dict_label: dict
    rate_renovation_ini: pd.Series
    ms_renovation_ini: pd.DataFrame
    dict_share: dict
    """

    start = time.time()
    calibration_year = sources_dict['stock_buildings']['year']

    logging.debug('Creation of output folder: {}'.format(folder['output']))

    # copyfile(os.path.join(folder['input'], scenario_file), os.path.join(folder['output'], scenario_file))
    pd.Series(scenario_dict).to_csv(os.path.join(folder['output'], 'scenario.csv'))
    copyfile(os.path.join(folder['input'], 'parameters.json'), os.path.join(folder['output'], 'parameters.json'))

    output = dict()
    logging.debug('Loading in output_dict all input needed for post-script analysis')
    input2output = ['Population total', 'Population', 'Population housing', 'Flow needed']
    for key in input2output:
        output[key] = dict_parameters[key]
    input2output = {'Cost envelope': cost_invest['Energy performance'],
                    'Cost construction': cost_invest_construction['Energy performance']}
    for key, val in input2output.items():
        output[key] = dict()
        output[key][calibration_year] = val

    logging.debug('Initialization')

    # function of scenario_dict
    label2horizon = dict()
    dict_label['label2horizon_heater'] = dict_label['label2horizon_heater'][scenario_dict['investor']]
    dict_label['label2horizon_envelope'] = dict_label['label2horizon_envelope'][scenario_dict['investor']]
    label2horizon[('Energy performance', )] = dict_label['label2horizon_envelope']
    label2horizon[('Heating energy', )] = dict_label['label2horizon_heater']
    dict_label['label2horizon'] = label2horizon

    # energy_price = forecast2myopic(energy_prices_dict['energy_price_forecast'], calibration_year)
    energy_price = energy_prices_dict['energy_price_forecast']

    logging.debug('Initialize public policies')
    policies_dict = {}
    dict_energy_taxes = {}
    dict_renovation_obligation = {}
    total_taxes = None
    for pol, item in dict_policies.items():
        if scenario_dict[item['name']]:
            logging.debug('Considering: {}'.format(pol))
            if item['policy'] == 'subsidies':
                policies_dict[pol] = Subsidies(item['name'], item['start'], item['end'], item['kind'], item['value'],
                                               transition=item['transition'])
            elif item['policy'] == 'energy_taxes':
                dict_energy_taxes[pol] = EnergyTaxes(item['name'], item['start'], item['end'], item['kind'],
                                                     item['value'])
                temp = dict_energy_taxes[pol].price_to_taxes(
                    energy_prices=energy_prices_dict['energy_price_forecast'],
                    co2_content=co2_content_data)
                if total_taxes is None:
                    total_taxes = temp
                else:
                    total_taxes += temp
            elif item['policy'] == 'regulated_loan':
                policies_dict[pol] = RegulatedLoan(item['name'], item['start'], item['end'],
                                                   ir_regulated=item['ir_regulated'], ir_market=item['ir_market'],
                                                   principal_min=item['principal_min'],
                                                   principal_max=item['principal_max'],
                                                   horizon=item['horizon'], targets=item['targets'],
                                                   transition=item['transition'])
                policies_dict[pol].reindex_attributes(stock_ini_seg.index)

            elif item['policy'] == 'renovation_obligation':
                dict_renovation_obligation[pol] = RenovationObligation(item['name'], item['start_targets'],
                                                                       item['participation_rate'],
                                                                       columns=index_input_year)

    policies = list(policies_dict.values())

    if total_taxes is not None:
        output['total_taxes'] = total_taxes
        energy_price = energy_price * (1 + total_taxes)

    logging.debug('Creating HousingStockRenovated Python object')
    buildings = HousingStockRenovated(stock_ini_seg, levels_dict, calibration_year,
                                      residual_rate=dict_parameters['Residual destruction rate'],
                                      destruction_rate=dict_parameters['Destruction rate'],
                                      rate_renovation_ini=rate_renovation_ini,
                                      learning_year=dict_parameters['Learning years renovation'],
                                      npv_min=dict_parameters['NPV min'],
                                      rate_max=dict_parameters['Renovation rate max'],
                                      rate_min=dict_parameters['Renovation rate min'],
                                      label2area=dict_label['label2area'],
                                      label2horizon=dict_label['label2horizon'],
                                      label2discount=dict_label['label2discount'],
                                      label2income=dict_label['label2income'],
                                      label2consumption=dict_label['label2consumption'])

    logging.debug('Initialize energy consumption and cash-flows')
    buildings.ini_energy_cash_flows(energy_price)
    io_share_seg = buildings.to_io_share_seg()
    stock_area_existing_seg = buildings.stock_area_seg

    logging.debug('Creating HousingStockConstructed Python object')
    segments_construction = pd.MultiIndex.from_tuples(list(product(*[v for _, v in levels_dict_construction.items()])))
    segments_construction.names = [k for k in levels_dict_construction.keys()]
    buildings_constructed = HousingStockConstructed(pd.Series(0, dtype='float64', index=segments_construction),
                                                    levels_dict_construction, calibration_year,
                                                    dict_parameters['Flow needed'],
                                                    param_share_multi_family=dict_parameters['Factor share multi-family'],
                                                    os_share_ht=dict_share['Occupancy status share housing type'],
                                                    io_share_seg=io_share_seg,
                                                    stock_area_existing_seg=stock_area_existing_seg,
                                                    label2area=dict_label['label2area_construction'],
                                                    label2horizon=dict_label['label2horizon_construction'],
                                                    label2discount=dict_label['label2discount_construction'],
                                                    label2income=dict_label['label2income'],
                                                    label2consumption=dict_label['label2consumption_construction'])

    """buildings_constructed.ini_all_indexes(energy_price,
                                          levels=['Occupancy status', 'Housing type', 'Energy performance',
                                                  'Heating energy'])"""
    cost_intangible_construction = None
    cost_intangible = None
    if scenario_dict['cost_intangible']:
        cost_intangible = dict()
        cost_intangible_construction = dict()
        logging.debug('Calibration market share construction --> intangible cost construction')
        name_file = sources_dict['cost_intangible_construction']['source']
        source = sources_dict['cost_intangible_construction']['source_type']
        if source == 'function':
            market_share_obj_construction = HousingStockConstructed.to_market_share_objective(
                dict_share['Occupancy status share housing type'],
                dict_share['Heating energy share housing type'],
                dict_share['Housing type share total'],
                dict_share['Energy performance share total construction'])
            policies_calibration = [policy for policy in policies if policy.calibration is True]

            cost_intangible_construction['Energy performance'] = buildings_constructed.to_calibration_market_share(
                energy_price,
                market_share_obj_construction,
                cost_invest=cost_invest_construction,
                policies=policies_calibration)
            logging.debug('End of calibration and dumping: {}'.format(name_file))
            cost_intangible_construction['Energy performance'].to_pickle(name_file)
        elif source == 'file':
            logging.debug('Loading cost_intangible_construction from {}'.format(name_file))
            cost_intangible_construction['Energy performance'] = pd.read_pickle(name_file)

        cost_intangible_construction['Heating energy'] = None
        output['Cost intangible construction'] = dict()
        output['Cost intangible construction'][calibration_year] = cost_intangible_construction['Energy performance']

        logging.debug('Calibration market share >>> intangible cost')
        name_file = sources_dict['cost_intangible']['source']
        source = sources_dict['cost_intangible']['source_type']
        if source == 'function':

            policies_calibration = [policy for policy in policies if policy.calibration is True]
            cost_intangible['Energy performance'] = buildings.to_calibration_market_share(energy_price,
                                                                                          ms_renovation_ini,
                                                                                          cost_invest=cost_invest,
                                                                                          consumption='conventional',
                                                                                          policies=policies_calibration)
            logging.debug('End of calibration and dumping: {}'.format(name_file))
            cost_intangible['Energy performance'].to_pickle(name_file)
        elif source == 'file':
            logging.debug('Loading intangible_cost from {}'.format(name_file))
            cost_intangible['Energy performance'] = pd.read_pickle(name_file)
            cost_intangible['Energy performance'].columns.set_names('Energy performance final', inplace=True)

        cost_intangible['Heating energy'] = None
        output['Cost intangible'] = dict()
        output['Cost intangible'][calibration_year] = cost_intangible['Energy performance']

    logging.debug('Calibration renovation rate >>> rho')
    name_file = sources_dict['rho']['source']
    source = sources_dict['rho']['source_type']
    if source == 'function':
        rho_seg = buildings.calibration_renovation_rate(energy_price, rate_renovation_ini,
                                                        cost_invest=cost_invest,
                                                        cost_intangible=cost_intangible,
                                                        policies=policies)
        logging.debug('End of calibration and dumping: {}'.format(name_file))
        rho_seg.to_pickle(name_file)
    elif source == 'file':
        logging.debug('Loading intangible_cost from {}'.format(name_file))
        rho_seg = pd.read_pickle(name_file)
    else:
        rho_seg = None
    buildings.rho_seg = rho_seg

    years = range(calibration_year, dict_parameters['End year'], 1)
    logging.debug('Launching iterations')
    for year in years[1:]:
        logging.debug('YEAR: {}'.format(year))
        policies_year = [policy for policy in policies if policy.start <= year <= policy.end]
        buildings.year = year

        logging.debug('Calculate energy consumption actual')
        buildings.to_consumption_actual(energy_price)

        logging.debug('Demolition dynamic')
        flow_demolition_seg = buildings.to_flow_demolition_seg()
        logging.debug('Demolition: {:,.0f} buildings, i.e.: {:.2f}%'.format(flow_demolition_seg.sum(),
                                                                            flow_demolition_seg.sum() / buildings.stock_seg.sum() * 100))

        logging.debug('Update demolition')
        buildings.add_flow(- flow_demolition_seg)

        logging.debug('Renovation dynamic')
        renovation_obligation = None
        if 'renovation_obligation' in dict_renovation_obligation:
            renovation_obligation = dict_renovation_obligation['renovation_obligation']
        flow_remained_seg, flow_area_renovation_seg = buildings.to_flow_remained(energy_price,
                                                                                 consumption='conventional',
                                                                                 cost_invest=cost_invest,
                                                                                 cost_intangible=cost_intangible,
                                                                                 policies=policies_year,
                                                                                 renovation_obligation=renovation_obligation,
                                                                                 mutation=dict_parameters['Mutation rate'],
                                                                                 rotation=dict_parameters['Rotation rate']
                                                                                 )

        logging.debug('Updating stock segmented and renovation knowledge after renovation')
        buildings.update_stock(flow_remained_seg, flow_area_renovation_seg=flow_area_renovation_seg)

        if scenario_dict['info_renovation']:
            logging.debug('Information acceleration - renovation')
            cost_intangible['Energy performance'] = HousingStock.acceleration_information(buildings.knowledge,
                                                                                          cost_intangible['Energy performance'],
                                                                                          dict_parameters['Information rate max renovation'],
                                                                                          dict_parameters['Learning information rate renovation'])
            output['Cost intangible'][year] = cost_intangible['Energy performance']

        if scenario_dict['lbd_renovation']:
            logging.debug('Learning by doing - renovation')
            cost_invest['Energy performance'] = HousingStock.learning_by_doing(buildings.knowledge,
                                                                               cost_invest['Energy performance'],
                                                                               dict_parameters[
                                                                                   'Learning by doing renovation'])
            output['Cost envelope'][year] = cost_invest['Energy performance']

        logging.debug('Construction dynamic')
        buildings_constructed._year = year
        flow_constructed = dict_parameters['Flow needed'].loc[year] - buildings.stock_seg.sum()
        logging.debug('Construction of: {:,.0f} buildings'.format(flow_constructed))
        buildings_constructed.flow_constructed = flow_constructed

        logging.debug('Updating label2area_construction')
        buildings_constructed.update_area_construction(dict_parameters['Elasticity area construction'],
                                                       dict_parameters['Available income real population'],
                                                       dict_parameters['Area max construction'])
        logging.debug('Updating flow_constructed segmented')
        # update_flow_constructed_seg will automatically update area constructed and so construction knowledge
        buildings_constructed.to_flow_constructed_seg(energy_price,
                                                      cost_intangible=cost_intangible_construction,
                                                      cost_invest=cost_invest_construction,
                                                      nu=dict_parameters['Nu construction'],
                                                      policies=None)

        if scenario_dict['info_construction']:
            logging.debug('Information acceleration - construction')
            cost_intangible_construction['Energy performance'] = HousingStock.acceleration_information(buildings_constructed.knowledge,
                                                                                                       cost_intangible_construction['Energy performance'],
                                                                                                       dict_parameters['Information rate max construction'],
                                                                                                       dict_parameters['Learning information rate construction'])
            output['Cost intangible construction'][year] = cost_intangible_construction['Energy performance']

        if scenario_dict['lbd_construction']:
            logging.debug('Learning by doing - construction')
            cost_invest_construction['Energy performance'] = HousingStock.learning_by_doing(buildings_constructed.knowledge,
                                                                                            cost_invest_construction['Energy performance'],
                                                                                            dict_parameters['Learning by doing renovation'],
                                                                                            cost_lim=dict_parameters['Cost construction lim'])
            output['Cost construction'][year] = cost_invest_construction['Energy performance']

        logging.debug(
            '\nSummary:\nYear: {}\nStock after demolition: {:,.0f}\nDemolition: {:,.0f}\nNeeded: {:,.0f}\nRenovation: {:,.0f}\nConstruction: {:,.0f}'.format(
                year, buildings.stock_seg.sum(), flow_demolition_seg.sum(), dict_parameters['Flow needed'].loc[year],
                buildings.flow_renovation_label_energy_dict[year].sum().sum(), flow_constructed))

    parse_output(output, buildings, buildings_constructed, logging, folder['output'])

    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))
    logging.debug('End')


if __name__ == '__main__':
    import copy
    from project.parse_input import *
    from multiprocessing import Process
    import argparse

    parser = argparse.ArgumentParser()

    multiple_scenario = False

    start = time.time()

    folder['output'] = os.path.join(folder['output'], datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
    if not os.path.isdir(folder['output']):
        os.mkdir(folder['output'])

    logging.basicConfig(filename=os.path.join(folder['output'], 'log.txt'),
                        filemode='a',
                        level=logging.DEBUG,
                        format='%(asctime)s - (%(lineno)s) - %(message)s')
    root_logger = logging.getLogger("")
    log_formatter = logging.Formatter('%(asctime)s - (%(lineno)s) - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel('DEBUG')
    root_logger.addHandler(console_handler)

    if not multiple_scenario:
        name_file = os.path.join(folder['input'], 'scenario.json')
        with open(name_file) as file:
            scenario_dict = json.load(file)
        res_irf(folder, scenario_dict, dict_parameters, dict_policies, levels_dict_construction, levels_dict,
                energy_prices_dict, cost_invest, cost_invest_construction,
                sources_dict, stock_ini_seg, co2_content_data, dict_label,
                rate_renovation_ini, ms_renovation_ini, dict_share, logging)
    else:
        name_file = os.path.join(folder['input'], 'scenarios.json')
        with open(name_file) as file:
            scenarios_dict = json.load(file)

        processes_list = []
        for key, scenario_dict in scenarios_dict.items():

            folder_scenario = copy.copy(folder)
            folder_scenario['output'] = os.path.join(folder['output'], key.replace(' ', '_'))
            os.mkdir(folder_scenario['output'])

            processes_list += [Process(target=res_irf,
                                       args=(folder_scenario, scenario_dict, dict_parameters, dict_policies,
                                             levels_dict_construction, levels_dict,
                                             energy_prices_dict, cost_invest, cost_invest_construction,
                                             sources_dict, stock_ini_seg, co2_content_data, dict_label,
                                             rate_renovation_ini, ms_renovation_ini, dict_share, logging))]

        for p in processes_list:
            p.start()
        for p in processes_list:
            p.join()

    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))
