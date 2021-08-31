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

from buildings import HousingStock, HousingStockRenovated, HousingStockConstructed
from policies import EnergyTaxes, Subsidies, RegulatedLoan, RenovationObligation
from parse_output import parse_output, quick_graphs


def res_irf(calibration_year, end_year, folder, config, parameters, policies_parameters, attributes, energy_prices_bp,
            energy_taxes, cost_invest, cost_invest_construction, stock_ini, co2_content,
            rate_renovation_ini, ms_renovation_ini, ms_construction_ini, logging):
    """Res-IRF main function.

    Parameters
    ----------
    calibration_year: int
    folder: dict
        Path to a scenario-specific folder used to store all outputs.
    end_year: int
    config: dict
        Dictionary with all scenarios configurations parameters.
    parameters: dict
        Dictionary with parameters.
    policies_parameters: dict
        Dictionary with all policies parameters.
    attributes: dict
        Specific dictionary setting up numerical values for each stock attribute.
        Attributes also contain a list of each attribute.
    energy_prices_bp: pd.DataFrame
        After VTA and other energy taxes but before any endogenous energy taxes.
    energy_taxes: pd.DataFrame
    cost_invest: dict
    cost_invest_construction: dict
    stock_ini: pd.Series
    co2_content: pd.DataFrame
    rate_renovation_ini: pd.Series
    ms_renovation_ini: pd.DataFrame
    ms_construction_ini: pd.DataFrame
    """

    start = time.time()

    logging.debug('Creation of output folder: {}'.format(folder['output']))

    # copyfile(os.path.join(folder['input'], scenario_file), os.path.join(folder['output'], scenario_file))
    pd.Series(config).to_csv(os.path.join(folder['output'], 'scenario.csv'))
    copyfile(os.path.join(folder['input'], 'parameters.json'), os.path.join(folder['output'], 'parameters.json'))

    output = dict()
    logging.debug('Loading in output_dict all input needed for post-script analysis')

    output['Population total'] = parameters['Population total']
    output['Population'] = parameters['Population']
    output['Population housing'] = parameters['Population housing']
    output['Stock needed'] = parameters['Stock needed']
    output['Cost envelope'] = dict()
    output['Cost envelope'][calibration_year] = cost_invest['Energy performance']
    output['Cost construction'] = dict()
    output['Cost construction'][calibration_year] = cost_invest_construction['Energy performance']

    logging.debug('Initialization')

    logging.debug('Initialize public policies')
    subsidies_dict = {}
    energy_taxes_dict = {}
    renovation_obligation_dict = {}
    for pol, item in policies_parameters.items():
        if config[item['name']]['activated']:
            logging.debug('Considering: {}'.format(pol))
            if item['policy'] == 'subsidies':
                subsidies_dict[pol] = Subsidies(item['name'], config[item['name']]['start'],
                                                config[item['name']]['end'], item['kind'], item['value'],
                                                transition=item['transition'],
                                                calibration=config[item['name']]['calibration'])
            elif item['policy'] == 'energy_taxes':
                energy_taxes_dict[pol] = EnergyTaxes(item['name'], config[item['name']]['start'],
                                                     config[item['name']]['end'], item['kind'],
                                                     item['value'],
                                                     calibration=config[item['name']]['calibration'])

            elif item['policy'] == 'regulated_loan':
                subsidies_dict[pol] = RegulatedLoan(item['name'], config[item['name']]['start'],
                                                    config[item['name']]['end'],
                                                    ir_regulated=item['ir_regulated'], ir_market=item['ir_market'],
                                                    principal_min=item['principal_min'],
                                                    principal_max=item['principal_max'],
                                                    horizon=item['horizon'], targets=item['targets'],
                                                    transition=item['transition'],
                                                    calibration=config[item['name']]['calibration'])
                subsidies_dict[pol].reindex_attributes(stock_ini.index)

            elif item['policy'] == 'renovation_obligation':
                renovation_obligation_dict[pol] = RenovationObligation(item['name'], item['start_targets'],
                                                                       participation_rate=item['participation_rate'],
                                                                       columns=range(calibration_year, 2081, 1))

    policies = list(subsidies_dict.values())
    energy_taxes_detailed = dict()
    energy_taxes_detailed['energy_taxes'] = energy_taxes
    total_taxes = None
    for _, tax in energy_taxes_dict.items():
        val = tax.price_to_taxes(energy_prices=energy_prices_bp, co2_content=co2_content)
        # if not indexed by heating energy
        if isinstance(val, pd.Series):
            val = pd.concat([val] * len(attributes['housing_stock_renovated']['Heating energy']), axis=1).T
            val.index = attributes['housing_stock_renovated']['Heating energy']
            val.index.set_names(['Heating energy'], inplace=True)

        if total_taxes is None:
            total_taxes = val
        else:
            total_taxes = total_taxes + val

        energy_taxes_detailed[tax.name] = val

    if total_taxes is not None:
        temp = total_taxes.reindex(energy_prices_bp.columns, axis=1).fillna(0)
        energy_prices = energy_prices_bp + temp
        energy_taxes = energy_taxes + temp
    else:
        energy_prices = energy_prices_bp

    logging.debug('Creating HousingStockRenovated Python object')
    buildings = HousingStockRenovated(stock_ini, attributes['housing_stock_renovated'], calibration_year,
                                      attributes2area=attributes['attributes2area'],
                                      attributes2horizon=attributes['attributes2horizon'],
                                      attributes2discount=attributes['attributes2discount'],
                                      attributes2income=attributes['attributes2income'],
                                      attributes2consumption=attributes['attributes2consumption'],
                                      residual_rate=parameters['Residual destruction rate'],
                                      destruction_rate=parameters['Destruction rate'],
                                      rate_renovation_ini=rate_renovation_ini,
                                      learning_year=parameters['Learning years renovation'],
                                      npv_min=parameters['NPV min'],
                                      rate_max=parameters['Renovation rate max'],
                                      rate_min=parameters['Renovation rate min'])

    logging.debug('Initialize energy consumption and cash-flows')
    buildings.ini_energy_cash_flows(energy_prices)
    io_share_seg = buildings.to_io_share_seg()
    stock_area_existing_seg = buildings.stock_area_seg

    logging.debug('Creating HousingStockConstructed Python object')
    segments_construction = pd.MultiIndex.from_tuples(list(product(*[v for _, v in attributes['housing_stock_constructed'].items()])))
    segments_construction.names = [k for k in attributes['housing_stock_constructed'].keys()]
    buildings_constructed = HousingStockConstructed(pd.Series(0, dtype='float64', index=segments_construction),
                                                    attributes['housing_stock_constructed'], calibration_year,
                                                    parameters['Stock needed'],
                                                    param_share_multi_family=parameters['Factor share multi-family'],
                                                    os_share_ht=parameters['Occupancy status share housing type'],
                                                    io_share_seg=io_share_seg,
                                                    stock_area_existing_seg=stock_area_existing_seg,
                                                    attributes2area=attributes['attributes2area_construction'],
                                                    attributes2horizon=attributes['attributes2horizon_construction'],
                                                    attributes2discount=attributes['attributes2discount_construction'],
                                                    attributes2income=attributes['attributes2income'],
                                                    attributes2consumption=attributes['attributes2consumption_construction'])

    cost_intangible_construction = None
    cost_intangible = None
    policies_calibration = [policy for policy in policies if policy.calibration is True]

    if config['cost_intangible']:
        cost_intangible = dict()
        cost_intangible_construction = dict()
        logging.debug('Calibration market share construction --> intangible cost construction')
        name_file = config['cost_intangible_construction_source']['source']
        source = config['cost_intangible_construction_source']['source_type']
        if source == 'function':
            cost_intangible_construction['Energy performance'] = buildings_constructed.to_calibration_market_share(
                energy_prices,
                ms_construction_ini,
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
        name_file = config['cost_intangible_source']['source']
        source = config['cost_intangible_source']['source_type']
        if source == 'function':

            cost_intangible['Energy performance'] = buildings.to_calibration_market_share(energy_prices,
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

    name_file = config['rho']['source']
    source = config['rho']['source_type']
    if source == 'function':
        rho = buildings.calibration_renovation_rate(energy_prices, rate_renovation_ini,
                                                    cost_invest=cost_invest,
                                                    cost_intangible=cost_intangible,
                                                    policies=policies_calibration)
        logging.debug('End of calibration and dumping: {}'.format(name_file))
        rho.to_pickle(name_file)
    elif source == 'file':
        logging.debug('Loading intangible_cost from {}'.format(name_file))
        rho = pd.read_pickle(name_file)
    else:
        rho = None

    buildings.rho = rho

    years = range(calibration_year, end_year, 1)
    logging.debug('Launching iterations')
    for year in years[1:]:
        logging.debug('YEAR: {}'.format(year))

        policies_year = [policy for policy in policies if policy.start <= year < policy.end]
        buildings.year = year

        # logging.debug('Calculate energy consumption actual')
        # buildings.to_consumption_actual(energy_price)

        logging.debug('Demolition dynamic')
        flow_demolition_seg = buildings.to_flow_demolition_seg()
        logging.debug('Demolition: {:,.0f} buildings, i.e.: {:.2f}%'.format(flow_demolition_seg.sum(),
                                                                            flow_demolition_seg.sum() / buildings.stock_seg.sum() * 100))

        logging.debug('Update demolition')
        buildings.add_flow(- flow_demolition_seg)

        logging.debug('Renovation dynamic')
        renovation_obligation = None
        if 'renovation_obligation' in renovation_obligation_dict:
            renovation_obligation = renovation_obligation_dict['renovation_obligation']
        flow_remained_seg, flow_area_renovation_seg = buildings.to_flow_remained(energy_prices,
                                                                                 consumption='conventional',
                                                                                 cost_invest=cost_invest,
                                                                                 cost_intangible=cost_intangible,
                                                                                 policies=policies_year,
                                                                                 renovation_obligation=renovation_obligation,
                                                                                 mutation=parameters['Mutation rate'],
                                                                                 rotation=parameters['Rotation rate']
                                                                                 )

        logging.debug('Updating stock segmented and renovation knowledge after renovation')
        buildings.update_stock(flow_remained_seg, flow_area_renovation_seg=flow_area_renovation_seg)

        if config['info_renovation']:
            logging.debug('Information acceleration - renovation')
            cost_intangible['Energy performance'] = HousingStock.acceleration_information(buildings.knowledge,
                                                                                          output['Cost intangible'][
                                                                                              buildings.calibration_year],
                                                                                          parameters[
                                                                                              'Information rate max renovation'],
                                                                                          parameters[
                                                                                              'Learning information rate renovation'])
            output['Cost intangible'][year] = cost_intangible['Energy performance']

        if config['lbd_renovation']:
            logging.debug('Learning by doing - renovation')
            cost_invest['Energy performance'] = HousingStock.learning_by_doing(buildings.knowledge,
                                                                               output['Cost envelope'][
                                                                                   buildings.calibration_year],
                                                                               parameters[
                                                                                   'Learning by doing renovation'])
            output['Cost envelope'][year] = cost_invest['Energy performance']

        logging.debug('Construction dynamic')
        buildings_constructed.year = year
        flow_constructed = parameters['Stock needed'].loc[
                               year] - buildings.stock_seg.sum() - buildings_constructed.stock_seg.sum()
        logging.debug('Construction of: {:,.0f} buildings'.format(flow_constructed))
        buildings_constructed.flow_constructed = flow_constructed

        logging.debug('Updating attributes2area_construction')
        buildings_constructed.update_area_construction(parameters['Elasticity area construction'],
                                                       parameters['Available income real population'],
                                                       parameters['Area max construction'])
        logging.debug('Updating flow_constructed segmented')
        # update_flow_constructed_seg will automatically update area constructed and so construction knowledge
        buildings_constructed.update_flow_constructed_seg(energy_prices,
                                                          cost_intangible=cost_intangible_construction,
                                                          cost_invest=cost_invest_construction,
                                                          nu=parameters['Nu construction'],
                                                          policies=None)

        if config['info_construction']:
            logging.debug('Information acceleration - construction')
            cost_intangible_construction['Energy performance'] = HousingStock.acceleration_information(
                buildings_constructed.knowledge,
                output['Cost intangible construction'][buildings_constructed.calibration_year],
                parameters['Information rate max construction'],
                parameters['Learning information rate construction'])
            output['Cost intangible construction'][year] = cost_intangible_construction['Energy performance']

        if config['lbd_construction']:
            logging.debug('Learning by doing - construction')
            cost_invest_construction['Energy performance'] = HousingStock.learning_by_doing(
                buildings_constructed.knowledge,
                output['Cost construction'][buildings_constructed.calibration_year],
                parameters['Learning by doing renovation'],
                cost_lim=parameters['Cost construction lim'])
            output['Cost construction'][year] = cost_invest_construction['Energy performance']

        logging.debug(
            '\nSummary:\nYear: {}\nStock after demolition: {:,.0f}\nDemolition: {:,.0f}\nNeeded: {:,.0f}\nRenovation: {:,.0f}\nConstruction: {:,.0f}'.format(
                year, buildings.stock_seg.sum(), flow_demolition_seg.sum(), parameters['Stock needed'].loc[year],
                buildings.flow_renovation_label_energy_dict[year].sum().sum(), flow_constructed))

    parse_output(output, buildings, buildings_constructed, energy_prices, energy_taxes, energy_taxes_detailed,
                 co2_content, parameters['Aggregated consumption coefficient {}'.format(calibration_year)],
                 folder['output'], lbd_output=True)

    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))
    logging.debug('End')


if __name__ == '__main__':

    print(os.getcwd())

    import copy
    from multiprocessing import Process
    import argparse
    import json
    from parse_input import parse_building_stock, parse_input, parse_parameters, parse_observed_data

    folder = dict()
    folder['input'] = os.path.join(os.getcwd(), 'project', 'input')
    folder['output'] = os.path.join(os.getcwd(), 'project', 'output')
    folder['intermediate'] = os.path.join(os.getcwd(), 'project', 'intermediate')

    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year_end', default=False, help='year end')
    parser.add_argument('-n', '--name', default=False, help='name scenarios')

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

    name_file = os.path.join(folder['input'], 'scenarios.json')
    if args.name:
        name_file = os.path.join(folder['input'], args.name)
    with open(name_file) as file:
        scenarios_dict = json.load(file)

    processes_list = []
    for key, config in scenarios_dict.items():

        calibration_year = config['stock_buildings']['year']

        stock_ini, attributes = parse_building_stock(config)
        parameters, summary_param = parse_parameters(folder['input'], config, stock_ini.sum())
        energy_prices, energy_taxes, cost_invest, cost_invest_construction, co2_content, policies_parameters, summary_input = parse_input(
            folder['input'], config)
        rate_renovation_ini, ms_renovation_ini, ms_construction_ini = parse_observed_data(config)

        end_year = config['end']
        if args.year_end:
            end_year = int(args.year_end)

        folder_scenario = copy.copy(folder)
        folder_scenario['output'] = os.path.join(folder['output'], key.replace(' ', '_'))
        os.mkdir(folder_scenario['output'])

        income = attributes['attributes2income'].T
        income.columns = ['Income {} (€)'.format(c) for c in income.columns]
        summary_param = pd.concat((summary_param, income), axis=1)
        pd.concat((summary_input, summary_param), axis=1).T.loc[:, calibration_year:].to_csv(os.path.join(folder_scenario['output'], 'summary_input.csv'))

        processes_list += [Process(target=res_irf,
                                   args=(calibration_year, end_year, folder_scenario, config, parameters, 
                                         policies_parameters, attributes,
                                         energy_prices, energy_taxes, cost_invest, cost_invest_construction,
                                         stock_ini, co2_content,
                                         rate_renovation_ini, ms_renovation_ini, ms_construction_ini, logging))]

    for p in processes_list:
        p.start()
    for p in processes_list:
        p.join()

    quick_graphs(folder['output'])

    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))
