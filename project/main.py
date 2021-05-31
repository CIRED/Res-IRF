import time
import logging
import datetime
from shutil import copyfile
import pandas as pd

from project.buildings import HousingStock, HousingStockRenovated, HousingStockConstructed
from project.policies import EnergyTaxes, Subsidies, RegulatedLoan
from project.parse_input import *
from project.parse_output import parse_output


if __name__ == '__main__':
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

    logging.debug('Creation of output folder: {}'.format(folder['output']))

    copyfile(os.path.join(folder['input'], 'scenario.json'), os.path.join(folder['output'], 'scenario.json'))
    copyfile(os.path.join(folder['input'], 'parameters.json'), os.path.join(folder['output'], 'parameters.json'))

    output = dict()
    logging.debug('Loading in output_dict all input needed for post-script analysis')
    input2output = ['Population total', 'Population', 'Population housing', 'Flow needed']
    for key in input2output:
        output[key] = dict_parameters[key]

    input2output = {'Cost envelope': cost_envelope, 'Cost construction': cost_construction}
    for key, val in input2output.items():
        output[key] = dict()
        output[key][calibration_year] = val

    logging.debug('Initialization')

    # energy_price = forecast2myopic(energy_prices_dict['energy_price_forecast'], calibration_year)
    energy_price = energy_prices_dict['energy_price_forecast']

    logging.debug('Initialize public policies')
    dict_subsidies = {}
    dict_energy_taxes = {}
    total_taxes = None
    for key, item in dict_policies.items():
        if scenario_dict[item['name']]:
            logging.debug('Considering: {}'.format(key))
            if item['policy'] == 'subsidies':
                dict_subsidies[key] = Subsidies(item['name'], item['start'], item['end'], item['kind'], item['value'],
                                                transition=item['transition'])
            elif item['policy'] == 'energy_taxes':
                dict_energy_taxes[key] = EnergyTaxes(item['name'], item['start'], item['end'], item['kind'],
                                                     item['value'])
                temp = dict_energy_taxes[key].price_to_taxes(
                    energy_prices=energy_prices_dict['energy_price_forecast'],
                    co2_content=co2_content_data)
                if total_taxes is None:
                    total_taxes = temp
                else:
                    total_taxes += temp
            elif item['policy'] == 'regulated_loan':
                pass
            elif item['policy'] == 'regulation':
                pass

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

    segments_construction = buildings.to_segments_construction(
        ['Energy performance', 'Heating energy', 'Income class', 'Income class owner'], {})
    io_share_seg = buildings.to_io_share_seg()
    stock_area_existing_seg = buildings.stock_area_seg

    logging.debug('Creating HousingStockConstructed Python object')
    buildings_constructed = HousingStockConstructed(pd.Series(dtype='float64', index=segments_construction),
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

    buildings_constructed.ini_all_indexes(energy_price,
                                          levels=['Occupancy status', 'Housing type', 'Energy performance',
                                                  'Heating energy'])

    if scenario_dict['cost_intangible']:
        logging.debug('Calibration market share construction --> intangible cost construction')
        name_file = sources_dict['cost_intangible_construction']['source']
        source = sources_dict['cost_intangible_construction']['source_type']
        if source == 'function':
            market_share_obj_construction = HousingStockConstructed.to_market_share_objective(
                dict_share['Occupancy status share housing type'],
                dict_share['Heating energy share housing type'],
                dict_share['Housing type share total'],
                dict_share['Energy performance share total construction'])

            cost_intangible_construction = buildings_constructed.to_calibration_market_share(market_share_obj_construction,
                                                                                             energy_price,
                                                                                             cost_construction=cost_construction)
            logging.debug('End of calibration and dumping: {}'.format(name_file))
            cost_intangible_construction.to_pickle(name_file)
        elif source == 'file':
            logging.debug('Loading cost_intangible_construction from {}'.format(name_file))
            cost_intangible_construction = pd.read_pickle(name_file)
        else:
            cost_intangible_construction = None
        output['Cost intangible construction'] = dict()
        output['Cost intangible construction'][calibration_year] = cost_intangible_construction

        logging.debug('Calibration market share >>> intangible cost')
        name_file = sources_dict['cost_intangible']['source']
        source = sources_dict['cost_intangible']['source_type']
        if source == 'function':
            cost_intangible_seg = buildings.calibration_market_share(energy_price, ms_renovation_ini,
                                                                     folder_output=folder['intermediate'],
                                                                     cost_invest=cost_envelope,
                                                                     consumption='conventional')
            logging.debug('End of calibration and dumping: {}'.format(name_file))
            cost_intangible_seg.to_pickle(name_file)
        elif source == 'file':
            logging.debug('Loading intangible_cost from {}'.format(name_file))
            cost_intangible_seg = pd.read_pickle(name_file)
            cost_intangible_seg.columns.set_names('Energy performance final', inplace=True)
        else:
            cost_intangible_seg = None
        output['Cost intangible'] = dict()
        output['Cost intangible'][calibration_year] = cost_intangible_seg

    logging.debug('Calibration renovation rate >>> rho')
    name_file = sources_dict['rho']['source']
    source = sources_dict['rho']['source_type']
    if source == 'function':
        rho_seg = buildings.calibration_renovation_rate(energy_price, rate_renovation_ini,
                                                        cost_invest=cost_envelope,
                                                        cost_intangible=cost_intangible_seg)
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
        flow_remained_seg, flow_area_renovation_seg = buildings.to_flow_remained(energy_price,
                                                                                 consumption='conventional',
                                                                                 cost_switch_fuel=cost_switch_fuel,
                                                                                 cost_invest=cost_envelope,
                                                                                 cost_intangible=cost_intangible_seg,
                                                                                 subsidies=list(dict_subsidies.values()))

        logging.debug('Updating stock segmented and renovation knowledge after renovation')
        buildings.update_stock(flow_remained_seg, flow_area_renovation_seg=flow_area_renovation_seg)

        if scenario_dict['info_renovation']:
            logging.debug('Information acceleration - renovation')
            cost_intangible_seg = HousingStock.acceleration_information(buildings.knowledge,
                                                                        cost_intangible_seg,
                                                                        dict_parameters[
                                                                            'Information rate max renovation'],
                                                                        dict_parameters[
                                                                            'Information rate renovation'])
            output['Cost intangible'][year] = cost_intangible_seg

        if scenario_dict['lbd_renovation']:
            logging.debug('Learning by doing - renovation')
            cost_envelope = HousingStock.learning_by_doing(buildings.knowledge,
                                                           cost_envelope,
                                                           dict_parameters['Learning by doing renovation'])
            output['Cost envelope'][year] = cost_envelope

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
        buildings_constructed.update_flow_constructed_seg(energy_price,
                                                          cost_intangible=cost_intangible_construction,
                                                          cost_construction=cost_construction,
                                                          nu=dict_parameters["Nu construction"])

        if scenario_dict['info_construction']:
            logging.debug('Information acceleration - construction')
            cost_intangible_construction = HousingStock.acceleration_information(buildings_constructed.knowledge,
                                                                                 cost_intangible_construction,
                                                                                 dict_parameters[
                                                                                     "Information rate max construction"],
                                                                                 dict_parameters[
                                                                                     "Information rate construction"])
            output['Cost intangible construction'][year] = cost_intangible_construction

        if scenario_dict['lbd_construction']:
            logging.debug('Learning by doing - construction')
            cost_construction = HousingStock.learning_by_doing(buildings_constructed.knowledge,
                                                               cost_construction,
                                                               dict_parameters["Learning by doing renovation"],
                                                               cost_lim=dict_parameters['Cost construction lim'])
            output['Cost construction'][year] = cost_construction

        logging.debug(
            '\nSummary: \nYear: {}\nStock after demolition: {:,.0f} \nDemolition: {:,.0f} \nNeeded: {:,.0f} \nRenovation: {:,.0f} \nConstruction: {:,.0f}'.format(
                year, buildings.stock_seg.sum(), flow_demolition_seg.sum(), dict_parameters['Flow needed'].loc[year],
                buildings.flow_renovation_label_energy_dict[year].sum().sum(), flow_constructed))

    parse_output(output, buildings, buildings_constructed, logging, folder['output'])

    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))
    logging.debug('End')
