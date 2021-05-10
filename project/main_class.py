from buildings_class import HousingStock, HousingStockRenovated, HousingStockConstructed
from project.parse_input import *
import time
import logging


if __name__ == '__main__':

    start = time.time()
    logging.debug('Initialization')
    energy_price = energy_prices_dict['energy_price_myopic']

    logging.debug('Creating HousingStockRenovated Python object')
    buildings = HousingStockRenovated(stock_ini_seg, levels_dict, 2018, 0.035,
                                      label2area=dict_label['label2area'],
                                      label2horizon=dict_label['label2horizon'],
                                      label2discount=dict_label['label2discount'],
                                      label2income=dict_label['label2income'],
                                      label2consumption=dict_label['label2consumption'])

    segments_construction = buildings.to_segments_construction(
        ['Energy performance', 'Heating energy', 'Income class', 'Income class owner'], {})
    io_share_seg = buildings.to_io_share_seg()
    stock_area_existing_seg = buildings.stock_area_seg

    logging.debug('Creating HousingStockConstructed Python object')

    buildings_constructed = HousingStockConstructed(pd.Series(dtype='float64', index=segments_construction),
                                                    levels_dict_construction, 2018,
                                                    pd.Series(dict_parameters['Flow needed']),
                                                    param_share_multi_family=dict_parameters['Factor share multi-family'],
                                                    os_share_ht=dict_result['Occupancy status share housing type'],
                                                    io_share_seg=io_share_seg,
                                                    stock_area_existing_seg=stock_area_existing_seg,
                                                    label2area=dict_label['label2area_construction'],
                                                    label2horizon=dict_label['label2horizon_construction'],
                                                    label2discount=dict_label['label2discount_construction'],
                                                    label2income=dict_label['label2income'],
                                                    label2consumption=dict_label['label2consumption_construction'])

    logging.debug('Calibration market share construction --> intangible cost construction')
    name_file = sources_dict['cost_intangible_construction']['source']
    source = sources_dict['cost_intangible_construction']['source_type']
    if source == 'function':
        market_share_obj_construction = HousingStockConstructed.to_market_share_objective(
            dict_result['Occupancy status share housing type'],
            dict_result['Heating energy share housing type'],
            dict_result['Housing type share total'],
            dict_result['Energy performance share total construction'])

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

    logging.debug('Calibration market share >>> intangible cost')
    name_file = sources_dict['cost_intangible']['source']
    source = sources_dict['cost_intangible']['source_type']
    if source == 'function':
        cost_intangible_seg = buildings.calibration_market_share(energy_price, marker_share_obj,
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

    logging.debug('Calibration renovation rate >>> rho')
    name_file = sources_dict['rho']['source']
    source = sources_dict['rho']['source_type']
    if source == 'function':
        rho_seg = buildings.calibration_renovation_rate(energy_price, renovation_obj,
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

    years = range(2018, 2020, 1)
    logging.debug('Launching iterations')
    for year in years[1:]:
        logging.debug('YEAR: {}'.format(year))

        logging.debug('Demolition and renovation dynamic')
        buildings._year = year

        flow_demolition_seg = buildings.to_flow_demolition()
        flow_remained_seg, flow_area_renovation_seg = buildings.to_flow_remained(energy_price,
                                                                                 consumption='conventional',
                                                                                 cost_switch_fuel=cost_switch_fuel,
                                                                                 cost_invest=cost_envelope,
                                                                                 cost_intangible=cost_intangible_seg)

        logging.debug('Updating stock segmented and renovation knowledge after demolition and renovation')
        buildings.update_stock(flow_demolition_seg, flow_remained_seg, flow_area_renovation_seg=flow_area_renovation_seg)

        if scenario_dict['info_renovation']:
            logging.debug('Information acceleration - renovation')
            cost_intangible_seg = HousingStock.acceleration_information(buildings.knowledge,
                                                                        cost_intangible_seg,
                                                                        dict_parameters[
                                                                            "Information rate max renovation"],
                                                                        dict_parameters[
                                                                            "Information rate renovation"])
        if scenario_dict['lbd_renovation']:
            logging.debug('Learning by doing - renovation')
            cost_envelope = HousingStock.learning_by_doing(buildings.knowledge,
                                                           cost_envelope,
                                                           dict_parameters["Learning by doing renovation"])

        logging.debug('Construction dynamic')
        buildings_constructed._year = year
        flow_constructed = dict_parameters['Flow needed'][year] - buildings.stock_seg.sum()
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

        if scenario_dict['lbd_construction']:
            logging.debug('Learning by doing - construction')
            cost_construction = HousingStock.learning_by_doing(buildings_constructed.knowledge,
                                                               cost_construction,
                                                               dict_parameters["Learning by doing renovation"],
                                                               cost_lim=dict_parameters['Cost construction lim'])

    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))
    logging.debug('End')
