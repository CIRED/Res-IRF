from buildings_class import HousingStockRenovated, HousingStockConstructed
from project.parse_input import *
from project.function_pandas import ds_mul_df
import time
import logging

import numpy as np

if __name__ == '__main__':

    start = time.time()
    logging.debug('Initialization')
    energy_price = energy_prices_dict['energy_price_myopic']

    logging.debug('Creating HousingStockRenovated Python object')
    buildings = HousingStockRenovated(stock_ini_seg, levels_dict, 2018, 0.035,
                                      label2area=dict_label['label2area'],
                                      label2horizon_envelope=dict_label['label2horizon_envelope'],
                                      label2horizon_heater=dict_label['label2horizon_heater'],
                                      label2discount=dict_label['label2discount'],
                                      label2income=dict_label['label2income'],
                                      label2consumption=dict_label['label2consumption'])

    logging.debug('Calibration market share --> intangible cost')
    name_file = sources_dict['intangible_cost']['source']
    if sources_dict['intangible_cost']['source_type'] == 'function':
        cost_intangible_seg = buildings.calibration_market_share(energy_price, marker_share_obj,
                                                                 folder_output=folder['intermediate'],
                                                                 cost_invest=cost_envelope,
                                                                 consumption='conventional')
        logging.debug('End of calibration and dumping: {}'.format(name_file))
        cost_intangible_seg.to_pickle(name_file)
    elif sources_dict['intangible_cost']['source_type'] == 'file':
        logging.debug('Loading intangible_cost from {}'.format(name_file))
        cost_intangible_seg = pd.read_pickle(name_file)

    logging.debug('Calibration renovation rate --> rho')
    buildings.make_calibration_renovation(energy_price, renovation_obj,
                                          cost_invest=cost_envelope,
                                          cost_intangible=cost_intangible_seg)

    years = range(2018, 2025, 1)

    logging.debug('Launching iterations')
    for year in years:
        logging.debug('YEAR: {}'.format(year))

        logging.debug('Demolition and renovation dynamic')
        flow_demolition_seg = buildings.to_flow_demolition()
        flow_remained_seg, flow_area_renovation_seg = buildings.to_flow_remained(energy_price, consumption='conventional',
                                                                                 cost_switch_fuel=cost_switch_fuel,
                                                                                 cost_invest=cost_envelope,
                                                                                 cost_intangible=cost_intangible_seg)

        logging.debug('Updating stock segmented and renovation knowledge after demolition and renovation')
        buildings.update_stock(flow_demolition_seg, flow_remained_seg, flow_area_renovation_seg=flow_area_renovation_seg)

        logging.debug('Information acceleration - renovation')
        info_rate = HousingStockRenovated.information_rate_func(buildings.knowledge,
                                                                parameters_dict["Information rate max renovation"],
                                                                parameters_dict["Information rate renovation"])

        cost_intangible_seg = ds_mul_df(info_rate.loc[cost_intangible_seg.columns], cost_intangible_seg.T).T

        logging.debug('Learning by doing - renovation')
        learning_rate = parameters_dict["Learning by doing renovation"]
        learning_by_doing_construction = buildings.knowledge ** (np.log(1 + learning_rate) / np.log(2))
        cost_envelope = cost_envelope * learning_by_doing_construction

        logging.debug('Construction dynamic')
        segments_construction = buildings.to_segments_construction(
            ['Energy performance', 'Heating energy', 'Income class owner'], {})

        buildings_constructed = HousingStockConstructed(pd.Series(dtype='float64', index=segments_construction),
                                                        levels_dict_construction, year,
                                                        label2area=dict_label['label2area_construction'],
                                                        label2horizon=dict_label['label2horizon_construction'],
                                                        label2discount=dict_label['label2discount_construction'],
                                                        label2income=dict_label['label2income'],
                                                        label2consumption=dict_label['label2consumption_construction'])

        a = pd.Series(dtype='float64', index=segments_construction)
        b = pd.concat([a] * len(levels_dict_construction['Energy performance']), keys=levels_dict_construction['Energy performance'], names=['Energy performance'], axis=1)
        c = pd.concat([b] * len(levels_dict_construction['Heating energy']), keys=levels_dict_construction['Heating energy'], names=['Heating energy'], axis=1)

    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))
    logging.debug('End')
