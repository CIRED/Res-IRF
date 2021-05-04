import pandas as pd
import os
import json
from function_pandas import linear2series, reindex_mi
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')


def json2miindex(json_dict):
    """Returns dictionary with miindex series based on a specific frame from json file.
    """

    def dict2series(item_dict):
        """Return pd.Series from a dict containing val and index labels.
        """
        if not isinstance(item_dict, dict):
            return item_dict

        if len(item_dict['index']) == 1:
            ds = pd.Series(item_dict['val'])
        elif len(item_dict['index']) == 2:
            ds = {(outerKey, innerKey): values for outerKey, innerDict in item_dict['val'].items() for
                  innerKey, values in innerDict.items()}
            ds = pd.Series(ds)
        else:
            # TODO: more than 2 MultiIndex
            raise ValueError('More than 2 MultiIndex is not yet developed')
        ds.index.set_names(item_dict['index'], inplace=True)
        return ds

    if not isinstance(json_dict, dict):
        return json_dict

    if 'val' in json_dict.keys():
        return dict2series(json_dict)
    else:
        # this means it is dict with scenario --> return dict with pd.Series
        result_dict = {}
        for scenario in json_dict.keys():
            result_dict[scenario] = dict2series(json_dict[scenario])
        return result_dict


def final2consumption(consumption, conversion):
    consumption = pd.concat([consumption] * len(conversion.index),
                            keys=conversion.index, names=conversion.index.names)
    conversion = reindex_mi(conversion, consumption.index, conversion.index.names)
    return consumption * conversion


# FOLDERS
####################################################################################################################
folder = dict()
folder['working_directory'] = os.getcwd()
folder['input'] = os.path.join(os.getcwd(), 'project', 'input')
folder['output'] = os.path.join(os.getcwd(), 'project', 'output')
folder['intermediate'] = os.path.join(os.getcwd(), 'project', 'intermediate')
folder['calibration'] = os.path.join(folder['input'], 'calibration')

####################################################################################################################

name_file = os.path.join(folder['input'], 'sources.json')
with open(name_file) as file:
    sources_dict = json.load(file)

calibration_year = sources_dict['stock_buildings']['year']

# years for input time series
start_year = calibration_year
last_year = 2080
index_input_year = range(calibration_year, last_year + 1, 1)

name_file = os.path.join(os.getcwd(), sources_dict['stock_buildings']['source'])
logging.debug('Loading parc pickle file {}'.format(name_file))
stock_ini_seg = pd.read_pickle(name_file)

name_file = os.path.join(folder['input'], 'scenario.json')
with open(name_file) as file:
    scenario_dict = json.load(file)

name_file = os.path.join(folder['input'], 'financial.json')
with open(name_file) as file:
    parameters_dict = json.load(file)

name_file = os.path.join(folder['input'], 'label2info.json')
dict_label = {}
dict_level = {}
with open(name_file) as file:
    file_dict = json.load(file)
    for key, item in file_dict.items():
        if key.split('2')[0] == 'label':
            dict_label[key] = json2miindex(item)
        else:
            dict_level[key] = item


dict_label['label2income'] = dict_label['label2income'].apply(linear2series, args=(
    parameters_dict["Household income rate"], index_input_year))

dict_label['label2consumption_heater'] = dict_label['label2primary_consumption'] * dict_label['label2heater']
dict_label['label2consumption'] = final2consumption(dict_label['label2consumption_heater'],
                                                    dict_label['label2final_energy'] ** -1)

dict_label['label2consumption_heater_construction'] = dict_label['label2primary_consumption_construction'] * dict_label[
    'label2heater_construction']
dict_label['label2consumption_construction'] = final2consumption(dict_label['label2consumption_heater_construction'],
                                                                 dict_label['label2final_energy'] ** -1)

dict_label['label2horizon_heater'] = dict_label['label2horizon_heater'][scenario_dict['investor']]
dict_label['label2horizon_envelope'] = dict_label['label2horizon_envelope'][scenario_dict['investor']]


file_dict = dict_level['levels_dict']
keys = ['Housing type', 'Occupancy status', 'Heating energy', 'Energy performance', 'Income class']
levels_dict = {key: file_dict[key] for key in keys}
levels_dict['Income class owner'] = file_dict['Income class']

keys = ['Housing type', 'Occupancy status', 'Heating energy', 'Energy performance construction', 'Income class']
levels_dict_construction = {key: file_dict[key] for key in keys}
levels_dict_construction['Income class owner'] = file_dict['Income class']

levels_dict_construction['Energy performance'] = file_dict['Energy performance construction']

name_file = os.path.join(os.getcwd(), sources_dict['renovation_cost']['source'])
cost_envelope = pd.read_csv(name_file, sep=',', header=[0], index_col=[0])
cost_envelope.index.set_names('Energy performance', inplace=True)

name_file = os.path.join(os.getcwd(), sources_dict['switch_fuel_cost']['source'])
cost_switch_fuel = pd.read_csv(name_file, index_col=[0], header=[0])

name_file = os.path.join(os.getcwd(), sources_dict['energy_prices']['source'])
with open(name_file) as file:
    file_dict = json.load(file)

energy_prices_dict = dict()
energy_price_data = pd.DataFrame()
for key, value in file_dict['price_w_taxes'].items():
    ds = linear2series(value, file_dict['price_rate'][key], index_input_year)
    ds.name = key
    energy_price_data = pd.concat((energy_price_data, ds), axis=1)
energy_prices_dict['energy_price_forecast'] = energy_price_data

temp = pd.concat([pd.Series(file_dict['price_w_taxes'])] * len(index_input_year), axis=1)
temp.index.set_names('Heating energy', inplace=True)
temp.columns = index_input_year
energy_prices_dict['energy_price_myopic'] = temp

file = 'renovation_rate_decision_maker'
name_file = os.path.join(folder['calibration'], file + '.csv')
renovation_obj = pd.read_csv(name_file, index_col=[0, 1], header=[0], squeeze=True)

name_file = os.path.join(folder['calibration'], 'market_share.csv')
marker_share_obj = pd.read_csv(name_file, index_col=[0], header=[0])
marker_share_obj.index.set_names(['Energy performance'], inplace=True)
