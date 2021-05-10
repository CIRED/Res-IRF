import pandas as pd
import os
import json
from function_pandas import linear2series, reindex_mi
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')


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


def json2miindex(json_dict):
    """Returns dictionary with miindex series based on a specific frame from json file.
    """
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


def _json2miindex(json_dict):
    if json_dict['type'] == 'pd.Series':
        return dict2series(json_dict)
    elif json_dict['type'] == 'pd.DataFrame':
        column_name = json_dict['index'][1]
        ds = dict2series(json_dict)
        return ds.unstack(column_name)
    else:
        print('Need to be done!!')


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

dict_parameters = {}
name_file = os.path.join(folder['input'], 'parameters.json')
with open(name_file) as file:
    file_dict = json.load(file)
    for key, item in file_dict.items():
        if isinstance(item, float) or isinstance(item, int):
            dict_parameters[key] = item
        else:
            dict_parameters[key] = _json2miindex(item)

name_file = os.path.join(os.getcwd(), sources_dict['population']['source'])
dict_parameters['Population total'] = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=None,
                                                  index_col=[0],
                                                  squeeze=True)
factor_population = stock_ini_seg.sum() / dict_parameters['Stock total ini']
dict_parameters['Population'] = dict_parameters['Population total'] * factor_population

dict_parameters['Available income'] = linear2series(dict_parameters['Available income ini'],
                                                    dict_parameters['Available income rate'], index_input_year)

# inflation
dict_parameters['Price index'] = pd.Series(1, index=index_input_year)
dict_parameters['Available income real'] = dict_parameters['Available income'] / dict_parameters['Price index']
dict_parameters['Available income real population'] = dict_parameters['Available income real'] / dict_parameters[
    'Population total']


def population_housing_dynamic(population_housing_prev, population_housing_min, population_housing_ini, factor):
    """Returns number of people by building for year.

    Number of people by housing decrease over the time.
    TODO: It seems we could get the number of people by buildings exogeneously.
    """
    eps_pop_housing = (population_housing_prev - population_housing_min) / (
            population_housing_ini - population_housing_min)
    eps_pop_housing = max(0, min(1, eps_pop_housing))
    factor_pop_housing = factor * eps_pop_housing
    return max(population_housing_min, population_housing_prev * (1 + factor_pop_housing))


population_housing_min = dict_parameters["Population housing min"]
population_housing = {}
population_housing[calibration_year] = dict_parameters['Population'].loc[calibration_year] / stock_ini_seg.sum()
max_year = max(dict_parameters['Population'].index)

flow_needed = dict()
flow_needed[calibration_year] = dict_parameters['Population'].loc[calibration_year] / population_housing[
    calibration_year]

for year in index_input_year[1:]:
    if year > max_year:
        break
    population_housing[year] = population_housing_dynamic(population_housing[year - 1],
                                                          population_housing_min,
                                                          population_housing[calibration_year],
                                                          dict_parameters['Factor population housing ini'])
    flow_needed[year] = dict_parameters['Population'].loc[year] / population_housing[year]

dict_parameters['Population housing'] = population_housing
dict_parameters['Flow needed'] = flow_needed

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
    dict_parameters["Household income rate"], index_input_year))
dict_label['label2consumption_heater'] = dict_label['label2primary_consumption'] * dict_label['label2heater']
dict_label['label2consumption'] = final2consumption(dict_label['label2consumption_heater'],
                                                    dict_label['label2final_energy'] ** -1)
dict_label['label2consumption_heater_construction'] = dict_label['label2primary_consumption_construction'] * dict_label[
    'label2heater_construction']
dict_label['label2consumption_construction'] = final2consumption(dict_label['label2consumption_heater_construction'],
                                                                 dict_label['label2final_energy'] ** -1)
dict_label['label2horizon_heater'] = dict_label['label2horizon_heater'][scenario_dict['investor']]
dict_label['label2horizon_envelope'] = dict_label['label2horizon_envelope'][scenario_dict['investor']]

label2horizon = dict()
label2horizon['envelope'] = dict_label['label2horizon_envelope']
label2horizon['heater'] = dict_label['label2horizon_heater']

dict_label['label2horizon'] = label2horizon

file_dict = dict_level['levels_dict']
keys = ['Housing type', 'Occupancy status', 'Heating energy', 'Energy performance', 'Income class']
levels_dict = {key: file_dict[key] for key in keys}
levels_dict['Income class owner'] = file_dict['Income class']

keys = ['Housing type', 'Occupancy status', 'Heating energy', 'Energy performance construction', 'Income class']
levels_dict_construction = {key: file_dict[key] for key in keys}
levels_dict_construction['Income class owner'] = file_dict['Income class']
levels_dict_construction['Energy performance'] = file_dict['Energy performance construction']

dict_result = {}
name_file = os.path.join(folder['input'], 'share.json')
with open(name_file) as file:
    file_dict = json.load(file)
    for key, item in file_dict.items():
        dict_result[key] = _json2miindex(item)

name_file = os.path.join(os.getcwd(), sources_dict['cost_renovation']['source'])
cost_envelope = pd.read_csv(name_file, sep=',', header=[0], index_col=[0])
cost_envelope.index.set_names('Energy performance', inplace=True)
cost_envelope.columns.set_names('Energy performance final', inplace=True)

name_file = os.path.join(os.getcwd(), sources_dict['cost_switch_fuel']['source'])
cost_switch_fuel = pd.read_csv(name_file, index_col=[0], header=[0])
cost_switch_fuel.index.set_names('Heating energy', inplace=True)
cost_switch_fuel.columns.set_names('Heating energy final', inplace=True)

name_file = os.path.join(os.getcwd(), sources_dict['cost_construction']['source'])
cost_construction = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0, 1], index_col=[0])
cost_construction.index.set_names('Housing type', inplace=True)

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
