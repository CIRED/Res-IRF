import pandas as pd
import os
import json
import pickle5 as pickle

from project.utils import apply_linear_rate, reindex_mi


def dict2series(item_dict):
    """Return pd.Series from a dict containing val and index labels.
    """
    if len(item_dict['index']) == 1:
        ds = pd.Series(item_dict['val'])
    elif len(item_dict['index']) == 2:
        ds = {(outerKey, innerKey): values for outerKey, innerDict in item_dict['val'].items() for
              innerKey, values in innerDict.items()}
        ds = pd.Series(ds)
    else:
        raise ValueError('More than 2 MultiIndex is not yet developed')
    ds.index.set_names(item_dict['index'], inplace=True)
    return ds


def json2miindex(json_dict):
    """Parse dict and returns pd.Series or pd.DataFrame.

    Parameters
    ----------
    json_dict: dict

    Returns
    -------
    pd.Series, pd.DataFrame

    Example
    _______
    d = {'type': 'pd.Series', 'val': {'D1': 0, 'D2': 1, 'D3': 1}, 'index': ['Income class']}
    >>> json_dict(d)
    pd.Series({'D1': 0, 'D2': 1, 'D3': 1}, names=['Income class'])
    """

    if isinstance(json_dict, float) or isinstance(json_dict, int) or isinstance(json_dict, str) or isinstance(json_dict,
                                                                                                              list) or json_dict is None:
        return json_dict
    if json_dict['type'] == 'pd.Series':
        return dict2series(json_dict)
    elif json_dict['type'] == 'pd.DataFrame':
        column_name = json_dict['index'][1]
        ds = dict2series(json_dict)
        return ds.unstack(column_name)
    elif json_dict['type'] == 'float' or json_dict['type'] == 'int':
        return json_dict['val']
    elif json_dict['type'] == 'file':
        pass
    else:
        print('Need to be done!!')


def parse_json(n_file):
    """Parse json file and return dict.

    For each primary key of json file assign:
    - float, int, str, list
    - MultiIndex pd.Series or pd.DataFrame thanks to json2miindex
    - dict to reapply the function

    Parameters
    ----------
    n_file: str
    Path to json file.

    Returns
    -------
    dict
    """
    result_dict = {}
    with open(n_file) as f:
        f_dict = json.load(f)
        for key, item in f_dict.items():
            if isinstance(item, float) or isinstance(item, int) or isinstance(item, str) or isinstance(item,
                                                                                                       list) or item is None:
                result_dict[key] = item
            elif isinstance(item, dict):
                if item['type'] == 'dict':
                    result_dict[key] = item
                elif item['type'] == 'pd.Series' or item['type'] == 'pd.DataFrame':
                    result_dict[key] = json2miindex(item)
                elif item['type'] == 'dict_to_parse':
                    r_dict = {}
                    for sub_key in [sub_k for sub_k in item.keys() if sub_k != 'type']:
                        r_dict[sub_key] = json2miindex(item[sub_key])
                    result_dict[key] = r_dict
    return result_dict


def final2consumption(consumption, conversion):
    """Conversion of primary consumption to final consumption.
    """
    consumption = pd.concat([consumption] * len(conversion.index),
                            keys=conversion.index, names=conversion.index.names)
    conversion = reindex_mi(conversion, consumption.index, conversion.index.names)
    return consumption * conversion


def population_housing_dynamic(pop_housing_prev, pop_housing_min, pop_housing_ini, factor):
    """Returns number of people by building for year.

    Number of people by housing decrease over the time.
    """
    eps_pop_housing = (pop_housing_prev - pop_housing_min) / (
            pop_housing_ini - pop_housing_min)
    eps_pop_housing = max(0, min(1, eps_pop_housing))
    factor_pop_housing = factor * eps_pop_housing
    return max(pop_housing_min, pop_housing_prev * (1 + factor_pop_housing))


def forecast2myopic(forecast_price, yr):
    """Returns myopic prices based on forecast prices and a year.
    """
    val = forecast_price.loc[:, yr]
    columns_year = forecast_price.columns[forecast_price.columns >= yr]
    myopic = pd.concat([val] * len(columns_year), axis=1)
    myopic.columns = columns_year
    myopic.index.set_names('Heating energy', inplace=True)
    return myopic


folder = dict()
folder['working_directory'] = os.getcwd()
folder['input'] = os.path.join(os.getcwd(), 'project', 'input')
folder['output'] = os.path.join(os.getcwd(), 'project', 'output')
folder['intermediate'] = os.path.join(os.getcwd(), 'project', 'intermediate')
folder['calibration'] = os.path.join(folder['input'], 'calibration')

name_file = os.path.join(folder['input'], 'sources.json')
with open(name_file) as file:
    sources_dict = json.load(file)

calibration_year = sources_dict['stock_buildings']['year']

# years for input time series
start_year = calibration_year
last_year = 2080
index_input_year = range(calibration_year, last_year + 1, 1)

name_file = os.path.join(os.getcwd(), sources_dict['stock_buildings']['source'])
"""with open(name_file, 'rb') as f:
    data = pickle.load(name_file)
pickle.loads(name_file)"""

stock_ini_seg = pd.read_pickle(name_file)
stock_ini_seg = stock_ini_seg.reorder_levels(
    ['Occupancy status', 'Housing type', 'Income class', 'Heating energy', 'Energy performance', 'Income class owner'])

name_file = os.path.join(os.getcwd(), sources_dict['colors']['source'])
with open(name_file) as file:
    colors_dict = json.load(file)

name_file = os.path.join(os.getcwd(), sources_dict['parameters']['source'])
dict_parameters = parse_json(name_file)

name_file = os.path.join(os.getcwd(), sources_dict['population']['source'])
dict_parameters['Population total'] = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=None,
                                                  index_col=[0],
                                                  squeeze=True)
factor_population = stock_ini_seg.sum() / dict_parameters['Stock total ini']
dict_parameters['Population'] = dict_parameters['Population total'] * factor_population

dict_parameters['Available income'] = apply_linear_rate(dict_parameters['Available income ini'],
                                                        dict_parameters['Available income rate'], index_input_year)

# inflation
dict_parameters['Price index'] = pd.Series(1, index=index_input_year)
dict_parameters['Available income real'] = dict_parameters['Available income'] / dict_parameters['Price index']
dict_parameters['Available income real population'] = dict_parameters['Available income real'] / dict_parameters[
    'Population total']

population_housing_min = dict_parameters['Population housing min']
population_housing = dict()
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

dict_parameters['Population housing'] = pd.Series(population_housing)
dict_parameters['Flow needed'] = pd.Series(flow_needed)

name_file = os.path.join(os.getcwd(), sources_dict['label2info']['source'])
dict_label = parse_json(name_file)

dict_label['label2income'] = dict_label['label2income'].apply(apply_linear_rate, args=(
    dict_parameters["Household income rate"], index_input_year))
dict_label['label2consumption_heater'] = dict_label['label2primary_consumption'] * dict_label['label2heater']
dict_label['label2consumption'] = final2consumption(dict_label['label2consumption_heater'],
                                                    dict_label['label2final_energy'] ** -1)
dict_label['label2consumption_heater_construction'] = dict_label['label2primary_consumption_construction'] * dict_label[
    'label2heater_construction']
dict_label['label2consumption_construction'] = final2consumption(dict_label['label2consumption_heater_construction'],
                                                                 dict_label['label2final_energy'] ** -1)

file_dict = dict_label['levels_dict']
keys = ['Housing type', 'Occupancy status', 'Heating energy', 'Energy performance', 'Income class']
levels_dict = {key: file_dict[key] for key in keys}
levels_dict['Income class owner'] = file_dict['Income class']

keys = ['Housing type', 'Occupancy status', 'Heating energy', 'Energy performance construction', 'Income class']
levels_dict_construction = {key: file_dict[key] for key in keys}
levels_dict_construction['Income class owner'] = file_dict['Income class']
levels_dict_construction['Energy performance'] = file_dict['Energy performance construction']
levels_dict_construction.pop('Energy performance construction', None)

name_file = os.path.join(os.getcwd(), sources_dict['share']['source'])
dict_share = parse_json(name_file)

name_file = os.path.join(os.getcwd(), sources_dict['policies']['source'])
dict_policies = parse_json(name_file)

carbon_tax = pd.read_csv(os.path.join(os.getcwd(), sources_dict['carbon_tax']['source']), index_col=[0]) / 1000000
carbon_tax = carbon_tax.T
carbon_tax.index.set_names('Heating energy', inplace=True)
dict_policies['carbon_tax']['value'] = carbon_tax

# cost_invest
cost_invest = dict()
name_file = os.path.join(os.getcwd(), sources_dict['cost_renovation']['source'])
cost_envelope = pd.read_csv(name_file, sep=',', header=[0], index_col=[0])
cost_envelope.index.set_names('Energy performance', inplace=True)
cost_envelope.columns.set_names('Energy performance final', inplace=True)
cost_envelope = cost_envelope * (1 + 0.1) / (1 + 0.055)
cost_invest['Energy performance'] = cost_envelope

name_file = os.path.join(os.getcwd(), sources_dict['cost_switch_fuel']['source'])
cost_switch_fuel = pd.read_csv(name_file, index_col=[0], header=[0])
cost_switch_fuel.index.set_names('Heating energy', inplace=True)
cost_switch_fuel.columns.set_names('Heating energy final', inplace=True)
cost_switch_fuel = cost_switch_fuel * (1 + 0.1) / (1 + 0.055)
cost_invest['Heating energy'] = cost_switch_fuel

cost_invest_construction = dict()
name_file = os.path.join(os.getcwd(), sources_dict['cost_construction']['source'])
cost_construction = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0, 1], index_col=[0])
cost_construction.index.set_names('Housing type', inplace=True)
cost_invest_construction['Energy performance'] = cost_construction
cost_invest_construction['Heating energy'] = None

name_file = os.path.join(os.getcwd(), sources_dict['energy_prices']['source'])
with open(name_file) as file:
    file_dict = json.load(file)

energy_prices_dict = dict()
energy_price_data = pd.DataFrame()
for key, value in file_dict['price_w_taxes'].items():
    temp = apply_linear_rate(value, file_dict['price_rate'][key], index_input_year)
    temp.name = key
    energy_price_data = pd.concat((energy_price_data, temp), axis=1)
energy_price_data = energy_price_data.T
energy_price_data.index.set_names('Heating energy', inplace=True)
energy_prices_dict['energy_price_forecast'] = energy_price_data

co2_content_data = pd.DataFrame()
for key, value in file_dict['co2_content'].items():
    temp = apply_linear_rate(value, file_dict['co2_rate'][key], index_input_year)
    temp.name = key
    co2_content_data = pd.concat((co2_content_data, temp), axis=1)
co2_content_data = co2_content_data.T
co2_content_data.index.set_names('Heating energy', inplace=True)

name_file = os.path.join(os.getcwd(), sources_dict['rate_renovation_ini']['source'])
rate_renovation_ini = pd.read_csv(name_file, index_col=[0, 1], header=[0], squeeze=True)

name_file = os.path.join(os.getcwd(), sources_dict['ms_renovation_ini']['source'])
ms_renovation_ini = pd.read_csv(name_file, index_col=[0], header=[0])
ms_renovation_ini.index.set_names(['Energy performance'], inplace=True)
