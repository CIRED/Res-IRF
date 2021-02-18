import pandas as pd
import os
from itertools import product

from function_pandas import linear2series

# from input import language_dict, parameters_dict, index_year, folder, exogenous_dict

# main language
language_dict = dict()
language_dict['occupancy_status_list'] = ['Homeowners', 'Landlords', 'Social-housing']
language_dict['housing_type_list'] = ['Single-family', 'Multi-family', 'Social-housing']
language_dict['decision_maker_list'] = list(product(language_dict['occupancy_status_list'],
                                                    language_dict['housing_type_list']))
language_dict['energy_performance_list'] = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
language_dict['income_class_list'] = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']

language_dict['heating_energy_list'] = ['Power', 'Natural gas', 'Oil fuel', 'Wood fuel']

language_dict['decision_maker_index'] = pd.MultiIndex.from_tuples(language_dict['decision_maker_list'],
                                                                  names=['Occupancy status', 'Housing type'])
language_dict['properties_names'] = ['Occupancy status', 'Housing type', 'Energy performance', 'Heating energy',
                                     'Income class', 'Income class owner']

all_combination_list = list(product(language_dict['occupancy_status_list'],
                                    language_dict['housing_type_list'],
                                    language_dict['energy_performance_list'],
                                    language_dict['heating_energy_list'],
                                    language_dict['income_class_list'],
                                    language_dict['income_class_list']
                                    ))


language_dict['all_combination_list'] = all_combination_list

all_combination_index = pd.MultiIndex.from_tuples(all_combination_list, names=language_dict['properties_names'])
language_dict['all_combination_index'] = all_combination_index

dict_replace = {'PO': 'Homeowners', 'P': 'Homeowners', 'PB': 'Landlords', 'LP': 'Landlords',
                'LS': 'Social-housing', 'MI': 'Single-family', 'MA': 'Single-family',  'LC': 'Multi-family',
                'AP': 'Multi-family', 'Electricité': 'Power', 'Gaz': 'Natural gas', 'Fioul': 'Oil fuel',
                'Bois': 'Wood fuel'}

language_dict['dict_replace'] = dict_replace

# main parameters
parameters_dict = dict()
parameters_dict['npv_min'] = -1000
parameters_dict['r'] = 1
parameters_dict['rate_max'] = 0.2
parameters_dict['rate_min'] = 0.00001

interest_rate_list = list(product(language_dict['occupancy_status_list'], language_dict['housing_type_list'],
                                  language_dict['income_class_list']))
interest_rate_index = pd.MultiIndex.from_tuples(interest_rate_list, names=["Occupancy status", "Housing type",
                                                                           "Income class"])
interest_rate_series = pd.Series(0.05, index=interest_rate_index)
parameters_dict['interest_rate_series'] = interest_rate_series
parameters_dict['investment_horizon_series'] = pd.Series(10, index=language_dict['decision_maker_index'])

ds_income = pd.Series([13628, 20391, 24194, 27426, 31139, 35178, 39888, 45400, 54309, 92735],
                      index=language_dict['income_class_list'][::-1])
parameters_dict['income_series'] = ds_income

ds_conso = pd.Series([596, 392, 280, 191, 125, 76, 39], index=language_dict['energy_performance_list'],
                     name='Conventional energy')
ds_conso_heater = pd.Series([0.85, 0.82, 0.77, 0.74, 0.72, 0.77, 1.12], index=language_dict['energy_performance_list'])
ds_conso = ds_conso * ds_conso_heater
parameters_dict['energy_consumption_series'] = ds_conso

ds_surface = pd.Series([109.5, 74.3, 87.1, 53.5, 77.8, 63.3],
                           index=[['Homeowners', 'Homeowners', 'Landlords', 'Landlords', 'Social-housing',
                                   'Social-housing'], ['Single-family', 'Multi-family', 'Single-family', 'Multi-family',
                                                       'Single-family', 'Multi-family']])
parameters_dict['surface'] = ds_surface

# index_year
calibration_year = 2018
final_year = 2050
index_year = range(calibration_year, final_year + 1, 1)

# folder
folder = dict()
folder['working_directory'] = os.getcwd()
folder['input'] = os.path.join(os.getcwd(), 'input')
folder['output'] = os.path.join(os.getcwd(), 'output')
folder['middle'] = os.path.join(os.getcwd(), 'middle')

# exogenous variable
exogenous_dict = dict()
energy_price_ini = {'Power': 0.171, 'Natural gas': 0.085, 'Wood fuel': 0.062, 'Oil fuel': 0.091}
energy_price_rate = {'Power': 0.0179, 'Natural gas': 0.0273, 'Wood fuel': 0.0128, 'Oil fuel': 0.0438}
energy_price_data = pd.DataFrame()
for key, value in energy_price_ini.items():
    ds = linear2series(value, energy_price_rate[key], index_year)
    ds.name = key
    energy_price_data = pd.concat((energy_price_data, ds), axis=1)
exogenous_dict['energy_price_data'] = energy_price_data

population_ini = 50
population_rate = 0.02
population_series = linear2series(population_ini, population_rate, index_year)
exogenous_dict['population_series'] = population_series

national_income_ini = 50
national_income_rate = 0.2
national_income_series = linear2series(national_income_ini, national_income_rate, index_year)
exogenous_dict['national_income_series'] = national_income_series

# cost
cost_dict = dict()
name_file = 'CINV_existant.csv'
df_cost_inv = pd.read_csv(os.path.join(folder['input'], name_file), sep=';', header=None, index_col=None)
df_cost_inv.columns = sorted(language_dict['energy_performance_list'])[1:]
df_cost_inv.index = sorted(language_dict['energy_performance_list'])[:-1]
cost_dict['cost_inv'] = df_cost_inv.T

energy_list = ['Power', 'Natural gas', 'Oil fuel', 'Wood fuel']
df_cost_switch_fuel = pd.DataFrame([(0, 70, 100, 120), (55, 0, 80, 100), (55, 50, 0 ,100), (55, 50, 80, 0)],
                                   index=energy_list, columns=energy_list)
cost_dict['cost_switch_fuel'] = df_cost_switch_fuel

print('pause')
