import pandas as pd
import os
from itertools import product

from function_pandas import linear2series, de_aggregate_columns, ds_mul_df, de_aggregating_series

# FOLDERS
########################################################################################################################
folder = dict()
folder['working_directory'] = os.getcwd()
folder['input'] = os.path.join(os.getcwd(), 'project', 'input')
folder['output'] = os.path.join(os.getcwd(), 'project', 'output')
folder['intermediate'] = os.path.join(os.getcwd(), 'project', 'intermediate')
folder['calibration'] = os.path.join(folder['input'], 'calibration')
folder['calibration_intermediate'] = os.path.join(folder['intermediate'], 'calibration_intermediate')

# INDEX-YEARS
########################################################################################################################
calibration_year = 2018
final_year = 2080
index_year = range(calibration_year, final_year + 1, 1)

# years for input time series
start_year = calibration_year
last_year = 2080
index_input_year = range(calibration_year, last_year + 1, 1)

# LANGUAGE-DICT
########################################################################################################################
language_dict = dict()
language_dict['occupancy_status_list'] = ['Homeowners', 'Landlords', 'Social-housing']
language_dict['housing_type_list'] = ['Single-family', 'Multi-family', 'Social-housing']
language_dict['decision_maker_list'] = list(product(language_dict['occupancy_status_list'],
                                                    language_dict['housing_type_list']))
language_dict['energy_performance_list'] = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
language_dict['energy_performance_new_list'] = ['BBC', 'BEPOS']
language_dict['income_class_list'] = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
language_dict['income_class_quintile_list'] = ['C1', 'C2', 'C3', 'C4', 'C5']
language_dict['decision_maker_income_list'] = list(product(language_dict['occupancy_status_list'],
                                                           language_dict['housing_type_list'],
                                                           language_dict['income_class_list']))

language_dict['heating_energy_list'] = ['Power', 'Natural gas', 'Oil fuel', 'Wood fuel']


language_dict['decision_maker_index'] = pd.MultiIndex.from_tuples(language_dict['decision_maker_list'],
                                                                  names=['Occupancy status', 'Housing type'])

language_dict['levels_names'] = ['Occupancy status', 'Housing type', 'Energy performance', 'Heating energy',
                                 'Income class', 'Income class owner']

dict_replace = {'PO': 'Homeowners', 'P': 'Homeowners', 'PB': 'Landlords', 'LP': 'Landlords',
                'LS': 'Social-housing', 'MI': 'Single-family', 'MA': 'Single-family', 'LC': 'Multi-family',
                'AP': 'Multi-family', 'Collective housing': 'Multi-family', 'Individual house': 'Single-family',
                'Electricité': 'Power', 'Gaz': 'Natural gas', 'Fioul': 'Oil fuel',
                'Bois': 'Wood fuel'}

language_dict['dict_replace'] = dict_replace

language_dict['list_all_scenarios'] = ['Full capitalization', 'Reference', 'No capitalization at resale',
                                       'No capitalization in rents nor sales']

# COLORS
########################################################################################################################

dict_color = {'Homeowners': 'lightcoral', 'Landlords': 'chocolate', 'Social-housing': 'orange',
              'Single-family': 'brown', 'Multi-family': 'darkolivegreen',
              'G': 'black', 'F': 'darkmagenta', 'E': 'rebeccapurple', 'D': 'red', 'C': 'orangered', 'B': 'lightcoral',
              'A': 'lightsalmon', 'D1': 'black', 'D2': 'maroon', 'D3': 'darkred', 'D4': 'brown', 'D5': 'firebrick',
              'D6': 'orangered', 'D7': 'tomato', 'D8': 'lightcoral', 'D9': 'lightsalmon', 'D10': 'darksalmon',
              'Power': 'darkorange', 'Natural gas': 'slategrey', 'Oil fuel': 'black', 'Wood fuel': 'saddlebrown',
              'C1': 'black', 'C2': 'darkred', 'C3': 'firebrick', 'C4': 'tomato', 'C5': 'lightsalmon', 'BBC': 'black',
              'BEPOS': 'red'}

# ENGLISH-TO-FRENCH
########################################################################################################################

dict_english_to_french = {'Housing type': 'Type de logement', 'Heating energy': 'Energie de chauffage',
                          'Homeowners': 'Propriétaires', 'Landlords': 'Propriétaire bailleur',
                          'Occupancy status': "Statut d'occupation", 'Multi-family': 'Logement collectif',
                          'Single-family': 'Logement individuel', 'Energy performance': 'DPE',
                          'Natural gas': 'Gaz naturel', 'Oil fuel': 'Fioul doméstique', 'Power': 'Électricité',
                          'Wood fuel': 'Bois de chauffage', 'Income class': 'Revenu ménages',
                          'Housing number': 'Nombre de logements', 'Energy performance final': 'DPE final',
                          'Heating energy final': 'Energie de chauffage finale'}


dict_color_french = {v: dict_color[k] for k, v in dict_english_to_french.items() if k in dict_color.keys()}
dict_color.update(dict_color_french)

language_dict['color'] = dict_color
language_dict['english_to_french'] = dict_english_to_french

########################################################################################################################
# main parameters
parameters_dict = dict()

# STOCK DYNAMIC
parameters_dict['destruction_rate'] = 0.0035
parameters_dict['residual_destruction_rate'] = 0.05

# CONSUMPTION
########################################################################################################################
consumption_ep = pd.Series([596, 392, 280, 191, 125, 76, 39], index=language_dict['energy_performance_list'],
                           name='Conventional energy')
consumption_ep.index.set_names('Energy performance', inplace=True)
consumption_heater_ep = pd.Series([0.85, 0.82, 0.77, 0.74, 0.72, 0.77, 1.12],
                                  index=language_dict['energy_performance_list'])
consumption_heater_ep.index.set_names('Energy performance', inplace=True)

consumption_new_ep = pd.Series([50, 40], index=language_dict['energy_performance_new_list'])
consumption_new_ep.index.set_names('Energy performance', inplace=True)
consumption_heater_new_ep = pd.Series([0.4, 0.4], index=language_dict['energy_performance_new_list'])
consumption_heater_new_ep.index.set_names('Energy performance', inplace=True)


def final_2heater_consumption(ds_consumption, ds_consumption_heater):
    ds_consumption = ds_consumption * ds_consumption_heater
    ds_consumption_conversion = pd.Series([1/2.58, 1, 1, 1], index=language_dict['heating_energy_list'])
    df_consumption = pd.concat([ds_consumption] * len(language_dict['heating_energy_list']),
                               keys=language_dict['heating_energy_list'], names=['Heating energy'])
    ds_consumption_conversion = ds_consumption_conversion.reindex(df_consumption.index.get_level_values('Heating energy'))
    df_consumption = pd.Series(df_consumption.values * ds_consumption_conversion.values, index=df_consumption.index)
    return df_consumption


parameters_dict['energy_consumption_df'] = final_2heater_consumption(consumption_ep, consumption_heater_ep)
parameters_dict['energy_consumption_new_series'] = final_2heater_consumption(consumption_new_ep, consumption_heater_new_ep)

# AREA
########################################################################################################################
ds_area = pd.Series([109.5, 74.3, 87.1, 53.5, 77.8, 63.3],
                    index=[['Homeowners', 'Homeowners', 'Landlords', 'Landlords', 'Social-housing',
                            'Social-housing'], ['Single-family', 'Multi-family', 'Single-family', 'Multi-family',
                                                'Single-family', 'Multi-family']])
ds_area.index.names = ['Occupancy status', 'Housing type']
parameters_dict['area'] = ds_area
ds_area_new = pd.Series([132, 81, 90, 60, 84, 71],
                        index=[['Homeowners', 'Homeowners', 'Landlords', 'Landlords', 'Social-housing',
                                'Social-housing'], ['Single-family', 'Multi-family', 'Single-family', 'Multi-family',
                                                    'Single-family', 'Multi-family']])
ds_area_new.index.names = ['Occupancy status', 'Housing type']
parameters_dict['area_new'] = ds_area_new

area_new_max = pd.DataFrame([[160, 101, 90], [89, 76, 76]],
                            columns=['Homeowners', 'Landlords', 'Social-housing'],
                            index=['Single-family', 'Multi-family'])
area_new_max = area_new_max.unstack()
area_new_max.index.names = ['Occupancy status', 'Housing type']
parameters_dict['area_new_max'] = area_new_max

elasticity_area_new = pd.Series([0.2, 0.01, 0.01], index=['Single-family', 'Multi-family', 'Social-housing'])
temp = elasticity_area_new.reindex(area_new_max.index.get_level_values('Housing type'))
temp.index = area_new_max.index
parameters_dict['elasticity_area_new_ini'] = temp

# INVESTMENT PARAMETERS
########################################################################################################################

# INTEREST RATE
interest_rate_seg = pd.DataFrame([[0.15, 0.37, 0.04], [0.15, 0.37, 0.04], [0.1, 0.25, 0.04], [0.1, 0.25, 0.04],
                                 [0.07, 0.15, 0.04], [0.07, 0.15, 0.04], [0.05, 0.07, 0.04], [0.05, 0.07, 0.04],
                                 [0.04, 0.05, 0.04], [0.04, 0.05, 0.04]], columns=language_dict['housing_type_list'],
                                 index=language_dict['income_class_list']).stack()
interest_rate_seg.index.names = ['Income class owner', 'Housing type']
parameters_dict['interest_rate_seg'] = interest_rate_seg

temp = pd.Series([0.07, 0.1, 0.04], index=language_dict['housing_type_list'])
temp.index.set_names('Housing type', inplace=True)
parameters_dict['interest_rate_new'] = temp

# INVESTMENT HORIZON
parameters_dict['scenario'] = 'Reference'
invest_hrz_envelope_seg = pd.DataFrame([[30, 30, 30], [30, 30, 3], [30, 7, 7], [30, 7, 3]],
                                       columns=['Social-housing', 'Homeowners', 'Landlords'],
                                       index=language_dict['list_all_scenarios'])
invest_hrz_envelope_seg.columns.set_names('Housing type')
invest_hrz_heater_seg = pd.DataFrame([[16, 16, 16], [16, 16, 3], [16, 7, 7], [15, 7, 3]],
                                     columns=['Social-housing', 'Homeowners', 'Landlords'],
                                     index=language_dict['list_all_scenarios'])
invest_hrz_heater_seg.columns.set_names('Housing type')
parameters_dict['investment_horizon_envelope_ds'] = invest_hrz_envelope_seg.loc[parameters_dict['scenario'], :]
parameters_dict['investment_horizon_heater_ds'] = invest_hrz_heater_seg.loc[parameters_dict['scenario'], :]
parameters_dict['investment_horizon_construction'] = 35

# MARKET SHARE
parameters_dict['nu_intangible_cost'] = 8
parameters_dict['nu_new'] = 8.0
parameters_dict['nu_label'] = 8
parameters_dict['nu_energy'] = 8


# RENOVATION RATE
parameters_dict['npv_min'] = -1000
parameters_dict['rate_max'] = 0.4
parameters_dict['rate_min'] = 0.00001


# SHARE
########################################################################################################################

name_file = 'distribution_heating_energy_new.csv'
distribution_heating_energy_new = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0],
                                              index_col=[0, 1])

parameters_dict['ht_share_tot'] = pd.Series([0.39, 0.61], index=['Multi-family', 'Single family'])
parameters_dict['factor_evolution_distribution'] = 0.87

temp = pd.DataFrame([[0.308, 0.397, 0.295], [0.803, 0.144, 0.053]],
                    index=['Multi-family', 'Single-family'],
                    columns=['Homeowners', 'Landlords', 'Social-housing'])
temp.index.set_names('Housing type', inplace=True)
temp.columns.set_names('Occupancy status', inplace=True)
parameters_dict['os_share_ht_construction'] = temp

temp = pd.Series([0.9, 0.1], index=['BBC', 'BEPOS'])
temp.index.set_names('Energy performance', inplace=True)
parameters_dict['ep_share_tot_construction'] = temp

temp = pd.Series([0.57, 0.43], index=['Multi-family', 'Single-family'])
temp.index.set_names('Housing type', inplace=True)
parameters_dict['ht_share_tot_construction'] = temp

temp = pd.DataFrame([[0.753, 0.185, 0.005, 0.058], [0.195, 0.795, 0, 0.01]],
                    index=['Single-family', 'Multi-family'],
                    columns=['Power', 'Natural gas', 'Oil fuel', 'Wood fuel'])
temp.index.set_names('Housing type', inplace=True)
temp.columns.set_names('Heating energy', inplace=True)
parameters_dict['he_share_ht_construction'] = temp

# TECHNICAL PROGRESS
########################################################################################################################

technical_progress_dict = dict()
technical_progress_dict['learning-by-doing-new'] = -0.15
technical_progress_dict['learning-by-doing-renovation'] = -0.1
technical_progress_dict['learning-by-doing-information'] = -0.25
technical_progress_dict['learning_year'] = 10
technical_progress_dict['information_rate_intangible'] = 0.25
technical_progress_dict['information_rate_max'] = 0.8
technical_progress_dict['information_rate_intangible_new'] = 0.25
technical_progress_dict['information_rate_max_new'] = 0.95
parameters_dict['technical_progress_dict'] = technical_progress_dict

# EXOGENOUS VARIABLE
########################################################################################################################

parameters_dict['factor_population_housing_ini'] = -0.007
parameters_dict['nb_population_housing_min'] = 2

household_income_rate = 0.012
ds_income_ini = pd.Series([13628, 20391, 24194, 27426, 31139, 35178, 39888, 45400, 54309, 92735],
                          index=language_dict['income_class_list'])
parameters_dict['income_series'] = ds_income_ini.apply(linear2series, args=(household_income_rate, index_input_year)).T

ds_income_ini = pd.Series([17009, 25810, 33159, 42643, 73523],
                          index=language_dict['income_class_quintile_list'])
parameters_dict['income_quintile_series'] = ds_income_ini.apply(linear2series, args=(household_income_rate, index_input_year)).T

exogenous_dict = dict()
energy_price_ini = {'Power': 0.171, 'Natural gas': 0.085, 'Wood fuel': 0.062, 'Oil fuel': 0.091}
energy_price_rate = {'Power': 0.011, 'Natural gas': 0.0142, 'Wood fuel': 0.0120, 'Oil fuel': 0.0222}
# energy_price_rate = {'Power': 0.0179, 'Natural gas': 0.0142, 'Wood fuel': 0.0128, 'Oil fuel': 0.0428}
energy_price_data = pd.DataFrame()
for key, value in energy_price_ini.items():
    ds = linear2series(value, energy_price_rate[key], index_input_year)
    ds.name = key
    energy_price_data = pd.concat((energy_price_data, ds), axis=1)
exogenous_dict['energy_price_forecast'] = energy_price_data
temp = pd.concat([pd.Series(energy_price_ini)] * len(index_input_year), axis=1)
temp.index.set_names('Heating energy', inplace=True)
temp.columns = index_input_year
exogenous_dict['energy_price_myopic'] = temp.T

# co2_content is in gCO2/kWh
co2_content = {'Power': 147, 'Natural gas': 227, 'Oil fuel': 324, 'Wood fuel': 30}

# total buildings stock that is bigger than segmented stock data sum
exogenous_dict['stock_ini'] = 29037000
# population_rate = 0.003
# population_series = linear2series(population_ini, population_rate, index_year)

name_file = 'projection_population_insee.csv'
population_total_series = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=None, index_col=[0],
                                      squeeze=True)

# TODO decrease population to reflect population based on building stock - change stock_segmented_sum
stock_segmented_sum = exogenous_dict['stock_ini']
factor_population = stock_segmented_sum / exogenous_dict['stock_ini']
exogenous_dict['population_total_ds'] = population_total_series * factor_population

# available income is in Md€ = aggregated income of the population
exogenous_dict['available_income_ini'] = 14210000000000 * factor_population
exogenous_dict['available_income_rate'] = 0.012
exogenous_dict['available_income_ds'] = linear2series(exogenous_dict['available_income_ini'],
                                                      exogenous_dict['available_income_rate'], index_input_year)

# inflation
exogenous_dict['price_index_ds'] = pd.Series(1, index=index_input_year)
exogenous_dict['available_income_real_ds'] = exogenous_dict['available_income_ds'] / exogenous_dict['price_index_ds']
exogenous_dict['available_income_real_pop_ds'] = exogenous_dict['available_income_real_ds'] / exogenous_dict[
    'population_total_ds']

# COST
########################################################################################################################
cost_dict = dict()
name_file = 'cost_renovation.csv'
df_cost_inv = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0], index_col=[0])
cost_dict['cost_inv'] = df_cost_inv

energy_list = ['Power', 'Natural gas', 'Oil fuel', 'Wood fuel']
df_cost_switch_fuel = pd.DataFrame([(0, 70, 100, 120), (55, 0, 80, 100), (55, 50, 0, 100), (55, 50, 80, 0)],
                                   index=energy_list, columns=energy_list)
cost_dict['cost_switch_fuel'] = df_cost_switch_fuel

df_intangible_cost = pd.DataFrame(
    [(0, 162, 65, 66, 21, 215, 289), (0, 0, 141, 172, 15, 39, 330), (0, 0, 0, 125, 31, 58, 313),
     (0, 0, 0, 0, 88, 62, 216), (0, 0, 0, 0, 0, 134, 77), (0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0)],
    index=language_dict['energy_performance_list'], columns=language_dict['energy_performance_list'])
cost_dict['cost_intangible'] = df_intangible_cost

name_file = 'cost_construction.csv'
ds_cost_new = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0], index_col=[0, 1, 2], squeeze=True)
cost_dict['cost_new'] = ds_cost_new

df_cost_new_lim = pd.DataFrame([[700, 750], [800, 850]], index=['Single-family', 'Multi-family'], columns=['Homeowners', 'Landlords'])
cost_dict['cost_new_lim'] = df_cost_new_lim.unstack()
cost_dict['cost_new_lim'].index.names = ['Occupancy status', 'Housing type']

# CALIBRATION
########################################################################################################################

calibration_file = ['market_share', 'renovation_rate_decision_maker', 'renovation_rate_energy_performance']
calibration_dict = dict()

renovation_rate_calibration = 0.03
calibration_dict['renovation_rate_calibration'] = renovation_rate_calibration

file = 'market_share'
name_file = os.path.join(folder['calibration'], file + '.csv')
calibration_dict[file] = pd.read_csv(name_file, index_col=[0], header=[0])

file = 'renovation_rate_decision_maker'
name_file = os.path.join(folder['calibration'], file + '.csv')
calibration_dict[file] = pd.read_csv(name_file, index_col=[0, 1], header=[0], squeeze=True)

"""file = 'renovation_share_energy_performance'
name_file = os.path.join(folder['calibration'], file + '.csv')
calibration_dict[file] = pd.read_csv(name_file, index_col=[0], header=[0], squeeze=True)

calibration_dict['renovation_rate_dm_ep'] = de_aggregating_series(calibration_dict['renovation_rate_decision_maker'],
                                                                  calibration_dict[
                                                                      'renovation_share_energy_performance'],
                                                                  'Energy performance')"""

# parameter that explicit where to find the variable or if it's need to be calculated
parameters_dict['intangible_cost_source'] = 'pickle'

# PUBLIC POLICY
########################################################################################################################
public_policy_list = ['carbon_tax', 'CITE', 'EPTZ', 'CEE']

rotation_rate = pd.Series([0.03, 0.18, 0.08], index=['Landlords', 'Homeowners', 'Social-housing'])
mutation_rate = pd.Series([0.035, 0.018, 0.003], index=['Landlords', 'Homeowners', 'Social-housing'])




