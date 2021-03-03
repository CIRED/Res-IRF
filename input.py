import pandas as pd
import os
from itertools import product

from function_pandas import linear2series

# folder
folder = dict()
folder['working_directory'] = os.getcwd()
folder['input'] = os.path.join(os.getcwd(), 'input')
folder['output'] = os.path.join(os.getcwd(), 'output')
folder['middle'] = os.path.join(os.getcwd(), 'middle')
folder['calibration'] = os.path.join(folder['input'], 'calibration')
folder['calibration_middle'] = os.path.join(folder['middle'], 'calibration_middle')

# index_year
calibration_year = 2018
final_year = 2050
index_year = range(calibration_year, final_year + 1, 1)

# main language
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
language_dict['properties_names'] = ['Occupancy status', 'Housing type', 'Energy performance', 'Heating energy',
                                     'Income class', 'Income class owner']

"""all_combination_list = list(product(language_dict['occupancy_status_list'],
                                    language_dict['housing_type_list'],
                                    language_dict['energy_performance_list'],
                                    language_dict['heating_energy_list'],
                                    language_dict['income_class_list'],
                                    language_dict['income_class_list']
                                    ))


language_dict['all_combination_list'] = all_combination_list

all_combination_index = pd.MultiIndex.from_tuples(all_combination_list, names=language_dict['properties_names'])
language_dict['all_combination_index'] = all_combination_index"""

dict_replace = {'PO': 'Homeowners', 'P': 'Homeowners', 'PB': 'Landlords', 'LP': 'Landlords',
                'LS': 'Social-housing', 'MI': 'Single-family', 'MA': 'Single-family', 'LC': 'Multi-family',
                'AP': 'Multi-family', 'Collective housing': 'Multi-family', 'Individual house': 'Single-family',
                'Electricité': 'Power', 'Gaz': 'Natural gas', 'Fioul': 'Oil fuel',
                'Bois': 'Wood fuel'}

language_dict['dict_replace'] = dict_replace

language_dict['list_all_scenarios'] = ['Full capitalization', 'Reference', 'No capitalization at resale',
                                       'No capitalization in rents nor sales']

dict_color = {'Homeowners': 'lightcoral', 'Landlords': 'chocolate', 'Social-housing': 'orange',
              'Single-family': 'brown', 'Multi-family': 'darkolivegreen',
              'G': 'black', 'F': 'maroon', 'E': 'darkred', 'D': 'firebrick', 'C': 'orangered', 'B': 'lightcoral',
              'A': 'lightsalmon', 'D1': 'black', 'D2': 'maroon', 'D3': 'darkred', 'D4': 'brown', 'D5': 'firebrick',
              'D6': 'orangered', 'D7': 'tomato', 'D8': 'lightcoral', 'D9': 'lightsalmon', 'D10': 'darksalmon',
              'Power': 'darkorange', 'Natural gas': 'slategrey', 'Oil fuel': 'black', 'Wood fuel': 'saddlebrown',
              'C1': 'black', 'C2': 'darkred', 'C3': 'firebrick', 'C4': 'tomato', 'C5': 'lightsalmon'}

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

# main parameters
parameters_dict = dict()
parameters_dict['npv_min'] = -1000
parameters_dict['rate_max'] = 0.2
parameters_dict['rate_min'] = 0.00001

# learning
parameters_dict['information_rate_intangible'] = 0.25
parameters_dict['information_rate_intangible_new'] = 0.25

parameters_dict['learning_rate'] = 0.1
parameters_dict['learning_rate_new'] = 0.15
parameters_dict['learning_year'] = 10


parameters_dict['scenario'] = 'Reference'

household_income_rate = 0.012
ds_income_ini = pd.Series([13628, 20391, 24194, 27426, 31139, 35178, 39888, 45400, 54309, 92735],
                          index=language_dict['income_class_list'])
parameters_dict['income_series'] = ds_income_ini.apply(linear2series, args=(household_income_rate, index_year)).T

ds_income_ini = pd.Series([17009, 25810, 33159, 42643, 73523],
                          index=language_dict['income_class_quintile_list'])
parameters_dict['income_quintile_series'] = ds_income_ini.apply(linear2series, args=(household_income_rate, index_year)).T


ds_consumption = pd.Series([596, 392, 280, 191, 125, 76, 39], index=language_dict['energy_performance_list'],
                           name='Conventional energy')
ds_consumption_heater = pd.Series([0.85, 0.82, 0.77, 0.74, 0.72, 0.77, 1.12],
                                  index=language_dict['energy_performance_list'])
ds_consumption = ds_consumption * ds_consumption_heater

ds_consumption_conversion = pd.Series([1/2.58, 1, 1, 1], index=language_dict['heating_energy_list'])
df_consumption = pd.concat([ds_consumption] * len(language_dict['heating_energy_list']),
                           keys=language_dict['heating_energy_list'], names=['Heating energy'])
ds_consumption_conversion = ds_consumption_conversion.reindex(df_consumption.index.get_level_values('Heating energy'))
df_consumption = pd.Series(df_consumption.values * ds_consumption_conversion.values, index=df_consumption.index)
parameters_dict['energy_consumption_df'] = df_consumption

ds_consumption_new = pd.Series([50, 40], index=language_dict['energy_performance_new_list'])
ds_consumption_heater = pd.Series([0.4, 0.4], index=language_dict['energy_performance_new_list'])
ds_consumption_new = ds_consumption_new * ds_consumption_heater
parameters_dict['energy_consumption_new_series'] = ds_consumption_new

ds_surface = pd.Series([109.5, 74.3, 87.1, 53.5, 77.8, 63.3],
                       index=[['Homeowners', 'Homeowners', 'Landlords', 'Landlords', 'Social-housing',
                               'Social-housing'], ['Single-family', 'Multi-family', 'Single-family', 'Multi-family',
                                                   'Single-family', 'Multi-family']])
ds_surface.index.names = ['Occupancy status', 'Housing type']
parameters_dict['surface'] = ds_surface
ds_surface_new = pd.Series([132, 81, 90, 60, 84, 71],
                           index=[['Homeowners', 'Homeowners', 'Landlords', 'Landlords', 'Social-housing',
                                   'Social-housing'], ['Single-family', 'Multi-family', 'Single-family', 'Multi-family',
                                                       'Single-family', 'Multi-family']])
ds_surface_new.index.names = ['Occupancy status', 'Housing type']
parameters_dict['surface_new'] = ds_surface_new

surface_new_max = pd.DataFrame([[160, 101, 90], [89, 76, 76]],
                               columns=['Homeowners', 'Landlords', 'Social-housing'],
                               index=['Single-family', 'Multi-family'])
surface_new_max = surface_new_max.unstack()
surface_new_max.index.names = ['Occupancy status', 'Housing type']
parameters_dict['surface_new_max'] = surface_new_max

elasticity_surface_new = pd.Series([0.2, 0.01, 0.01], index=['Single-family', 'Multi-family', 'Social-housing'])
parameters_dict['elasticity_surface_new'] = elasticity_surface_new


df_discount_rate = pd.DataFrame([[0.15, 0.37, 0.04], [0.15, 0.37, 0.04], [0.1, 0.25, 0.04], [0.1, 0.25, 0.04],
                                 [0.07, 0.15, 0.04], [0.07, 0.15, 0.04], [0.05, 0.07, 0.04], [0.05, 0.07, 0.04],
                                 [0.04, 0.05, 0.04], [0.04, 0.05, 0.04]], columns=language_dict['housing_type_list'],
                                index=language_dict['income_class_list'])
parameters_dict['interest_rate_series'] = df_discount_rate.stack()

df_investment_horizon = pd.DataFrame([[30, 30, 30], [30, 30, 3], [30, 7, 7], [30, 7, 3]],
                                     columns=['Social-housing', 'Homeowners', 'Landlords'],
                                     index=language_dict['list_all_scenarios'])
parameters_dict['investment_horizon_series'] = df_investment_horizon.loc[parameters_dict['scenario'], :]

parameters_dict['nu_intangible_cost'] = 8
parameters_dict['nu_new'] = 8
parameters_dict['nu_label'] = 8
parameters_dict['nu_heating'] = 8


parameters_dict['lifetime_investment'] = pd.DataFrame({'enveloppe': [30, 30, 3, 3, 30, 30],
                                                       'heater': [20, 20, 3, 3, 20, 20],
                                                       'new': [25, 25, 25, 25, 25, 25]})
parameters_dict['destruction_rate'] = 0.0035

distribution_performance_new = pd.Series([0.9, 0.1], index=language_dict['energy_performance_new_list'])

name_file = 'distribution_heating_energy_new.csv'
distribution_heating_energy_new = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0], index_col=[0, 1])

parameters_dict['distribution_housing'] = pd.Series([0.39, 0.61], index=['Multi-family', 'Single family'])

parameters_dict['factor_evolution_distribution'] = 0.87

distribution_type = pd.DataFrame([[0.308, 0.397, 0.295], [0.803, 0.144, 0.053]], index=['Multi-family', 'Single-family'],
                                columns=['Homeowners', 'Landlords', 'Social-housing'])
distribution_type = distribution_type.unstack()
distribution_type.index.names = ['Occupancy status', 'Housing type']
parameters_dict['distribution_type'] = distribution_type


parameters_dict['factor_population_housing_ini'] = -0.007

parameters_dict['nb_population_housing_min'] = 2

technical_progress_dict = dict()
technical_progress_dict['learning-by-doing-new'] = -0.15
technical_progress_dict['learning-by-doing-remaining'] = -0.1
technical_progress_dict['learning-by-doing-information'] = -0.25
parameters_dict['technical_progress_dict'] = technical_progress_dict


rate_growth = 0.012

# exogenous variable
exogenous_dict = dict()
energy_price_ini = {'Power': 0.171, 'Natural gas': 0.085, 'Wood fuel': 0.062, 'Oil fuel': 0.091}
energy_price_rate = {'Power': 0.011, 'Natural gas': 0.0142, 'Wood fuel': 0.0120, 'Oil fuel': 0.0222}
# energy_price_rate = {'Power': 0.0179, 'Natural gas': 0.0142, 'Wood fuel': 0.0128, 'Oil fuel': 0.0428}
energy_price_data = pd.DataFrame()
for key, value in energy_price_ini.items():
    ds = linear2series(value, energy_price_rate[key], index_year)
    ds.name = key
    energy_price_data = pd.concat((energy_price_data, ds), axis=1)
exogenous_dict['energy_price_data'] = energy_price_data


exogenous_dict['stock_ini'] = 29037000
# population_rate = 0.003
# population_series = linear2series(population_ini, population_rate, index_year)

name_file = 'projection_population_insee.csv'
population_total_series = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=None, index_col=[0],
                                      squeeze=True)
exogenous_dict['population_total_series'] = population_total_series


# cost
cost_dict = dict()
name_file = 'cost_existing.csv'
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

name_file = 'cost_new.csv'
ds_cost_new = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0], index_col=[0, 1, 2], squeeze=True)
cost_dict['cost_new'] = ds_cost_new

df_cost_new_lim = pd.DataFrame([[700, 750], [800, 850]], index=['Single-family', 'Multi-family'], columns=['Homeowners', 'Landlords'])
cost_dict['cost_new_lim'] = df_cost_new_lim.unstack()
cost_dict['cost_new_lim'].index.names = ['Occupancy status', 'Housing type']

# for calibration
number_housing_calibration = 0.03
calibration_file = ['market_share', 'renovation_rate_decision_maker', 'renovation_rate_energy_performance']
calibration_dict = dict()

file = 'market_share'
name_file = os.path.join(folder['calibration'], file + '.csv')
calibration_dict[file] = pd.read_csv(name_file, index_col=[0], header=[0])

file = 'renovation_rate_decision_maker'
name_file = os.path.join(folder['calibration'], file + '.csv')
calibration_dict[file] = pd.read_csv(name_file, index_col=[0, 1], header=[0])




# public policy

public_policy_list = ['carbon_tax', 'CITE', 'EPTZ', 'CEE']

# TODO: Calculate public policy by surface with the average surface for decision-maker or for all.

# EPTZ
interest_rate_ini = 0.03
lifetime_eptz_ini = 5
discount_factor_eptz = (1 - (1 + interest_rate_ini) ** -lifetime_eptz_ini) / interest_rate_ini
max_eptz_ini = 21000
coef_eptz_ini = pd.Series([0.8, 0.8, 0.85, 0.85, 0.9, 0.9, 0.95, 0.95, 1, 1], index=language_dict['income_class_list'])

# CITE
rate_invest_cite = 0.17
max_cite = 16000

Part_conso_chauffage_neuf = 0.4


# Lors du calibrage, il faudra faire coïncider à l'année initiale les consommations observées par énergie
# avec le calcul fourni par Res-IRF --> Pour chaque énergie, calcul de coefficients de conversion pour faire coïncider

conso_bois_ini = 11.55 # Conso de bois à l'année réf (source ADEME) (en Mm3)// A mettre à jour
conso_2018 = [33.163 * 10**9, 105.582 * 10**9, 36.138 * 10**9, 79.554 * 10**9] # Données 2018 (CEREN climat normal), en TWh
conso_2019 = [33.8 * 10**9,  103.7 * 10**9,  34.2 * 10**9,  78.16 * 10**9] # 2019 CEREN climat normal TWh

# calcul surface moyenne pondérée
# ds_surface.reindex(dfp.index) * dfp / dfp.sum()


# rotation concerne les
rotation_rate = pd.Series([0.03, 0.18, 0.08], index=['Landlords', 'Homeowners', 'Social-housing'])
mutation_rate = pd.Series([0.035, 0.018, 0.003], index=['Landlords', 'Homeowners', 'Social-housing'])

"""stock_residuel = stock_existant_ini2*tx_destruction_residuel
stock_mobile_ini = stock_existant_ini2-stock_residue"""

