import pickle
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import re
import argparse
import os
import pandas as pd
import ui_utils


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default=False, help='config file path')
args = parser.parse_args()

config = pd.read_csv(os.path.join('project', args.config), squeeze=True, header=None, index_col=[0])
config.dropna(inplace=True)

list_years = [int(re.search('20[0-9][0-9]', idx)[0]) for idx in config.index if re.search('20[0-9][0-9]', idx)]

# 0 Discount factor to extend energy and emission saving
discount_rate = 0.04
lifetime = 26
discount_factor = (1 - (1 + discount_rate) ** -lifetime) / discount_rate


folder = config['Folder name']
folder_output = os.path.join(os.getcwd(), 'project', 'output', folder)
scenarios = [f for f in os.listdir(folder_output) if os.path.isdir(os.path.join(folder_output, f))]
folders = {scenario: os.path.join(folder_output, scenario) for scenario in scenarios} 

summaries = {scenario: pd.read_csv(os.path.join(folders[scenario], 'summary.csv'), index_col=[0]) for scenario in scenarios}
summaries = ui_utils.reverse_nested_dict(summaries)
summaries = {key: pd.DataFrame(item) for key, item in summaries.items()}

detailed = {scenario: pd.read_csv(os.path.join(folders[scenario], 'detailed.csv'), index_col=[0]).T for scenario in scenarios}
detailed = ui_utils.reverse_nested_dict(detailed)
detailed = {key: pd.DataFrame(item) for key, item in detailed.items()}

# 1. Efficency
# 1.1 Cost-effectivness

# 1.1.1 Consumption
df = summaries['Consumption actual (kWh)'].copy()
marginal_consumption_actual = pd.Series([(df[config['All politics']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years], index=list_years)

df = summaries['Consumption conventional (kWh)'].copy()
marginal_consumption_conventional = pd.Series([(df[config['All politics']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years], index=list_years)

# 1.1.2 Emission
df = summaries['Emission (gCO2)'].copy()
marginal_emission = pd.Series([(df[config['All politics']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years], index=list_years)

# 1.1.3 Policy cost

df = summaries['{} (€)'.format(config['Policy name']).replace('_', ' ').capitalize()].copy()
df.fillna(0, inplace=True)
marginal_subsidies = pd.Series([(df[config['All politics']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years], index=list_years)


# 1.1.5 Results
cost_effectivness = abs(pd.concat((marginal_subsidies / (marginal_consumption_actual * discount_factor),
                                   marginal_subsidies / (marginal_consumption_conventional * discount_factor),
                                   marginal_subsidies / (marginal_emission * discount_factor) * 10**6),
                                  axis=1))
cost_effectivness.columns = ['Cost-effectivness actual (€/kWh)',
                             'Cost-effectivness conventional (€/kWh)',
                             'Cost-effectivness emission (€/tCO2)']
cost_effectivness.sort_index(inplace=True)


# 2.1 Leverage effect
                        
df = summaries['Capex renovation (€)'].copy()
df.fillna(0, inplace=True)
marginal_investment = pd.Series([(df[config['All politics']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years], index=list_years)

leverage = marginal_investment / marginal_subsidies
leverage.sort_index(inplace=True)
leverage.name = 'Leverage'
print(leverage)

pd.concat((cost_effectivness, leverage), axis=1).to_csv(os.path.join(folder_output, 'indicator_policies.csv'))

# 2. Effectivness
# 2.1 Consumption actual
# 2.1.1 Reduction of final energy consumption by 20% by 2030 compared to 2012
# 2.1.2 Reduction of 50% by 2050 compared to 2012
df = summaries['Consumption actual (kWh)']
simple_diff = df[config['All politics']] - df[config['All politics - 1']]
double_diff = simple_diff.diff()
double_diff.iloc[0] = simple_diff.iloc[0]
energy_saving = double_diff * discount_factor / 10**9
energy_saving = energy_saving.cumsum()
energy_saving.name = 'Energy difference (TWh)'
result = energy_saving

# 2.2 Renovation by year
# 2. Energy renovation of 500,000 homes per year, including 120,000 in social housing;

df = summaries['Flow transition renovation']
simple_diff = df[config['All politics']] - df[config['All politics - 1']]
additional_renovation = simple_diff.cumsum() / 10**3
additional_renovation.name = 'Renovation difference (Thousands)'
result = pd.concat((result, additional_renovation), axis=1)

# G and F buildings in 2025
df = detailed['Stock F (Thousands)'] + detailed['Stock G (Thousands)']
df.index = df.index.astype('int64')
simple_diff = df[config['All politics']] - df[config['All politics - 1']]
high_energy_buildings = simple_diff
high_energy_buildings.name = 'High energy building difference (Thousands)'
result = pd.concat((result, high_energy_buildings), axis=1)

# Share of F and G buildings in the total buildings stock
detailed['Stock (Thousands)'].index = detailed['Stock (Thousands)'].index.astype('int64')
df = df / detailed['Stock (Thousands)']
simple_diff = df[config['All politics']] - df[config['All politics - 1']]
share_high_energy_buildings = simple_diff
share_high_energy_buildings.name = 'Share high energy building difference (%)'
result = pd.concat((result, share_high_energy_buildings), axis=1)

# Entire housing stock to the "low-energy building" level or similar by 2050
df = detailed['Stock G (Thousands)'] + detailed['Stock F (Thousands)'] + detailed['Stock E (Thousands)'] + detailed['Stock D (Thousands)'] + detailed['Stock C (Thousands)']
df.index = df.index.astype('int64')
df = detailed['Stock (Thousands)'] - df
simple_diff = df[config['All politics']] - df[config['All politics - 1']]
low_energy_buildings = simple_diff
low_energy_buildings.name = 'Low energy building difference (Thousands)'
result = pd.concat((result, low_energy_buildings), axis=1)

df = df / detailed['Stock (Thousands)']
simple_diff = df[config['All politics']] - df[config['All politics - 1']]
share_low_energy_buildings = simple_diff
share_low_energy_buildings.name = 'Share low energy building difference (%)'
result = pd.concat((result, share_low_energy_buildings), axis=1)

# Reducing fuel poverty by 15% by 2020
df = summaries['Energy poverty'] / 10**3
simple_diff = df[config['All politics']] - df[config['All politics - 1']]
energy_poverty_buildings = simple_diff
energy_poverty_buildings.name = 'Energy poverty difference (Thousands)'
result = pd.concat((result, energy_poverty_buildings), axis=1)

# Share of 'fuel poverty' buildings in the total buildings stock
df = df / summaries['Stock']
simple_diff = df[config['All politics']] - df[config['All politics - 1']]
share_energy_poverty_buildings = simple_diff
share_energy_poverty_buildings.name = 'Share energy poverty difference (%)'
result = pd.concat((result, share_energy_poverty_buildings), axis=1)

print('break')

print('break')
