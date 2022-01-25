# Copyright 2020-2021 Ecole Nationale des Ponts et Chaussées
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Original author Lucas Vivier <vivier@centre-cired.fr>
# Based on a scilab program mainly by written by L.G Giraudet and others, but fully rewritten.

import re
import argparse
import os
import pandas as pd
import ui_utils


def double_diff(df_ref, df_compare, discount_factor):
    """Calculate double difference.

    Double difference is a proxy of marginal flow produced by year.
    Flow have effect during a long period of time and need to be extended during the period.

    Parameters
    ----------
    df_ref: pd.Series
    df_compare: pd.Series
    discount_factor: float

    Returns
    -------
    pd.Series
    """

    simple_diff = df_ref - df_compare
    double_diff = simple_diff.diff()
    double_diff.iloc[0] = simple_diff.iloc[0]
    discounted_diff = double_diff * discount_factor
    return discounted_diff.cumsum()


def run_indicators(config, folder, discount_rate=0.04, lifetime=26):
    """Calculate main indicators to assess public policy.

    Parameters
    ----------
    config: dict
    folder: str
    discount_rate: float
        Discount rate to extend energy and emission saving
    lifetime : int
        Lifetime to extend energy and emission saving

    Returns
    -------
    """
    config = {key: item for key, item in config.items() if item is not None}
    list_years = [int(re.search('20[0-9][0-9]', key)[0]) for key in config.keys() if re.search('20[0-9][0-9]', key)]
    temp = ['Policy - {}'.format(year) for year in list_years]
    for key, item in config.items():
        if key in ['All policies', 'All policies - 1', 'Zero policies', 'Zero policies + 1'] or key in temp:
            config[key] = item.replace(' ', '_')

    if 'Discount rate' in config.keys():
        discount_rate = float(config['Discount rate'])

    if 'Lifetime' in config.keys():
        lifetime = int(config['Lifetime'])

    # Discount factor to extend energy and emission saving
    discount_factor = (1 - (1 + discount_rate) ** -lifetime) / discount_rate

    folder_output = os.path.join(os.getcwd(), 'project', 'output', folder)
    scenarios = [f for f in os.listdir(folder_output) if os.path.isdir(os.path.join(folder_output, f)) and f != 'img']
    folders = {scenario: os.path.join(folder_output, scenario) for scenario in scenarios}

    detailed = {scenario: pd.read_csv(os.path.join(folders[scenario], 'detailed.csv'), index_col=[0]).T for scenario in
                scenarios}
    detailed = ui_utils.reverse_nested_dict(detailed)
    detailed = {key: pd.DataFrame(item) for key, item in detailed.items()}

    for _, df in detailed.items():
        df.index = df.index.astype('int64')


    if list_years != []:
        # 1. Efficiency
        # 1.1 Cost-effectiveness

        # 1.1.1 Consumption
        df = detailed['Consumption actual (TWh/year)'].copy()
        marginal_consumption_actual = pd.Series(
            [(df[config['All policies']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years],
            index=list_years) * 10**9

        df = detailed['Consumption conventional (TWh/year)'].copy()
        marginal_consumption_conventional = pd.Series(
            [(df[config['All policies']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years],
            index=list_years) * 10**9

        # 1.1.2 Emission
        df = detailed['Emission (MtCO2/year)'].copy()
        marginal_emission = pd.Series(
            [(df[config['All policies']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years],
            index=list_years) * 10**6

        # 1.1.3 Policy cost

        df = detailed['{} (euro)'.format(config['Policy name']).replace('_', ' ').capitalize()].copy()
        df.fillna(0, inplace=True)
        marginal_subsidies = pd.Series(
            [(df[config['All policies']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years],
            index=list_years)

        # 1.1.4 Results
        cost_effectiveness = abs(pd.concat((marginal_subsidies / (marginal_consumption_actual * discount_factor),
                                           marginal_subsidies / (marginal_consumption_conventional * discount_factor),
                                           marginal_subsidies / (marginal_emission * discount_factor)),
                                           axis=1))
        cost_effectiveness.columns = ['Cost-effectiveness actual (euro/kWh)',
                                      'Cost-effectiveness conventional (euro/kWh)',
                                      'Cost-effectiveness emission (euro/tCO2)']
        cost_effectiveness.sort_index(inplace=True)

        # 2.1 Leverage effect
        df = detailed['Annual renovation expenditure (Billions euro)'].copy()
        df.fillna(0, inplace=True)
        marginal_investment = pd.Series(
            [(df[config['All policies']] - df[config['Policy - {}'.format(year)]]).loc[year] for year in list_years],
            index=list_years) * 10**9

        leverage = marginal_investment / marginal_subsidies
        leverage.sort_index(inplace=True)
        leverage.name = 'Leverage'

        pd.concat((cost_effectiveness, leverage), axis=1).to_csv(os.path.join(folder_output, 'indicator_policies.csv'))

    # 2. Effectiveness
    # 2.1 Consumption actual
    # 2.1.1 Reduction of final energy consumption by 20% by 2030 compared to 2012
    # 2.1.2 Reduction of 50% by 2050 compared to 2012

    methods = {'AP': ('All policies', 'All policies - 1'),
               'ZP': ('Zero policies', 'Zero policies + 1')}

    result = pd.DataFrame()
    for name, method in methods.items():

        ref = config[method[0]]
        compare = config[method[1]]

        df = detailed['Consumption actual (TWh/year)']
        simple_diff = df[ref] - df[compare]
        double_diff = simple_diff.diff()
        double_diff.iloc[0] = simple_diff.iloc[0]
        energy_saving = double_diff * discount_factor
        energy_saving = energy_saving.cumsum()
        energy_saving.name = 'Energy difference discounted (TWh) {}'.format(name)
        result = pd.concat((result, energy_saving), axis=1)

        df = detailed['Emission (MtCO2/year)']
        simple_diff = df[ref] - df[compare]
        double_diff = simple_diff.diff()
        double_diff.iloc[0] = simple_diff.iloc[0]
        emission_saving = double_diff * discount_factor
        emission_saving = emission_saving.cumsum()
        emission_saving.name = 'Emission difference discounted (MtCO2) {}'.format(name)
        result = pd.concat((result, emission_saving), axis=1)

        # 2.2 Renovation by year
        # 2. Energy renovation of 500,000 homes per year, including 120,000 in social housing;

        df = detailed['Flow renovation (Thousands)']
        simple_diff = df[ref] - df[compare]
        additional_renovation = simple_diff.cumsum()
        additional_renovation.name = 'Renovation difference (Thousands) {}'.format(name)
        result = pd.concat((result, additional_renovation), axis=1)

        # G and F buildings in 2025
        df = detailed['Stock F (Thousands)'] + detailed['Stock G (Thousands)']
        simple_diff = df[ref] - df[compare]
        high_energy_buildings = simple_diff
        high_energy_buildings.name = 'High energy building difference (Thousands)'
        result = pd.concat((result, high_energy_buildings), axis=1)

        # Share of F and G buildings in the total buildings stock
        df = df / detailed['Stock (Thousands)']
        simple_diff = df[ref] - df[compare]
        share_high_energy_buildings = simple_diff
        share_high_energy_buildings.name = 'Share high energy building difference (%) {}'.format(name)
        result = pd.concat((result, share_high_energy_buildings), axis=1)

        # Entire housing stock to the "low-energy building" level or similar by 2050
        df = detailed['Stock G (Thousands)'] + detailed['Stock F (Thousands)'] + detailed['Stock E (Thousands)'] + \
             detailed['Stock D (Thousands)'] + detailed['Stock C (Thousands)']
        df = detailed['Stock (Thousands)'] - df
        simple_diff = df[ref] - df[compare]
        low_energy_buildings = simple_diff
        low_energy_buildings.name = 'Low energy building difference (Thousands) {}'.format(name)
        result = pd.concat((result, low_energy_buildings), axis=1)

        df = df / detailed['Stock (Thousands)']
        simple_diff = df[ref] - df[compare]
        share_low_energy_buildings = simple_diff
        share_low_energy_buildings.name = 'Share low energy building difference (%) {}'.format(name)
        result = pd.concat((result, share_low_energy_buildings), axis=1)

        # Reducing fuel poverty by 15% by 2020
        df = detailed['Energy poverty (Thousands)']
        simple_diff = df[ref] - df[compare]
        energy_poverty_buildings = simple_diff
        energy_poverty_buildings.name = 'Energy poverty difference (Thousands) {}'.format(name)
        result = pd.concat((result, energy_poverty_buildings), axis=1)

        # Share of 'fuel poverty' buildings in the total buildings stock
        df = df / detailed['Stock (Thousands)']
        simple_diff = df[ref] - df[compare]
        share_energy_poverty_buildings = simple_diff
        share_energy_poverty_buildings.name = 'Share energy poverty difference (%) {}'.format(name)
        result = pd.concat((result, share_energy_poverty_buildings), axis=1)

    result.to_csv(os.path.join(folder_output, 'policies_effectiveness.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=False, help='config file path')
    args = parser.parse_args()

    config_policies = pd.read_csv(os.path.join('project', args.config), squeeze=True, header=None, index_col=[0])
    config_policies = config_policies.dropna()
    carbon_value = pd.read_csv('project/input/policies/carbon_value.csv', index_col=[0], squeeze=True, header=None)
    run_indicators(config_policies.to_dict(), config_policies['Folder name'], carbon_value)
