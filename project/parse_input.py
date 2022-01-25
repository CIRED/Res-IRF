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

import pandas as pd
import os
import json
import numpy as np
from utils import apply_linear_rate, reindex_mi, add_level


def dict2series(item_dict):
    """Return pd.Series from a dict containing val and index attributes.
    """
    if len(item_dict['index']) == 1:
        ds = pd.Series(item_dict['val'])
    elif len(item_dict['index']) == 2:
        ds = {(outerKey, innerKey): values for outerKey, innerDict in item_dict['val'].items() for
              innerKey, values in innerDict.items()}
        ds = pd.Series(ds)
    elif len(item_dict['index']) == 3:
        ds = pd.DataFrame({k: pd.DataFrame(item).stack() for k, item in item_dict['val'].items()}).stack()
        ds.index.names = [i for i in item_dict['index'][::-1]]
        ds = ds.reorder_levels(item_dict['index'])
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
        return json_dict['source']
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

    Parameters
    ----------
    pop_housing_prev: int
    pop_housing_min: int
    pop_housing_ini: int
    factor: int

    Returns
    -------
    int
    """
    eps_pop_housing = (pop_housing_prev - pop_housing_min) / (
            pop_housing_ini - pop_housing_min)
    eps_pop_housing = max(0, min(1, eps_pop_housing))
    factor_pop_housing = factor * eps_pop_housing
    return max(pop_housing_min, pop_housing_prev * (1 + factor_pop_housing))


def to_share_multi_family_tot(stock_needed, param):
    """Calculate share of multi-family buildings in the total stock.

    In Res-IRF 2.0, the share of single- and multi-family dwellings was held constant in both existing and new
    housing stocks, but at different levels; it therefore evolved in the total stock by a simple composition
    effect. These dynamics are now more precisely parameterized in Res-IRF 3.0 thanks to recent empirical
    work linking the increase in the share of multi-family housing in the total stock to the rate of growth of
    the total stock housing growth (Fisch et al., 2015).
    This relationship in particular reflects urbanization effects.

    Parameters
    ----------
    stock_needed: pd.Series
    param: float

    Returns
    -------
    dict
        Dictionary with year as keys and share of multi-family in the total stock as value.
        {2012: 0.393, 2013: 0.394, 2014: 0.395}
    """

    def func(stock, stock_ini, p):
        """Share of multi-family dwellings as a function of the growth rate of the dwelling stock.

        Parameters
        ----------
        stock: float
        stock_ini: float
        p: float

        Returns
        -------
        float
        """
        trend_housing = (stock - stock_ini) / stock_ini * 100
        share = 0.1032 * np.log(10.22 * trend_housing / 10 + 79.43) * p
        return share

    share_multi_family_tot = {}
    stock_needed_ini = stock_needed.iloc[0]
    for year in stock_needed.index:
        share_multi_family_tot[year] = func(stock_needed.loc[year], stock_needed_ini, param)

    return pd.Series(share_multi_family_tot)


def forecast2myopic(forecast_price, yr):
    """Returns myopic prices based on forecast prices and a year.
    """
    val = forecast_price.loc[:, yr]
    columns_year = forecast_price.columns[forecast_price.columns >= yr]
    myopic = pd.concat([val] * len(columns_year), axis=1)
    myopic.columns = columns_year
    myopic.index.set_names('Heating energy', inplace=True)
    return myopic


def parse_building_stock(config):
    """
    Parses and returns building stock and attributes to match Res-IRF input requirement.

    Parameters
    ----------
    config: dict

    Returns
    -------
    stock_ini : pd.Series
        Initial buildings stock. Attributes are stored as MultiIndex.
    attributes :  dict
        Multiple information regarding stock attributes.
        All values that could possibly taken buildings attributes.
        Conversion of attributes in numerical values (energy performance certificate, etc...).
    """

    # 1. Read building stock data

    name_file = os.path.join(os.getcwd(), config['stock_buildings']['source'])
    stock_ini = pd.read_csv(name_file, index_col=config['stock_buildings']['levels'], squeeze=True)
    calibration_year = config['stock_buildings']['year']

    # 2. Numerical value of stock attributes

    # years for input time series
    # maximum investment horizon is 30 years and model horizon is 2040. Input must be extended at least to 2070.
    last_year = 2080
    index_input_year = range(calibration_year, last_year + 1, 1)

    name_file = os.path.join(os.getcwd(), config['attributes']['source'])
    attributes = parse_json(name_file)

    attributes['attributes2income'] = attributes['attributes2income'].apply(apply_linear_rate, args=(
        config['Household income rate'], index_input_year))
    attributes['attributes2consumption_heater'] = attributes['attributes2primary_consumption'] * attributes[
        'attributes2heater']
    attributes['attributes2consumption'] = final2consumption(attributes['attributes2consumption_heater'],
                                                             attributes['attributes2final_energy'] ** -1)
    attributes['attributes2consumption_heater_construction'] = attributes[
                                                                   'attributes2primary_consumption_construction'] * \
                                                               attributes[
                                                                   'attributes2heater_construction']
    attributes['attributes2consumption_construction'] = final2consumption(
        attributes['attributes2consumption_heater_construction'],
        attributes['attributes2final_energy'] ** -1)

    # function of config
    attributes2horizon = dict()
    attributes['attributes2horizon_heater'] = attributes['attributes2horizon_heater'][config['green_value']]
    attributes['attributes2horizon_envelope'] = attributes['attributes2horizon_envelope'][config['green_value']]
    attributes2horizon[('Energy performance',)] = attributes['attributes2horizon_envelope']
    attributes2horizon[('Heating energy',)] = attributes['attributes2horizon_heater']
    attributes['attributes2horizon'] = attributes2horizon

    attributes['attributes2discount'] = attributes['attributes2discount'][config['budget_constraint']]
    if isinstance(attributes['attributes2discount'], str):
        attributes['attributes2discount'] = pd.read_csv(attributes['attributes2discount'], index_col=[0, 1, 2], squeeze=True)

    file_dict = attributes['attributes_dict']
    keys = ['Housing type', 'Occupancy status', 'Heating energy', 'Energy performance', 'Income class']
    attributes['housing_stock_renovated'] = {key: file_dict[key] for key in keys}
    attributes['housing_stock_renovated']['Income class owner'] = file_dict['Income class']

    keys = ['Housing type', 'Occupancy status', 'Heating energy', 'Energy performance construction', 'Income class']
    attributes['housing_stock_constructed'] = {key: file_dict[key] for key in keys}
    attributes['housing_stock_constructed']['Income class owner'] = file_dict['Income class']
    attributes['housing_stock_constructed']['Energy performance'] = file_dict['Energy performance construction']
    attributes['housing_stock_constructed'].pop('Energy performance construction')

    return stock_ini, attributes


def parse_exogenous_input(config):
    """Parses prices and costs input to match Res-IRF input requirement.

    Parameters
    ----------
    config: dict

    Returns
    -------
    pd.DataFrame
        Energy prices.
    pd.DataFrame
        Energy taxes
    dict
        Investment cost.
        Keys are transition cost_envelope = cost_invest(tuple([Energy performance]).
    pd.DataFrame
        co2_tax
    pd.DataFrame
        co2_emission
    dict
        policies
    pd.DataFrame
        summary_input

    """

    calibration_year = config['stock_buildings']['year']
    last_year = 2080

    name_file = os.path.join(os.getcwd(), config['policies']['source'])
    policies = parse_json(name_file)

    if 'carbon_tax' in config.keys() and config['carbon_tax']['activated'] is True:
        carbon_tax = pd.read_csv(os.path.join(os.getcwd(), config['carbon_tax']['value']), index_col=[0])
        carbon_tax = carbon_tax.T
        carbon_tax.index.set_names('Heating energy', inplace=True)
        policies['carbon_tax']['value'] = carbon_tax * (1 + 0.2)

    if 'cee_taxes' in policies.keys() and config['cee_taxes']['activated'] is True:
        cee_tax = pd.read_csv(os.path.join(os.getcwd(), config['cee_taxes']['value']), index_col=[0])
        cee_tax.index.set_names('Heating energy', inplace=True)
        cee_tax.columns = cee_tax.columns.astype('int')
        policies['cee_taxes']['value'] = cee_tax * (1 + 0.2)

    if 'cee_subsidy' in policies.keys() and config['cee_subsidy']['activated'] is True:
        cee_subsidy = pd.read_csv(os.path.join(os.getcwd(), config['cee_subsidy']['value']), index_col=[0])
        cee_subsidy.index.set_names('Income class owner', inplace=True)
        cee_subsidy.columns = cee_subsidy.columns.astype('int')
        policies['cee_subsidy']['value'] = cee_subsidy

    if 'ma_prime_renov' in policies.keys() and config['ma_prime_renov']['activated'] is True:
        policies['ma_prime_renov']['value'] = policies['ma_prime_renov']['value'].unstack('Energy performance final').fillna(0)
        if config['stock_buildings']['income_class'] == 'quintile':
            policies['ma_prime_renov']['value'].rename(
                index={'D1': 'C1', 'D3': 'C2', 'D5': 'C3', 'D6': 'C4', 'D10': 'C5'}, inplace=True)
            policies['ma_prime_renov']['value'] = policies['ma_prime_renov']['value'][policies['ma_prime_renov']['value'].index.get_level_values('Income class owner').isin(['C1', 'C2', 'C3', 'C4', 'C5'])]


    if 'subsidies_curtailment' in policies.keys() and config['subsidies_curtailment']['activated'] is True:
        policies['subsidies_curtailment']['value'] = pd.read_csv(policies['subsidies_curtailment']['value'],
                                                                 index_col=[0], header=[0], squeeze=True)


    # cost_invest
    cost_invest = dict()
    name_file = os.path.join(os.getcwd(), config['cost_renovation']['source'])
    cost_envelope = pd.read_csv(name_file, sep=',', header=[0], index_col=[0])
    cost_envelope.index.set_names('Energy performance', inplace=True)
    cost_envelope.columns.set_names('Energy performance final', inplace=True)
    cost_envelope = cost_envelope * (1 + 0.1) / (1 + 0.055)
    cost_invest['Energy performance'] = cost_envelope

    """
    name_file = os.path.join(os.getcwd(), config['cost_switch_fuel']['source'])
    cost_switch_fuel = pd.read_csv(name_file, index_col=[0], header=[0])
    cost_switch_fuel.index.set_names('Heating energy', inplace=True)
    cost_switch_fuel.columns.set_names('Heating energy final', inplace=True)
    # cost_switch_fuel = cost_switch_fuel * (1 + 0.1) / (1 + 0.055)
    """
    cost_invest['Heating energy'] = None

    name_file = os.path.join(os.getcwd(), config['energy_prices_bt']['source'])
    energy_prices_bt = pd.read_csv(name_file, index_col=[0], header=[0]).T
    energy_prices_bt.index.set_names('Heating energy', inplace=True)
    energy_prices = energy_prices_bt

    # initialize energy taxes
    energy_taxes = energy_prices.copy()
    for col in energy_prices.columns:
        energy_taxes[col].values[:] = 0

    if config['energy_taxes']['vta']:
        vta = pd.Series([0.16, 0.16, 0.2, 0.2], index=['Power', 'Natural gas', 'Oil fuel', 'Wood fuel'])
        vta.index.set_names('Heating energy', inplace=True)
        vta_energy = (energy_prices_bt.T * vta).T
        energy_prices = energy_prices + vta_energy
        energy_taxes = energy_taxes + vta_energy

    if config['energy_taxes']['activated']:
        name_file = os.path.join(os.getcwd(), config['energy_taxes']['source'])
        energy_tax = pd.read_csv(name_file, index_col=[0], header=[0]).T
        energy_tax.index.set_names('Heating energy', inplace=True)

        # energy prices before cee and carbon tax and after vta and other energy taxes
        energy_prices = energy_prices + energy_tax
        energy_taxes = energy_taxes + energy_tax

    # extension of energy_prices time series
    last_year_prices = energy_prices.columns[-1]
    if last_year > last_year_prices:
        add_yrs = range(last_year_prices + 1, last_year + 1, 1)
        temp = pd.concat([energy_prices.loc[:, last_year_prices]] * len(add_yrs), axis=1)
        temp.columns = add_yrs
        energy_prices = pd.concat((energy_prices, temp), axis=1)

    if config['energy_prices_evolution'] == 'forecast':
        energy_prices = energy_prices.loc[:, calibration_year:]
    elif config['energy_prices_evolution'] == 'constant':
        energy_prices = pd.Series(energy_prices.loc[:, calibration_year], index=energy_prices.index)
        energy_prices.index.set_names('Heating energy', inplace=True)
        idx_yrs = range(calibration_year, last_year + 1, 1)
        energy_prices = pd.concat([energy_prices] * len(idx_yrs), axis=1)
        energy_prices.columns = idx_yrs
    else:
        raise ValueError("energy_prices_evolution should be 'forecast' or 'constant'")

    # CO2 content used for tax cost
    name_file = os.path.join(os.getcwd(), config['carbon_tax']['co2_content'])
    co2_tax = pd.read_csv(name_file, index_col=[0], header=[0]).T
    co2_tax.index.set_names('Heating energy', inplace=True)
    # extension of co2_content time series
    last_year_prices = co2_tax.columns[-1]
    if last_year > last_year_prices:
        add_yrs = range(last_year_prices + 1, last_year + 1, 1)
        temp = pd.concat([co2_tax.loc[:, last_year_prices]] * len(add_yrs), axis=1)
        temp.columns = add_yrs
        co2_tax = pd.concat((co2_tax, temp), axis=1)
    co2_tax = co2_tax.loc[:, calibration_year:]

    # CO2 content used for emission
    name_file = os.path.join(os.getcwd(), config['co2_emission']['source'])
    co2_emission = pd.read_csv(name_file, index_col=[0], header=[0]).T
    co2_emission.index.set_names('Heating energy', inplace=True)

    # extension of co2_content time series
    last_year_prices = co2_emission.columns[-1]
    if last_year > last_year_prices:
        add_yrs = range(last_year_prices + 1, last_year + 1, 1)
        temp = pd.concat([co2_emission.loc[:, last_year_prices]] * len(add_yrs), axis=1)
        temp.columns = add_yrs
        co2_emission = pd.concat((co2_emission, temp), axis=1)
    co2_emission = co2_emission.loc[:, calibration_year:]

    summary_input = dict()

    summary_input['Power prices (euro/kWh)'] = energy_prices.loc['Power', :]
    summary_input['Natural gas prices (euro/kWh)'] = energy_prices.loc['Natural gas', :]
    summary_input['Oil fuel prices (euro/kWh)'] = energy_prices.loc['Oil fuel', :]
    summary_input['Wood fuel prices (euro/kWh)'] = energy_prices.loc['Wood fuel', :]

    summary_input['Power emission (gCO2/kWh)'] = co2_emission.loc['Power', :]
    summary_input['Natural gas emission (gCO2/kWh)'] = co2_emission.loc['Natural gas', :]
    summary_input['Oil fuel emission (gCO2/kWh)'] = co2_emission.loc['Oil fuel', :]
    summary_input['Wood fuel (gCO2/kWh)'] = co2_emission.loc['Wood fuel', :]

    summary_input = pd.DataFrame(summary_input)
    summary_input = summary_input.loc[calibration_year:, :]

    return energy_prices, energy_taxes, cost_invest, co2_tax, co2_emission, policies, summary_input


def parse_parameters(folder, config, stock_sum):
    """
    Parse input that are not implicitly subject to a scenario.

    Parameters
    ----------
    folder : str
        Folder where to look for input files.
    config: dict
    stock_sum : float
        Number of buildings in stock data.

    Returns
    -------
    dict
        Mainly contains demographic and macro-economic variables. Also contains function parameter (lbd).
    pd.DataFrame
    """

    # years for input time series
    calibration_year = config['stock_buildings']['year']

    last_year = 2080
    index_input_year = range(calibration_year, last_year + 1, 1)

    # 1. Parameters
    name_file = os.path.join(os.getcwd(), config['parameters']['source'])
    parameters = parse_json(name_file)

    # 2. Demographic variables

    name_file = os.path.join(os.getcwd(), config['population']['source'])
    parameters['Population total'] = pd.read_csv(os.path.join(folder, name_file), sep=',', header=None,
                                                 index_col=[0],
                                                 squeeze=True)

    # sizing_factor < 1 --> all extensive results are calibrated by the size of the initial parc
    sizing_factor = stock_sum / parameters['Stock total ini {}'.format(calibration_year)]
    parameters['Sizing factor'] = sizing_factor
    parameters['Population'] = parameters['Population total'] * sizing_factor

    if parameters['Stock needed']['source_type'] == 'function':
        population_housing_min = parameters['Stock needed']['Population housing min']
        population_housing = dict()
        population_housing[calibration_year] = parameters['Population'].loc[calibration_year] / stock_sum
        max_year = max(parameters['Population'].index)

        stock_needed = dict()
        stock_needed[calibration_year] = parameters['Population'].loc[calibration_year] / population_housing[
            calibration_year]

        for year in index_input_year[1:]:
            if year > max_year:
                break
            population_housing[year] = population_housing_dynamic(population_housing[year - 1],
                                                                  population_housing_min,
                                                                  population_housing[calibration_year],
                                                                  parameters['Stock needed']['Factor population housing ini'])
            stock_needed[year] = parameters['Population'].loc[year] / population_housing[year]

        parameters['Population housing'] = pd.Series(population_housing)
        parameters['Stock needed'] = pd.Series(stock_needed)
    elif parameters['Stock needed']['source_type'] == 'file':
        parameters['Stock needed'] = pd.read_csv(parameters['Stock needed']['source'], index_col=[0], header=[0])
        parameters['Population housing'] = parameters['Population'] / parameters['Stock needed']
    else:
        raise ValueError('Stock needed source_type must be defined as a function or a file')

    if parameters['Share multi-family']['source_type'] == 'function':
        parameters['Share multi-family'] = to_share_multi_family_tot(parameters['Stock needed'], parameters['Share multi-family']['factor'])
    elif parameters['Share multi-family']['source_type'] == 'file':
        parameters['Share multi-family'] = pd.read_csv(parameters['Share multi-family']['source'], index_col=[0], header=None, squeeze=True)
    else:
        raise ValueError('Share multi-family source_type must be defined as a function or a file')

    # 5. Macro-economic variables
    parameters['Available income'] = apply_linear_rate(parameters['Available income ini {}'.format(calibration_year)],
                                                       parameters['Available income rate'], index_input_year)

    # inflation
    parameters['Price index'] = pd.Series(1, index=index_input_year)
    parameters['Available income real'] = parameters['Available income'] / parameters['Price index']
    parameters['Available income real population'] = parameters['Available income real'] / parameters[
        'Population total']

    # 6. Others

    parameters['Health cost (euro)'] = pd.read_csv(config['health_cost']['source'], index_col=[0, 1], squeeze=True)
    parameters['Carbon value (euro/tCO2)'] = pd.read_csv(config['carbon_value']['source'], index_col=[0], header=None,
                                                         squeeze=True)
    parameters['Demolition rate'] = config['Demolition rate']

    if 'Rotation rate' in config.keys():
        parameters['Rotation rate'] = config['Rotation rate']

    # 6. Summary

    summary_param = dict()
    summary_param['Sizing factor (%)'] = pd.Series(sizing_factor, index=parameters['Population'].index)
    summary_param['Total population (Millions)'] = parameters['Population'] / 10**6
    summary_param['Income (Billions euro)'] = parameters['Available income real'] * sizing_factor / 10**9
    summary_param['Buildings stock (Millions)'] = parameters['Stock needed'] / 10**6
    summary_param['Person by housing'] = parameters['Population housing']
    summary_param['Share multi-family (%)'] = parameters['Share multi-family']
    summary_param = pd.DataFrame(summary_param)
    summary_param = summary_param.loc[calibration_year:, :]

    return parameters, summary_param


def parse_observed_data(config):
    """Parses and returns observed data to match Res-IRF input requirement.

    Parameters
    ----------
    config: dict

    Returns
    -------
    pd.DataFrame
        Observed renovation rate in calibration year
    pd.DataFrame
        Observed market share in calibration year for existing buildings
    pd.DataFrame
        Observed market share in calibration year for switching fuel
    """

    name_file = os.path.join(os.getcwd(), config['renovation_rate']['calibration_data'])
    renovation_rate_ini = pd.read_csv(name_file, header=[0], squeeze=True)
    columns = list(renovation_rate_ini.columns[:renovation_rate_ini.shape[1] - 1])
    renovation_rate_ini = renovation_rate_ini.set_index(columns).iloc[:, 0]

    name_file = os.path.join(os.getcwd(), config['market_share']['calibration_data'])
    ms_renovation_ini = pd.read_csv(name_file, index_col=[0], header=[0])
    ms_renovation_ini.index.set_names(['Energy performance'], inplace=True)
    ms_renovation_ini.columns.set_names(['Energy performance final'], inplace=True)

    name_file = os.path.join(os.getcwd(), config['ms_switch_fuel_ini']['source'])
    ms_switch_fuel_ini = pd.read_csv(name_file, index_col=[0, 1], header=[0])
    ms_switch_fuel_ini.index.set_names(['Housing type', 'Heating energy'], inplace=True)
    ms_switch_fuel_ini.columns.set_names(['Heating energy final'], inplace=True)

    return renovation_rate_ini, ms_renovation_ini, ms_switch_fuel_ini


