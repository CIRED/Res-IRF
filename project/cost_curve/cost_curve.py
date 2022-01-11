
import os
import pandas as pd
import argparse
import json
from multiprocessing import Process
import datetime
from itertools import product

from buildings import HousingStock
from parse_input import parse_json, apply_linear_rate, final2consumption
from utils import reindex_mi
from ui_abattement_curve import *


def prepare_input(config):
    """Returns input used to create the abattement curve.

    Parameters
    ----------
    config: dict

    Returns
    -------
    """

    name_file = os.path.join(os.getcwd(), config['stock_buildings']['source'])
    stock_ini = pd.read_pickle(name_file)
    stock_ini = stock_ini.groupby([i for i in stock_ini.index.names if i != 'Income class owner']).sum()

    name_file = os.path.join(os.getcwd(), config['attributes']['source'])
    attributes = parse_json(name_file)

    last_year = 2080
    index_input_year = range(config['calibration_year'], last_year + 1, 1)
    attributes['attributes2income'] = attributes['attributes2income'].apply(apply_linear_rate, args=(
        config['Household income rate'], index_input_year))

    attributes['attributes2consumption_heater'] = attributes['attributes2primary_consumption'] * attributes[
        'attributes2heater']
    attributes['attributes2consumption'] = final2consumption(attributes['attributes2consumption_heater'],
                                                             attributes['attributes2final_energy'] ** -1)

    # cost_invest
    cost_invest = dict()
    name_file = os.path.join(os.getcwd(), config['cost_renovation']['source'])
    cost_envelope = pd.read_csv(name_file, sep=',', header=[0], index_col=[0])
    cost_envelope.index.set_names('Energy performance', inplace=True)
    cost_envelope.columns.set_names('Energy performance final', inplace=True)
    cost_envelope = cost_envelope / (1 + 0.055)
    cost_invest['Energy performance'] = cost_envelope * (1 + config['investment_cost_factor'])

    name_file = os.path.join(os.getcwd(), config['cost_switch_fuel']['source'])
    cost_switch_fuel = pd.read_csv(name_file, index_col=[0], header=[0])
    cost_switch_fuel.index.set_names('Heating energy', inplace=True)
    cost_switch_fuel.columns.set_names('Heating energy final', inplace=True)
    cost_switch_fuel = cost_switch_fuel / (1 + 0.055)
    cost_invest['Heating energy'] = cost_switch_fuel

    name_file = os.path.join(os.getcwd(), config['energy_prices_bt']['source'])
    energy_prices_bt = pd.read_csv(name_file, index_col=[0], header=[0]).T
    energy_prices_bt.index.set_names('Heating energy', inplace=True)

    name_file = os.path.join(os.getcwd(), config['energy_prices_at']['source'])
    energy_prices_at = pd.read_csv(name_file, index_col=[0], header=[0]).T
    energy_prices_at.index.set_names('Heating energy', inplace=True)

    name_file = os.path.join(os.getcwd(), config['energy_taxes']['source'])
    energy_taxes = pd.read_csv(name_file, index_col=[0], header=[0]).T
    energy_taxes.index.set_names('Heating energy', inplace=True)

    name_file = os.path.join(os.getcwd(), config['co2_emission']['source'])
    co2_emission = pd.read_csv(name_file, index_col=[0], header=[0]).T
    co2_emission.index.set_names('Heating energy', inplace=True)

    name_file = os.path.join(os.getcwd(), config['carbon_value']['source'])
    carbon_value = pd.read_csv(name_file, index_col=[0], header=[0], squeeze=True)

    carbon_rate = -carbon_value.diff(-1) / carbon_value

    name_file = os.path.join(os.getcwd(), config['health_cost']['source'])
    health_cost = pd.read_csv(name_file, index_col=[0, 1], header=[0], squeeze=True)

    return stock_ini, attributes, cost_invest, energy_prices_bt, energy_prices_at, energy_taxes, co2_emission, carbon_value, health_cost, carbon_rate


def select_final_state(df, energy_performance, dict_replace=None):
    """Returns series with same heating energy than initial state

    Parameters
    ----------
    df: pd.DataFrame
    energy_performance: str
    dict_replace: dict

    Returns
    -------
    pd.Series
    """
    ds_status_quo = pd.Series(dtype='float64')
    for val in df.index.get_level_values('Heating energy').unique():
        val_out = val
        if dict_replace is not None:
            if val in dict_replace.keys():
                val_out = dict_replace[val]

        temp = df[df.index.get_level_values('Heating energy') == val].loc[:, (energy_performance, val_out)]
        ds_status_quo = ds_status_quo.append(temp)

    ds_status_quo.index = pd.MultiIndex.from_tuples(ds_status_quo.index)
    ds_status_quo.index.names = df.index.names
    return ds_status_quo


def to_result(carbon_cost, emission_final_end, emission_trend_end, potential_emission_saving, emission_ini, stock,
              path, yr, name='', dict_replace=None, energy_performance='A', horizon=30, private_carbon_cost=None,
              health_carbon_cost=None, lost_carbon_cost=None):
    """Formatting results.

    Parameters
    ----------
    carbon_cost: pd.DataFrame
        Social carbon cost by agent archetype and possible final state.
    emission_final_end: pd.DataFrame
        Emission (tCO2) by agent archetype in the prospective scenario scenario to the horizon (final year) and
        possible final state.
    emission_trend_end: pd.Series
        Emission (tCO2) by agent archetype in the trend scenario scenario to the horizon (final year).
    potential_emission_saving
        Emission (tCO2) by agent archetype in the trend scenario scenario to the horizon.
    emission_ini
        Initial Emission (tCO2) by agent archetype (first year).
    stock: pd.Series
    path: str
        Scenario folder path.
    name: str
        Energy transition name.
    dict_replace: dict, None
        Energy transition scenario.
    energy_performance: {'A', 'B', 'C', 'D', 'E', 'F', 'G'}
    horizon: int
    private_carbon_cost : pd.DataFrame, optional
    health_carbon_cost : pd.DataFrame, optional
    lost_carbon_cost : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame
    """

    name = '{}_{}_{}.csv'.format(energy_performance, name.lower(), yr)
    path = os.path.join(path, name)

    emission_final_end = select_final_state(emission_final_end, energy_performance, dict_replace=dict_replace)

    # gCO2/m2 -> tCO2
    emission_final_end_ref = emission_final_end.sum()
    emission_diff = (emission_ini - emission_final_end) / 10**6
    emission_total_reference = emission_ini.sum() / 10**6

    output = dict()
    output['Carbon cost (euro/tCO2)'] = select_final_state(carbon_cost, energy_performance, dict_replace=dict_replace)

    if private_carbon_cost is not None:
        output['Private carbon cost (euro/tCO2)'] = select_final_state(private_carbon_cost, energy_performance,
                                                                       dict_replace=dict_replace)

    if health_carbon_cost is not None:
        output['Health carbon cost (euro/tCO2)'] = select_final_state(health_carbon_cost, energy_performance,
                                                                      dict_replace=dict_replace)

    if lost_carbon_cost is not None:
        output['Opportunity carbon cost (euro/tCO2)'] = select_final_state(lost_carbon_cost, energy_performance,
                                                                           dict_replace=dict_replace)

    output['Potential emission saving (tCO2)'] = select_final_state(potential_emission_saving, energy_performance, dict_replace=dict_replace)
    output['Potential emission saving (tCO2/yr)'] = output['Potential emission saving (tCO2)'] / horizon
    output['Emission difference (tCO2/yr)'] = emission_diff
    output['Emission final horizon (tCO2/yr)'] = emission_final_end
    output['Emission trend horizon (tCO2/yr)'] = emission_trend_end
    output['Dwelling number'] = stock

    output = pd.DataFrame(output)

    # output = output[output['Potential emission saving (tCO2)'] > 0]
    output.loc[output['Potential emission saving (tCO2)'] <= 0, 'Carbon cost (euro/tCO2)'] = float('nan')

    output = output.reorder_levels(
        ['Occupancy status', 'Housing type', 'Income class', 'Energy performance', 'Heating energy'])
    output = output.sort_values('Carbon cost (euro/tCO2)')

    output['Cumulated potential emission saving (tCO2/yr)'] = output['Potential emission saving (tCO2/yr)'].cumsum()
    output['Cumulated potential emission saving (%)'] = output[
                                                            'Cumulated potential emission saving (tCO2)'] / emission_total_reference

    output['Cumulated emission difference (tCO2/yr)'] = output['Emission difference (tCO2/yr)'].cumsum()
    output['Cumulated emission difference (%)'] = output[
                                                      'Cumulated emission difference (tCO2)'] / emission_total_reference

    output['Cumulated dwelling number'] = output['Dwelling number'].cumsum()
    output['Cumulated dwelling number (%)'] = output['Cumulated dwelling number'] / stock.sum()

    output.to_csv(path)
    return output


def run_yrs(config, path):

    if config['carbon_cost_formula'] == 'social_carbon_value':
        idx_yr = range(config['calibration_year'], config['social_carbon_value'])
    else:
        idx_yr = [config['calibration_year']]

    for yr in idx_yr:
        print(yr)
        to_carbon_cost(config, path, yr)


def to_carbon_cost(config, path, calibration_year):
    """
    Calculate CO2 cost and potential emission saving for building stock.

    Potential emission saving represent emission difference flow between a scenario and a baseline scenario.

    Parameters
    ----------
    config: dict
        Configuration file to prepare every input.
    path: str
        Path

    Returns
    -------
    pd.DataFrame
    """

    discount_rate = config['discount_rate']
    investment_tax_rate = config['investment_tax_rate']
    cofp_factor = config['cofp_factor']
    transition = ['Energy performance', 'Heating energy']
    consumption_type = config['consumption_type']
    horizon = config['horizon']

    stock_ini, attributes, cost_invest, energy_prices_bt, energy_prices_at, energy_taxes, co2_emission, carbon_value, health_cost, carbon_rate = prepare_input(config)

    index_year = range(calibration_year, calibration_year + horizon)
    discount_factor = (1 - (1 + discount_rate) ** -horizon) / discount_rate
    emission_discount_rate = discount_rate - carbon_rate
    emission_discount_rate = emission_discount_rate.reindex(index_year)
    emission_discount_rate.loc[2040:] = 0

    buildings = HousingStock(stock_ini, attributes['attributes_dict'], calibration_year,
                             attributes2area=attributes['attributes2area'],
                             attributes2horizon=attributes['attributes2horizon'],
                             attributes2discount=attributes['attributes2discount'],
                             attributes2income=attributes['attributes2income'],
                             attributes2consumption=attributes['attributes2consumption'],
                             price_behavior=None
                             )

    # kWh/m2 * m2/building * buildings = kWh -> TWh
    consumption = buildings.to_consumption(consumption_type, energy_prices=energy_prices_at)
    if isinstance(consumption, pd.DataFrame):
        consumption.columns.names = [None]

    output = dict()
    output['Stock total (Thousands)'] = stock_ini.sum() / 10**3

    if consumption_type == 'conventional':
        consumption = pd.concat([consumption] * len(index_year), axis=1)
        consumption.columns = index_year

    energy_reference = consumption.loc[:, calibration_year] * buildings.stock * buildings.to_area()
    output['Energy consumption initial (TWh)'] = energy_reference.sum() / 10**9

    # gCO2/kWh * kWh -> gCO2 -> tCO2
    emission_ini = reindex_mi(co2_emission, energy_reference.index).loc[:, calibration_year] * energy_reference
    output['CO2 emission initial (MtCO2)'] = emission_ini.sum() / 10**12

    # lcc €/m2
    lcc_final = buildings.to_lcc_final(energy_prices_bt, cost_invest=cost_invest, transition=transition,
                                       consumption=consumption_type)
    energy_lcc = buildings.to_energy_lcc(energy_prices_bt, transition=transition, consumption=consumption_type)
    lcc_saving = lcc_final.apply(lambda x: x - energy_lcc, axis=0)

    # health_cost €/building -> €/m2
    health_cost_initial = reindex_mi(health_cost, buildings.stock.index)
    health_cost_initial_lc = health_cost_initial * discount_factor
    health_cost_final_lc = buildings.to_final(health_cost_initial_lc, transition=transition)
    health_cost_diff_lc = health_cost_final_lc.apply(lambda x: x - health_cost_initial_lc, axis=0)
    health_cost_diff_lc = (health_cost_diff_lc.T / buildings.to_area()).T

    # €/m2 - additional tax
    capex = reindex_mi(cost_invest['Energy performance'], lcc_final.index)
    capex = reindex_mi(capex, lcc_final.columns, axis=1)
    capex_switch = reindex_mi(cost_invest['Heating energy'], lcc_final.index)
    capex_switch = reindex_mi(capex_switch, lcc_final.columns, axis=1)
    investment_tax = (capex_switch + capex) * investment_tax_rate

    # kWh/m2 * €/kWh = €/m2
    energy_taxes_re = reindex_mi(energy_taxes, consumption.index)
    energy_tax = (consumption * energy_taxes_re).loc[:, index_year]
    energy_tax_lc = HousingStock.to_discounted(energy_tax, discount_rate).sum(axis=1)
    energy_tax_final_lc = buildings.to_final(energy_tax_lc, transition=transition)
    energy_tax_diff_lc = energy_tax_final_lc.apply(lambda x: x - energy_tax_lc, axis=0)

    # €/m2
    lost_revenue_lc = - (investment_tax + energy_tax_diff_lc) * cofp_factor

    # €/m2
    investment_cost = capex_switch + capex
    energy_cost = lcc_saving - investment_cost

    social_cost = lcc_saving + health_cost_diff_lc + lost_revenue_lc

    # test kWh/m2 * gCO2/kWh = gCO2/m2
    co2_content_re = reindex_mi(co2_emission, consumption.index)
    emission = (consumption * co2_content_re).loc[:, index_year]
    emission_lc = HousingStock.to_discounted(emission, 0).sum(axis=1)
    emission_final_lc = buildings.to_final(emission_lc, transition=transition)
    emission_diff_lc = emission_final_lc.apply(lambda x: x - emission_lc, axis=0)
    emission_saving_lc = - emission_diff_lc

    # gCO2/m2 -> tCO2
    potential_emission_saving = (emission_saving_lc.T * (buildings.to_area() * buildings.stock)).T / 10**6

    emission_cost_saving_lc = emission_saving_lc.copy()
    if config['carbon_cost_formula'] == 'social_carbon_value':
        emission_discount_rate = emission_discount_rate.loc[index_year]
        discount = pd.Series(
            [(1 + rate) ** -(yr - calibration_year) for yr, rate in emission_discount_rate.iteritems()],
            index=index_year)
        emission_cost_lc = (emission * discount).sum(axis=1)
        emission_cost_final_lc = buildings.to_final(emission_cost_lc, transition=transition)
        emission_cost_diff_lc = emission_cost_final_lc.apply(lambda x: x - emission_lc, axis=0)
        emission_cost_saving_lc = - emission_cost_diff_lc

    # €/gCO2 -> €/tCO2
    carbon_cost = (social_cost / emission_cost_saving_lc) * 10**6
    private_carbon_cost = (lcc_saving / emission_cost_saving_lc) * 10**6
    health_carbon_cost = (health_cost_diff_lc / emission_cost_saving_lc) * 10**6
    lost_carbon_cost = (lost_revenue_lc / emission_cost_saving_lc) * 10**6

    # gCO2/m2 -> tCO2
    emission_final = buildings.to_final(emission, transition=transition)
    emission_final = (emission_final.T * (buildings.stock * buildings.to_area())).T

    for t in transition:
        emission_final = emission_final.stack('{} final'.format(t))
    emission_final_end = emission_final.iloc[:, -1]
    for t in transition:
        emission_final_end = emission_final_end.unstack('{} final'.format(t))

    # emission flow during last year on the baseline scenario
    emission_trend_end = emission.iloc[:, -1] * buildings.stock * buildings.to_area()

    list_energy_transition = [
                              {'2Power': {'Natural gas': 'Power', 'Oil fuel': 'Power', 'Wood fuel': 'Power'}},
                              {'Oil2Power': {'Oil fuel': 'Power'}}]

    list_performance_transition = ['A', 'B', 'C']

    list_transition = list(product(list_performance_transition, list_energy_transition))

    for performance_transition, energy_transition in list_transition:

        name = list(energy_transition.keys())[0]

        to_result(carbon_cost, emission_final_end, emission_trend_end, potential_emission_saving, emission_ini,
                  buildings.stock, path, calibration_year, name=name, dict_replace=energy_transition[name],
                  energy_performance=performance_transition, horizon=horizon,
                  private_carbon_cost=private_carbon_cost, health_carbon_cost=health_carbon_cost,
                  lost_carbon_cost=lost_carbon_cost)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default=False, help='name scenarios')

    args = parser.parse_args()

    name_file = os.path.join(args.name)
    with open(name_file) as file:
        config = json.load(file)

    path = os.path.join('project/output', datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
    if not os.path.isdir(path):
        os.mkdir(path)

    processes_list = []
    for key, item in config.items():
        path_key = os.path.join(path, key.lower())
        if not os.path.isdir(path_key):
            os.mkdir(path_key)
        processes_list += [Process(target=run_yrs,
                                   args=(item, path_key, )
                                   )]

    for p in processes_list:
        p.start()
    for p in processes_list:
        p.join()

    for key in config.keys():
        path_key = os.path.join(path, key.lower())
        scenarios = [s for s in os.listdir(path_key)]
        scenarios_dict = {s: pd.read_csv(os.path.join(path_key, s), index_col=[0, 1, 2, 3, 4]) for s in scenarios}

        cost_cumulated_emission_plots(scenarios_dict,
                                      save=os.path.join(path_key, 'cost_emission_end_{}.png'.format(key.lower())),
                                      graph='Cumulated emission difference (%)')
        cost_cumulated_emission_plots(scenarios_dict,
                                      save=os.path.join(path_key, 'cost_potential_emission_{}.png'.format(key.lower())),
                                      graph='Cumulated potential emission saving (%)')
        cost_cumulated_emission_plots(scenarios_dict,
                                      save=os.path.join(path_key, 'cost_dwelling_{}.png'.format(key.lower())),
                                      graph='Cumulated dwelling number (%)')





