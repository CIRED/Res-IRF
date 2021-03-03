from input import parameters_dict, language_dict
import pandas as pd
import numpy as np
from input import calibration_dict

"""
['Occupancy status', 'Housing type', 'Energy performance', 'Heating energy', 'Income class', 'Income class owner']
occupancy_status = segment[0]
housing_type = segment[1]
energy_performance = segment[2]
heating_energy = segment[3]
income_class = segment[4]
income_class_owner = segment[5]
"""

# TODO: check reindex like


def discount_factor_func(segments):
    """
    segments pd MultiIndex
    """
    idx_occ = pd.MultiIndex.from_tuples(segments).get_level_values(0)
    investment_horizon = parameters_dict['investment_horizon_series'].reindex(idx_occ)
    idx_income = pd.MultiIndex.from_tuples(segments).get_level_values(4)
    idx_housing = pd.MultiIndex.from_tuples(segments).get_level_values(1)
    idx_interest = pd.MultiIndex.from_tuples(list(zip(list(idx_income), list(idx_housing))))
    interest_rate = parameters_dict['interest_rate_series'].reindex(idx_interest)
    discount_factor = (1 - (1 + interest_rate.values) ** -investment_horizon.values) / interest_rate.values
    idx_discount = pd.MultiIndex.from_tuples(list(zip(list(idx_occ), list(idx_housing), list(idx_income))))
    discount_factor = pd.Series(discount_factor, index=idx_discount)
    return discount_factor


def energy_cost_func(segments, energy_prices_df):
    """
    Calculate for all segments:
    - budget_share_df
    - use_intensity_df
    - energy_consumption_actual_df
    - energy_cost_df
    pandas DataFrame index = segments, columns = index_year
    """

    # TODO: use energy_prices_df.columns to return time indexed data.

    idx_occ = pd.MultiIndex.from_tuples(segments).get_level_values(0)
    idx_housing = pd.MultiIndex.from_tuples(segments).get_level_values(1)
    idx_surface = pd.MultiIndex.from_tuples(list(zip(list(idx_occ), list(idx_housing))))
    surface = parameters_dict['surface'].reindex(idx_surface)

    idx_income = pd.MultiIndex.from_tuples(segments).get_level_values(4)
    income_ts = parameters_dict['income_series'].T.reindex(idx_income)

    idx_performance = pd.MultiIndex.from_tuples(segments).get_level_values(2)
    idx_energy = pd.MultiIndex.from_tuples(segments).get_level_values(3)
    idx_consumption = pd.MultiIndex.from_tuples(list(zip(list(idx_energy), list(idx_performance))))
    energy_consumption_theoretical = parameters_dict['energy_consumption_df'].reindex(idx_consumption)

    energy_prices_df = energy_prices_df.T.reindex(idx_energy)

    budget_share_df = (energy_prices_df.values.T * surface.values * energy_consumption_theoretical.values) / income_ts.values.T
    budget_share_df = pd.DataFrame(budget_share_df.T, index=segments)

    use_intensity_df = -0.191 * budget_share_df.apply(np.log) + 0.1105

    energy_consumption_actual_df = pd.DataFrame((use_intensity_df.values.T * energy_consumption_theoretical.values).T,
                                                index=segments)
    energy_cost_df = pd.DataFrame(energy_consumption_actual_df.values * energy_prices_df.values, index=segments)

    return budget_share_df, use_intensity_df, energy_consumption_actual_df, energy_cost_df


def lcc_func(energy_discount_lcc_ds, cost_invest_df, cost_switch_fuel_df, intangible_cost):
    """
    Calculate life cycle cost of energy consumption for every segment and for every possible transition.
    """

    pivot = pd.pivot_table(energy_discount_lcc_ds, values='Values', columns=['Energy performance', 'Heating energy'],
                           index=['Occupancy status', 'Housing type', 'Income class', 'Income class owner'])

    lcc_transition_df = pd.concat([pivot] * len(language_dict['energy_performance_list']),
                                           keys=language_dict['energy_performance_list'], names=['Energy performance'])

    lcc_transition_df = pd.concat([lcc_transition_df] * len(language_dict['heating_energy_list']),
                                           keys=language_dict['heating_energy_list'], names=['Heating energy'])

    invest_cost = cost_invest_df.reindex(lcc_transition_df.index.get_level_values('Energy performance'), axis=0)
    invest_cost = invest_cost.reindex(lcc_transition_df.columns.get_level_values('Energy performance'), axis=1)

    switch_fuel_cost = cost_switch_fuel_df.reindex(lcc_transition_df.index.get_level_values('Heating energy'), axis=0)
    switch_fuel_cost = switch_fuel_cost.reindex(lcc_transition_df.columns.get_level_values('Heating energy'), axis=1)

    intangible_cost = intangible_cost.reindex(lcc_transition_df.index.get_level_values('Energy performance'), axis=0)
    intangible_cost = intangible_cost.reindex(lcc_transition_df.columns.get_level_values('Energy performance'), axis=1)

    lcc_transition_df = pd.DataFrame(
        lcc_transition_df.values + invest_cost.values + switch_fuel_cost.values + intangible_cost.values,
        index=lcc_transition_df.index, columns=lcc_transition_df.columns)

    return lcc_transition_df


def market_share_func(lcc_df):

    if isinstance(lcc_df, pd.DataFrame):
        def to_apply(series):
            return series * (series.name[1] > series.index.get_level_values(0))

        lcc_df = lcc_df.apply(to_apply, axis=1)
        lcc_df.replace({0: float('nan')}, inplace=True)
        lcc_reverse_df = lcc_df.apply(lambda x: x**-parameters_dict['nu_intangible_cost'])
        market_share_df = pd.DataFrame((lcc_reverse_df.values.T / lcc_reverse_df.sum(axis=1).values).T,
                                       index=lcc_reverse_df.index, columns=lcc_reverse_df.columns)
        return market_share_df

    elif isinstance(lcc_df, pd.Series):
        # TODO: calculate market share of the average lcc which is not the same than the average of market share
        def to_apply(ds):
            return ds * (ds.name > ds.index)

        lcc_ds = to_apply(lcc_df)
        lcc_ds.replace({0: float('nan')}, inplace=True)
        lcc_reverse_ds = lcc_ds.apply(lambda x: x**-parameters_dict['nu_intangible_cost'])
        market_share_ds = lcc_reverse_ds / lcc_reverse_ds.sum()
        return market_share_ds

    else:
        raise


def logistic(x, a=1, r=1, K=1):
    return K / (1 + a * np.exp(- r * x))

# market_share_func(intangible_cost, energy_discount_lcc_ds, cost_invest_df, cost_switch_fuel_df)
