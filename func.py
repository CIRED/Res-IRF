import numpy as np

from input import parameters_dict, language_dict, cost_dict, index_year
from function_pandas import *


def discount_factor_func(segments):
    """
    Calculate discount factor for all segments.
    :param segments: pandas MultiIndex
    :return:
    """
    investment_horizon = reindex_mi(parameters_dict['investment_horizon_enveloppe_ds'], segments, ['Occupancy status'])
    interest_rate = reindex_mi(parameters_dict['interest_rate_series'], segments, ['Income class', 'Housing type'])
    discount_factor = (1 - (1 + interest_rate) ** -investment_horizon) / interest_rate

    return discount_factor


def discount_rate_series_func(index_year, kind='remaining'):
    """
    Return pd DataFrame - partial segments in index and years in column - corresponding to discount rate.
    :param index_year:
    :return:
    """
    def interest_rate2series(interest_rate, index_yr):
        return [(1 + interest_rate) ** -(yr - index_yr[0]) for yr in index_yr]

    interest_rate_ds = parameters_dict['interest_rate_series']
    if kind == 'new':
        interest_rate_ds = parameters_dict['interest_rate_new_series']

    discounted_df = interest_rate_ds.apply(interest_rate2series, args=[index_year])
    discounted_df = pd.DataFrame.from_dict(dict(zip(discounted_df.index, discounted_df.values))).T
    discounted_df.columns = index_year
    return discounted_df


def segments2energy_func(segments, energy_prices_df, kind='remaining'):
    """
    Calculate for all segments:
    - budget_share_df
    - use_intensity_df
    - energy_consumption_actual_df
    - energy_cost_df
    pandas DataFrame index = segments, columns = index_year
    """

    surface = reindex_mi(parameters_dict['surface'], segments, ['Occupancy status', 'Housing type'])

    income_ts = parameters_dict['income_series'].T.reindex(segments.get_level_values('Income class'))
    income_ts.index = segments

    energy_consumption_conventional = reindex_mi(parameters_dict['energy_consumption_df'], segments,
                                                 ['Heating energy', 'Energy performance'])
    if kind == 'new':
        energy_consumption_conventional = reindex_mi(parameters_dict['energy_consumption_new_series'], segments,
                                                     ['Energy performance'])

    energy_prices_df = energy_prices_df.T.reindex(segments.get_level_values('Heating energy'))
    energy_prices_df.index = segments

    budget_share_df = (energy_prices_df.values.T * surface.values * energy_consumption_conventional.values) / income_ts.values.T
    budget_share_df = pd.DataFrame(budget_share_df.T, index=segments, columns=energy_prices_df.columns)
    use_intensity_df = -0.191 * budget_share_df.apply(np.log) + 0.1105

    energy_consumption_actual = pd.DataFrame((use_intensity_df.values.T * energy_consumption_conventional.values).T,
                                                index=segments, columns=use_intensity_df.columns)
    energy_cost_conventional = pd.DataFrame((energy_prices_df.T.values * energy_consumption_conventional.values).T,
                                            index=segments)
    energy_cost_actual = energy_prices_df * energy_consumption_actual

    result_dict = {'Budget share': budget_share_df,
                   'Use intensity': use_intensity_df,
                   'Consumption-conventional': energy_consumption_conventional,
                   'Consumption-actual': energy_consumption_actual,
                   'Energy cost-conventional': energy_cost_conventional,
                   'Energy cost-actual': energy_cost_actual,
                   }

    return result_dict


def segments2energy_lcc(segments, energy_prices_df, yr, kind='remaining'):
    """
    Return energy life cycle cost discounted from segments, and energy prices.
    :param segments:
    :param energy_prices_df:
    :param calibration_year:
    :return:
    """

    energy_cost_ts_df = segments2energy_func(segments, energy_prices_df, kind=kind)['Energy cost-actual']
    discounted_df = discount_rate_series_func(index_year, kind=kind)
    if kind == 'remaining':
        discounted_df_reindex = reindex_mi(discounted_df, energy_cost_ts_df.index, ['Income class owner', 'Housing type'])
    elif kind == 'new':
        discounted_df_reindex = reindex_mi(discounted_df, energy_cost_ts_df.index, ['Housing type'])

    energy_cost_discounted_ts_df = discounted_df_reindex * energy_cost_ts_df
    invest_horizon_reindex = reindex_mi(parameters_dict['investment_horizon_enveloppe_ds'], energy_cost_ts_df.index,
                                        ['Occupancy status'])

    def horizon2years(num, start_yr):
        """
        Return list of years based on a number of years and starting year.
        :param num:
        :param start_yr:
        :return:
        """
        return [start_yr + k for k in range(num)]

    invest_years = invest_horizon_reindex.apply(horizon2years, args=[yr])

    def time_series2sum(ds, invest_years):
        """
        Return sum of ds for each segment based on list of years in invest years.
        :param ds: segments as index, time series as column
        :param invest_years: pandas Series with list of years to use for each segment
        :return:
        """
        idx_invest = [ds[label] for label in ['Occupancy status', 'Housing type', 'Energy performance', 'Heating energy', 'Income class', 'Income class owner']]
        idx_years = invest_years.loc[tuple(idx_invest)]
        return ds.loc[idx_years].sum()

    energy_lcc_ds = energy_cost_discounted_ts_df.reset_index().apply(time_series2sum, args=[invest_years], axis=1)
    energy_lcc_ds.index = energy_cost_discounted_ts_df.index
    energy_lcc_ds = energy_lcc_ds.to_frame()
    energy_lcc_ds.columns = ['Values']

    return energy_lcc_ds


def lcc_func(energy_discount_lcc_ds, cost_invest_df, cost_switch_fuel_df, intangible_cost, transition='label'):
    """
    Calculate life cycle cost of energy consumption for every segment and for every possible transition.
    Transition can be defined as label transition, 
    param: option - 'label-energy', 'label', 'energy'
    """

    if transition == 'label-energy':
        pivot = pd.pivot_table(energy_discount_lcc_ds, values='Values', columns=['Energy performance', 'Heating energy'],
                               index=['Occupancy status', 'Housing type', 'Income class', 'Income class owner'])
        lcc_transition_df = pd.concat([pivot] * len(language_dict['energy_performance_list']),
                                      keys=language_dict['energy_performance_list'], names=['Energy performance'])
        lcc_transition_df = pd.concat([lcc_transition_df] * len(language_dict['heating_energy_list']),
                                               keys=language_dict['heating_energy_list'], names=['Heating energy'])

    elif transition == 'label':
        pivot = pd.pivot_table(energy_discount_lcc_ds, values='Values', columns=['Energy performance'],
                               index=['Occupancy status', 'Housing type', 'Heating energy', 'Income class', 'Income class owner'])
        lcc_transition_df = pd.concat([pivot] * len(language_dict['energy_performance_list']),
                                      keys=language_dict['energy_performance_list'], names=['Energy performance'])

    elif transition == 'energy':
        pivot = pd.pivot_table(energy_discount_lcc_ds, values='Values', columns=['Heating energy'],
                               index=['Occupancy status', 'Housing type', 'Energy performance', 'Income class', 'Income class owner'])
        lcc_transition_df = pd.concat([pivot] * len(language_dict['heating_energy_list']),
                                               keys=language_dict['heating_energy_list'], names=['Heating energy'])

    if 'label' in transition:
        invest_cost = cost_invest_df.reindex(lcc_transition_df.index.get_level_values('Energy performance'), axis=0)
        invest_cost = invest_cost.reindex(lcc_transition_df.columns.get_level_values('Energy performance'), axis=1)

        intangible_cost = intangible_cost.reindex(lcc_transition_df.index.get_level_values('Energy performance'),
                                                  axis=0)
        intangible_cost = intangible_cost.reindex(lcc_transition_df.columns.get_level_values('Energy performance'),
                                                  axis=1)
    if 'energy' in transition:
        switch_fuel_cost = cost_switch_fuel_df.reindex(lcc_transition_df.index.get_level_values('Heating energy'), axis=0)
        switch_fuel_cost = switch_fuel_cost.reindex(lcc_transition_df.columns.get_level_values('Heating energy'), axis=1)

    if transition == 'label-energy':
        lcc_transition_df = pd.DataFrame(
            lcc_transition_df.values + invest_cost.values + switch_fuel_cost.values + intangible_cost.values,
            index=lcc_transition_df.index, columns=lcc_transition_df.columns)
    elif transition == 'label':
        lcc_transition_df = pd.DataFrame(
            lcc_transition_df.values + invest_cost.values + intangible_cost.values,
            index=lcc_transition_df.index, columns=lcc_transition_df.columns)
    elif transition == 'energy':
        # TODO: energy transition intangible cost
        lcc_transition_df = pd.DataFrame(
            lcc_transition_df.values + switch_fuel_cost.values,
            index=lcc_transition_df.index, columns=lcc_transition_df.columns)

    return lcc_transition_df


def lcc2market_share(lcc_df, nu=8):
    """
    Returns market share for each segment based on lcc_df.
    """

    lcc_reverse_df = lcc_df.apply(lambda x: x**-nu)
    market_share_df = pd.DataFrame((lcc_reverse_df.values.T / lcc_reverse_df.sum(axis=1).values).T,
                                   index=lcc_reverse_df.index, columns=lcc_reverse_df.columns)
    return market_share_df


def logistic(x, a=1, r=1, K=1):
    return K / (1 + a * np.exp(- r * x))


def segments2renovation_rate(segments, yr, energy_prices_df, cost_invest_df, cost_switch_fuel_df, cost_intangible_df, rho, transition='label'):
    """
    Routine calculating renovation rate from segments for a particular yr.
    Cost (energy, investment) & rho parameter are also required.
    :param segments:
    :param energy_prices_df:
    :param yr:
    :param cost_invest_df:
    :param cost_switch_fuel_df:
    :param cost_intangible_df:
    :return:
    """
    energy_lcc_ds = segments2energy_lcc(segments, energy_prices_df, yr)
    lcc_df = lcc_func(energy_lcc_ds, cost_invest_df, cost_switch_fuel_df, cost_intangible_df, transition=transition)
    lcc_df = lcc_df.reorder_levels(energy_lcc_ds.index.names)
    market_share_df = lcc2market_share(lcc_df)
    pv_df = (market_share_df * lcc_df).sum(axis=1)

    segments_initial = pv_df.index
    energy_initial_lcc_ds = segments2energy_lcc(segments_initial, energy_prices_df, yr)
    npv_df = energy_initial_lcc_ds.iloc[:, 0] - pv_df

    def func(ds, rho):
        if isinstance(rho, pd.Series):
            rho_f = rho.loc[tuple(ds.iloc[:-1].tolist())]
        else:
            rho_f = rho

        if np.isnan(rho_f):
            return float('nan')
        else:
            return logistic(ds.loc[0] - parameters_dict['npv_min'],
                            a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                            r=rho_f,
                            K=parameters_dict['rate_max'])

    renovation_rate_df = npv_df.reset_index().apply(func, rho=rho, axis=1)
    renovation_rate_df.index = npv_df.index

    return {'Market share': market_share_df, 'NPV': pv_df, 'Renovation rate': renovation_rate_df}


def segments_new2lcc(segments_new, yr, energy_prices_df, cost_new=cost_dict['cost_new']):
    energy_lcc_new_ds = segments2energy_lcc(segments_new, energy_prices_df, yr, kind='new').iloc[:, 0]
    cost_new_reindex = reindex_mi(cost_new, energy_lcc_new_ds.index, ['Heating energy', 'Housing type', 'Energy performance'])
    lcc_new_ds = energy_lcc_new_ds + cost_new_reindex
    return lcc_new_ds


def stock_mobile2stock_destroyed(stock_mobile, stock_mobile_ini, stock_remaining, type_housing_destroyed, logging):
    """
    Returns stock_destroyed -  housing number destroyed this year for each segment.
    Houses to destroy are chosen in stock_mobile.
    1. type_housing_destroyed is respected to match decision-maker proportion; - type_housing_destroyed_reindex
    2. income_class, income_class_owner, heating_energy match stock_remaining proportion; - type_housing_destroyed_wo_performance
    3. worst energy_performance_label for each segment are targeted. - stock_destroyed
    """

    segments_mobile = stock_mobile.index
    segments_mobile = segments_mobile.droplevel('Energy performance')
    segments_mobile = segments_mobile.drop_duplicates()

    def worst_label(segments_mobile):
        """
        Returns worst label for each segment.
        Dictionary that returns the worst
        """
        idx_worst_label_list = []
        worst_label_dict = dict()
        for segment in segments_mobile:
            for label in language_dict['energy_performance_list']:
                idx = (segment[0], segment[1], label, segment[2], segment[3], segment[4])
                if stock_mobile.loc[idx] > 1:
                    idx_worst_label_list.append(idx)
                    worst_label_dict[segment] = label
                    break
        return idx_worst_label_list, worst_label_dict

    idx_worst_label_list, worst_label_dict = worst_label(segments_mobile)

    # we know type_destroyed, then we calculate nb_housing_destroyed_ini based on proportion of stock remaining
    levels = ['Occupancy status', 'Housing type']
    levels_wo_performance = [l for l in language_dict['properties_names'] if l != 'Energy performance']
    stock_remaining_woperformance = stock_remaining.groupby(levels_wo_performance).sum()
    prop_housing_remaining_decision = val2share(stock_remaining_woperformance, levels)
    logging.debug('Number of destroyed houses {:,.0f}'.format(type_housing_destroyed.sum()))
    type_housing_destroyed_reindex = reindex_mi(type_housing_destroyed,
                                                prop_housing_remaining_decision.index, levels)
    type_housing_destroyed_wo_performance = type_housing_destroyed_reindex * prop_housing_remaining_decision
    logging.debug('Number of destroyed houses {:,.0f}'.format(type_housing_destroyed_wo_performance.sum()))

    logging.debug('De-aggregate destroyed houses by labels')
    # we don't have the information about which labels are going to be destroyed first
    prop_stock_worst_label = stock_mobile.loc[idx_worst_label_list] / stock_mobile_ini.loc[
        idx_worst_label_list]
    nb_housing_destroyed_ini = reindex_mi(type_housing_destroyed_wo_performance, prop_stock_worst_label.index,
                                          levels_wo_performance)

    # initialize nb_housing_destroyed_theo for worst label based on how much have been destroyed so far
    nb_housing_destroyed_theo = prop_stock_worst_label * nb_housing_destroyed_ini
    stock_destroyed = pd.Series(0, index=stock_mobile.index, dtype='float64')

    logging.debug('Start while loop!')
    # TODO clean these lines, and create a function that returns stock_destroyed

    # Returns stock_destroyed for each segment
    # we start with the worst label and we stop when nb_housing_destroyed_theo == 0
    for segment in segments_mobile:
        label = worst_label_dict[segment]
        num = language_dict['energy_performance_list'].index(label)
        idx_tot = (segment[0], segment[1], label, segment[2], segment[3], segment[4])

        while nb_housing_destroyed_theo.loc[idx_tot] != 0:
            # stock_destroyed cannot be sup to stock_mobile and to nb_housing_destroyed_theo
            stock_destroyed.loc[idx_tot] = min(stock_mobile.loc[idx_tot], nb_housing_destroyed_theo.loc[idx_tot])
            if label != 'A':
                num += 1
                label = language_dict['energy_performance_list'][num]
                labels = language_dict['energy_performance_list'][:num + 1]
                idx = (segment[0], segment[1], segment[2], segment[3], segment[4])
                idx_tot = (segment[0], segment[1], label, segment[2], segment[3], segment[4])
                idxs_tot = [(segment[0], segment[1], label, segment[2], segment[3], segment[4]) for label in labels]

                # nb_housing_destroyed_theo is the remaining number of housing that need to be destroyed for this segment
                nb_housing_destroyed_theo[idx_tot] = type_housing_destroyed_wo_performance.loc[idx] - \
                                                     stock_destroyed.loc[idxs_tot].sum()

            else:
                nb_housing_destroyed_theo[idx_tot] = 0

    logging.debug('Number of destroyed houses {:,.0f}'.format(stock_destroyed.sum()))
    # check if nb_housing_destroyed is constant
    logging.debug('End while loop!')

    return stock_destroyed

# lcc2market_share(intangible_cost, energy_discount_lcc_ds, cost_invest_df, cost_switch_fuel_df)
