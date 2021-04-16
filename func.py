import numpy as np
from scipy.optimize import fsolve

from input import parameters_dict, language_dict, technical_progress_dict, cost_dict, exogenous_dict, index_year
from function_pandas import *


def logistic(x, a=1, r=1, K=1):
    return K / (1 + a * np.exp(- r * x))


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
        interest_rate_ds = parameters_dict['interest_rate_new']

    if isinstance(interest_rate_ds, pd.Series):
        discounted_df = interest_rate_ds.apply(interest_rate2series, args=[index_year])
        discounted_df = pd.DataFrame.from_dict(dict(zip(discounted_df.index, discounted_df.values))).T
        discounted_df.columns = index_year
    elif isinstance(interest_rate_ds, float):
        discounted_df = pd.Series(interest_rate2series(interest_rate_ds, index_year), index=index_year)
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

    if kind == 'new':
        area = parameters_dict['area_new']
    elif kind == 'remaining':
        area = parameters_dict['area']

    area = reindex_mi(area, segments, ['Occupancy status', 'Housing type'])

    income_ts = parameters_dict['income_series'].T.reindex(segments.get_level_values('Income class'))
    income_ts.index = segments

    energy_consumption_conventional = reindex_mi(parameters_dict['energy_consumption_df'], segments,
                                                 ['Heating energy', 'Energy performance'])
    if kind == 'new':
        energy_consumption_conventional = reindex_mi(parameters_dict['energy_consumption_new_series'], segments,
                                                     ['Energy performance'])

    energy_prices_df = energy_prices_df.T.reindex(segments.get_level_values('Heating energy'))
    energy_prices_df.index = segments

    budget_share_df = (energy_prices_df.values.T * area.values * energy_consumption_conventional.values) / income_ts.values.T
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


def segments2energy_lcc(segments, energy_prices_df, yr, kind='remaining', transition='label'):
    """
    Return energy life cycle cost (LCC) discounted from segments, and energy prices starting year=yr.
    Energy LCC is calculated on an segment-specific horizon, and using a segment-specific discount rate.
    Because, time horizon depends of type of renovation (label, or heating energy), lcc needs to know which transition.
    Energy LCC can also be calculated for new constructed buildings: kind='new'.

    :param segments:
    :param energy_prices_df:
    :param yr: starting year
    :param kind='remaining' or 'new'
    :param transition='label' or 'energy' ('label-energy' for later)
    :return: energy life cycle cost (LCC)
    """

    energy_cost_ts_df = segments2energy_func(segments, energy_prices_df, kind=kind)['Energy cost-actual']
    discounted_df = discount_rate_series_func(index_year, kind=kind)
    if kind == 'remaining':
        discounted_df_reindex = reindex_mi(discounted_df, energy_cost_ts_df.index, ['Income class owner', 'Housing type'])
    elif kind == 'new':
        discounted_df_reindex = pd.concat([discounted_df] * len(energy_cost_ts_df.index), axis=1).T
        discounted_df_reindex.index = energy_cost_ts_df.index
    energy_cost_discounted_ts_df = discounted_df_reindex * energy_cost_ts_df

    if kind == 'remaining':
        if transition == 'label':
            invest_horizon = parameters_dict['investment_horizon_enveloppe_ds']
        elif transition == 'energy':
            invest_horizon = parameters_dict['investment_horizon_heater_ds']
        invest_horizon_reindex = reindex_mi(invest_horizon, energy_cost_ts_df.index,
                                            ['Occupancy status'])
    elif kind == 'new':
        invest_horizon = parameters_dict['investment_horizon_construction']
        invest_horizon_reindex = pd.Series(invest_horizon, index=energy_cost_ts_df.index)

    def horizon2years(num, start_yr):
        """
        Return list of years based on a number of years and starting year.
        :param num:
        :param start_yr:
        :return:
        """
        return [start_yr + k for k in range(num)]

    invest_years = invest_horizon_reindex.apply(horizon2years, args=[yr])

    def time_series2sum(ds, invest_years, levels):
        """
        Return sum of ds for each segment based on list of years in invest years.
        :param ds: segments as index, time series as column
        :param invest_years: pandas Series with list of years to use for each segment
        :return:
        """
        idx_invest = [ds[l] for l in levels]
        idx_years = invest_years.loc[tuple(idx_invest)]
        return ds.loc[idx_years].sum()

    if kind == 'remaining':
        levels = language_dict['levels_names']
    elif kind == 'new':
        levels = [l for l in language_dict['levels_names'] if l != 'Income class owner']

    energy_lcc_ds = energy_cost_discounted_ts_df.reset_index().apply(time_series2sum, args=[invest_years, levels], axis=1)
    energy_lcc_ds.index = energy_cost_discounted_ts_df.index
    energy_lcc_ds = energy_lcc_ds.to_frame()
    energy_lcc_ds.columns = ['Values']

    return energy_lcc_ds


def cost2lcc(energy_discount_lcc_ds, cost_invest_df, cost_switch_fuel_df, intangible_cost, transition='label'):
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


def renov_rate_func(ds, rho):
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
    lcc_df = cost2lcc(energy_lcc_ds, cost_invest_df, cost_switch_fuel_df, cost_intangible_df, transition=transition)
    lcc_df = lcc_df.reorder_levels(energy_lcc_ds.index.names)
    market_share_df = lcc2market_share(lcc_df)
    pv_df = (market_share_df * lcc_df).sum(axis=1)

    segments_initial = pv_df.index
    energy_initial_lcc_ds = segments2energy_lcc(segments_initial, energy_prices_df, yr)
    npv_df = energy_initial_lcc_ds.iloc[:, 0] - pv_df
    renovation_rate_df = npv_df.reset_index().apply(renov_rate_func, rho=rho, axis=1)
    renovation_rate_df.index = npv_df.index

    return {'Market share': market_share_df, 'NPV': pv_df, 'Renovation rate': renovation_rate_df,
            'Energy LCC': energy_lcc_ds}


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
    levels_wo_performance = [l for l in language_dict['levels_names'] if l != 'Energy performance']
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


def renovation_label2renovation_label_energy(energy_lcc_ds, cost_invest_df, cost_switch_fuel_df, cost_intangible_df, stock_renovation_label):
    """
    De-aggregate stock_renovation_label by heating energy.
    stock_renovation segmented by final label and final heating energy.
    """

    lcc_energy_transition = cost2lcc(energy_lcc_ds, cost_invest_df, cost_switch_fuel_df, cost_intangible_df, transition='energy')
    lcc_energy_transition = lcc_energy_transition.reorder_levels(energy_lcc_ds.index.names)
    market_share_energy = lcc2market_share(lcc_energy_transition)
    ms_temp = pd.concat([market_share_energy.T] * len(language_dict['energy_performance_list']),
                        keys=language_dict['energy_performance_list'], names=['Energy performance'])
    sr_temp = pd.concat([stock_renovation_label.T] * len(language_dict['heating_energy_list']),
                        keys=language_dict['heating_energy_list'], names=['Heating energy'])
    stock_renovation_label_energy = (sr_temp * ms_temp).T
    return stock_renovation_label_energy


def information_rate_func(knowldege_normalize, kind='remaining'):
    """
    Ref: Res-IRF Scilab
    Returns information rate. More info_rate big, more intangible_cost small.
    intangible_cost[yr] = intangible_cost[calibration_year] * info_rate with info rate [1-info_rate_max ; 1]
    Calibration of information rate logistic function.
    """
    sh = technical_progress_dict['information_rate_max']
    if kind == 'new':
        sh = technical_progress_dict['information_rate_max_new']

    alpha = technical_progress_dict['information_rate_intangible']
    if kind == 'new':
        alpha = technical_progress_dict['information_rate_intangible_new']

    def equations(p, sh=sh, alpha=alpha):
        a, r = p
        return (1 + a * np.exp(-r)) ** -1 - sh, (1 + a * np.exp(-2 * r)) ** -1 - sh - (1 - alpha) * sh + 1

    a, r = fsolve(equations, (1, -1))

    return logistic(knowldege_normalize, a=a, r=r) + 1 - sh


def learning_by_doing_func(knowledge_normalize, learning_rate, yr, cost_new, cost_new_lim_ds, calibration_yr):
    learning_by_doing_new = knowledge_normalize ** (np.log(1 + learning_rate) / np.log(2))
    learning_by_doing_new_reindex = reindex_mi(learning_by_doing_new, cost_new_lim_ds.index, ['Energy performance'])
    cost_new[yr] = cost_new[calibration_yr] * learning_by_doing_new_reindex + cost_new_lim_ds * (
            1 - learning_by_doing_new_reindex)
    return cost_new


def housing_need2housing_construction(nb_housing_need, nb_housing_construction, share_multi_family, yr, yr_ini):
    """
    Returns segmented (Occupancy status, Housing type) number of new construction for a year.
    Also returns share of Multi-family buildings in the total building parc.
    Using trend we calculate share of multi_family for the entire parc and the one in construction
    """
    trend_housing = (nb_housing_need[yr] - nb_housing_need[yr_ini]) / nb_housing_need[
        yr] * 100
    share_multi_family[yr] = 0.1032 * np.log(10.22 * trend_housing / 10 + 79.43) * parameters_dict[
        'factor_evolution_distribution']
    share_multi_family_construction = (nb_housing_need[yr] * share_multi_family[yr] - nb_housing_need[
        yr - 1] * share_multi_family[yr - 1]) / nb_housing_construction[yr]
    share_type_housing_new = pd.Series([share_multi_family_construction, 1 - share_multi_family_construction],
                                       index=['Multi-family', 'Single-family'])
    # share of occupancy status in housing type is constant
    share_type_housing_new = reindex_mi(share_type_housing_new, parameters_dict['distribution_type'].index,
                                        ['Housing type'])
    share_type_new = parameters_dict['distribution_type'] * share_type_housing_new
    return share_type_new * nb_housing_construction[yr], share_multi_family


def segments2segments_new(segments):
    """
    Returns segments_new from segments.
    Segments_new doesn't get Income class owner at first and got other Energy performance value.
    """
    levels_wo_owner = [l for l in language_dict['levels_names'] if l != 'Income class owner']
    segments_new = get_levels_values(segments, levels_wo_owner)
    segments_new = segments_new.droplevel('Energy performance')
    segments_new = segments_new.drop_duplicates()
    segments_new = pd.concat(
        [pd.Series(index=segments_new, dtype='float64')] * len(language_dict['energy_performance_new_list']),
        keys=language_dict['energy_performance_new_list'], names=['Energy performance'])
    segments_new = segments_new.reorder_levels(levels_wo_owner).index
    return segments_new


def area_new_dynamic(average_area_new, yr, yr_ini):
    """
    Every year, average area of new buildings increase with available income.
    Trend is based on elasticity area / income.
    eps_area_new decrease over time and reduce elasticity while average area converge towards area_max.
    exogenous_dict['population_total_ds']
    """
    eps_area_new = (parameters_dict['area_new_max'] - average_area_new[yr - 1]) / (
            parameters_dict['area_new_max'] - average_area_new[yr_ini])
    eps_area_new = eps_area_new.apply(lambda x: max(0, min(1, x)))
    elasticity_area_new = eps_area_new.multiply(parameters_dict['elasticity_area_new_ini'])
    factor_area_new = elasticity_area_new * max(0, (
            exogenous_dict['available_income_real_pop_ds'].loc[yr] /
            exogenous_dict['available_income_real_pop_ds'].loc[yr_ini] - 1))

    average_area_new[yr] = pd.concat(
        [parameters_dict['area_new_max'], average_area_new[yr - 1] * (1 + factor_area_new)],
        axis=1).min(axis=1)
    return average_area_new


def nb_population_housing_dynamic(nb_population_housing_prev, nb_population_housing_ini):
    """
    Returns number of people by building for year.
    Number of people by housing decrease over the time.
    TODO: It seems we could get the number of people by buildings exogeneously.
    """
    eps_pop_housing = (nb_population_housing_prev - parameters_dict['nb_population_housing_min']) / (
            nb_population_housing_ini - parameters_dict['nb_population_housing_min'])
    eps_pop_housing = max(0, min(1, eps_pop_housing))
    factor_pop_housing = parameters_dict['factor_population_housing_ini'] * eps_pop_housing
    return max(parameters_dict['nb_population_housing_min'], nb_population_housing_prev * (1 + factor_pop_housing))


def segments_new2flow_constructed(flow_constructed_seg_dm, segments_new, energy_prices_df, ds_income_owner_prop,
                                  yr_ini):
    """
    Returns flow of constructed buildings fully segmented.
    1. Calculate lcc for every segment in segments_new;
    2. Based on lcc, calculate the market-share by decision-maker: distribution_construction_dm;
    3. Multiply by flow_constructed_seg_dm;
    4. De-aggregate levels to add income class owner information.
    """
    lcc_new_ds = segments_new2lcc(segments_new, yr_ini, energy_prices_df, cost_new=cost_dict['cost_new'])
    distribution_construction_dm = val2share(lcc_new_ds, ['Occupancy status', 'Housing type'],
                                             func=lambda x: x ** -parameters_dict['nu_new'], option='column')
    # logging.debug('Construction number: {:,.0f}'.format(flow_constructed_seg_dm.sum()))
    flow_constructed_new = ds_mul_df(flow_constructed_seg_dm, distribution_construction_dm)
    np.testing.assert_almost_equal(flow_constructed_new.sum().sum(), flow_constructed_seg_dm.sum(),
                                   err_msg='Not normal')
    flow_constructed_new = flow_constructed_new.stack(flow_constructed_new.columns.names)
    # keep the same proportion for income class owner than in the initial parc
    flow_constructed_new = de_aggregating_series(flow_constructed_new, ds_income_owner_prop, 'Income class owner')
    return flow_constructed_new[flow_constructed_new > 0]


