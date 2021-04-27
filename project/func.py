import numpy as np
from scipy.optimize import fsolve
import os

from input import parameters_dict, language_dict, technical_progress_dict, cost_dict, exogenous_dict, calibration_dict, index_year, folder
from function_pandas import *


def logistic(x, a=1, r=1, k=1):
    return k / (1 + a * np.exp(- r * x))


def buildings_number2area(ds):
    """Returns area of a building number pd.Series or pd.DataFrame.

    Buildings data should contain at least Occupancy status and Housing type levels.
    """
    area = reindex_mi(parameters_dict['area'], ds.index, ['Occupancy status', 'Housing type'])
    if isinstance(ds, pd.Series):
        return area * ds
    elif isinstance(ds, pd.DataFrame):
        return ds_mul_df(area, ds)


def discount_factor_func(segments):
    """
    Calculate discount factor for all segments.
    :param segments: pandas MultiIndex
    :return:
    """
    investment_horizon = reindex_mi(parameters_dict['investment_horizon_envelope_ds'], segments, ['Occupancy status'])
    interest_rate = reindex_mi(parameters_dict['interest_rate_seg'], segments, ['Income class', 'Housing type'])
    discount_factor = (1 - (1 + interest_rate) ** -investment_horizon) / interest_rate

    return discount_factor


def discount_rate_series_func(index_yr, kind='remaining'):
    """Return pd.DataFrame - partial segments in index and years in column - with value corresponding to discount rate.

    """
    def interest_rate2series(interest_rate, idx_yr):
        return [(1 + interest_rate) ** -(yr - idx_yr[0]) for yr in idx_yr]

    if kind == 'remaining':
        interest_rate_ds = parameters_dict['interest_rate_seg']
    elif kind == 'new':
        interest_rate_ds = parameters_dict['interest_rate_new']
    else:
        raise ValueError

    if isinstance(interest_rate_ds, pd.Series):
        discounted_df = interest_rate_ds.apply(interest_rate2series, args=[index_yr])
        discounted_df = pd.DataFrame.from_dict(dict(zip(discounted_df.index, discounted_df.values))).T
        discounted_df.columns = index_year
        discounted_df.index.names = interest_rate_ds.index.names
    elif isinstance(interest_rate_ds, float):
        discounted_df = pd.Series(interest_rate2series(interest_rate_ds, index_year), index=index_year)
    else:
        return ValueError
    return discounted_df


def population_housing_dynamic(nb_population_housing_prev, nb_population_housing_ini):
    """Returns number of people by building for year.

    Number of people by housing decrease over the time.
    TODO: It seems we could get the number of people by buildings exogeneously.
    """
    eps_pop_housing = (nb_population_housing_prev - parameters_dict['nb_population_housing_min']) / (
            nb_population_housing_ini - parameters_dict['nb_population_housing_min'])
    eps_pop_housing = max(0, min(1, eps_pop_housing))
    factor_pop_housing = parameters_dict['factor_population_housing_ini'] * eps_pop_housing
    return max(parameters_dict['nb_population_housing_min'], nb_population_housing_prev * (1 + factor_pop_housing))


def segments2energy_consumption(segments, energy_prices, kind='remaining'):
    """Returns real energy consumption real considering household behavior, and conventional.

    Calculate for all segments:
    - budget_share_df
    - use_intensity_df
    - energy_consumption_actual_df
    - energy_cost_df
    pandas DataFrame index = segments, columns = index_year
    """
    # TODO: function could returns consumption conventional even if income class not in levels

    if kind == 'new':
        area = parameters_dict['area_new']
    elif kind == 'remaining':
        area = parameters_dict['area']
    else:
        raise ValueError('Kind should be in {}'.format(['new', 'remaining']))
    area = reindex_mi(area, segments, ['Occupancy status', 'Housing type'])

    income_ts = parameters_dict['income_series'].T.reindex(segments.get_level_values('Income class'))
    income_ts.index = segments

    if kind == 'remaining':
        energy_consumption = parameters_dict['energy_consumption_df']
    elif kind == 'new':
        energy_consumption = parameters_dict['energy_consumption_new_series']
    energy_consumption_conventional = reindex_mi(energy_consumption, segments, ['Energy performance', 'Heating energy'])
    energy_prices = energy_prices.T.reindex(segments.get_level_values('Heating energy'))
    energy_prices.index = segments

    budget_share_seg = ds_mul_df(area * energy_consumption_conventional, energy_prices / income_ts)
    use_intensity_seg = -0.191 * budget_share_seg.apply(np.log) + 0.1105
    energy_consumption_actual = ds_mul_df(energy_consumption_conventional, use_intensity_seg)

    result_dict = {'Budget share': budget_share_seg,
                   'Use intensity': use_intensity_seg,
                   'Consumption-conventional': energy_consumption_conventional,
                   'Consumption-actual': energy_consumption_actual,
                   }

    return result_dict


def energy_consumption2cost(energy_consumption, energy_prices):
    """Returns energy cost segmented and for every year based on energy consumption and energy prices.
    """
    energy_prices = reindex_mi(energy_prices.T, energy_consumption.index, ['Heating energy'])
    if isinstance(energy_consumption, pd.DataFrame):
        return energy_prices * energy_consumption
    elif isinstance(energy_consumption, pd.Series):
        return ds_mul_df(energy_consumption, energy_prices)


def segments2energy_lcc(segments, yr, kind='remaining', transition='label', consumption='conventional', e_prices='myopic'):
    """Return segmented energy-life-cycle-cost discounted from segments, and energy prices starting year=yr.

    Energy LCC is calculated on an segment-specific horizon, and using a segment-specific discount rate.
    Because, time horizon depends of type of renovation (label, or heating energy), lcc needs to know which transition.
    Energy LCC can also be calculated for new constructed buildings: kind='new'.
    """
    energy_prices = exogenous_dict['energy_price_' + e_prices]
    energy_consumption_seg = segments2energy_consumption(segments, energy_prices, kind=kind)['Consumption-' + consumption]
    energy_cost_ts_df = energy_consumption2cost(energy_consumption_seg, energy_prices)
    discounted_df = discount_rate_series_func(index_year, kind=kind)

    if isinstance(discounted_df, pd.DataFrame):
        discounted_df_reindex = reindex_mi(discounted_df, energy_cost_ts_df.index, discounted_df.index.names)

    elif isinstance(discounted_df, pd.Series):
        discounted_df_reindex = pd.concat([discounted_df] * len(energy_cost_ts_df.index), axis=1).T
        discounted_df_reindex.index = energy_cost_ts_df.index
    energy_cost_discounted_ts_df = discounted_df_reindex * energy_cost_ts_df

    if kind == 'remaining':
        if transition == 'label':
            invest_horizon = parameters_dict['investment_horizon_envelope_ds']
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


def cost2lcc(energy_lcc_seg, cost_invest=None, cost_switch_fuel=None, cost_intangible=None, transition='label'):
    """Calculate life-cycle-cost of home-energy retrofits for every segment and for every possible transition.

    Parameters
    ----------
    energy_lcc_seg : pd.Series
    Segmented energy life-cycle-cost (discounted). Equivalent to NPV.

    cost_invest: pd.Series, optional
    Label initial, Label final.

    cost_switch_fuel: pd.DataFrame, optional
    Energy initial, Energy final.

    cost_intangible: pd.DataFrame, optional
    Label initial, Label final.

    transition: str, optional
    ['label', 'energy', 'label-energy']
    Define transition. Transition can be defined as label transition, energy transition, or label-energy transition.

    Returns
    -------
    pd.Series
    param: option - 'label-energy', 'label', 'energy'
    """

    if transition == 'label-energy':
        pivot = pd.pivot_table(energy_lcc_seg, values='Values', columns=['Energy performance', 'Heating energy'],
                               index=['Occupancy status', 'Housing type', 'Income class', 'Income class owner'])
        lcc_transition_seg = pd.concat([pivot] * len(language_dict['energy_performance_list']),
                                       keys=language_dict['energy_performance_list'], names=['Energy performance'])
        lcc_transition_seg = pd.concat([lcc_transition_seg] * len(language_dict['heating_energy_list']),
                                       keys=language_dict['heating_energy_list'], names=['Heating energy'])

    elif transition == 'label':
        pivot = pd.pivot_table(energy_lcc_seg, values='Values', columns=['Energy performance'],
                               index=['Occupancy status', 'Housing type', 'Heating energy', 'Income class', 'Income class owner'])
        lcc_transition_seg = pd.concat([pivot] * len(language_dict['energy_performance_list']),
                                       keys=language_dict['energy_performance_list'], names=['Energy performance'])

    elif transition == 'energy':
        pivot = pd.pivot_table(energy_lcc_seg, values='Values', columns=['Heating energy'],
                               index=['Occupancy status', 'Housing type', 'Energy performance', 'Income class', 'Income class owner'])
        lcc_transition_seg = pd.concat([pivot] * len(language_dict['heating_energy_list']),
                                       keys=language_dict['heating_energy_list'], names=['Heating energy'])

    else:
        raise ValueError('Transition should be in {}'.format(['label', 'label_energy', 'energy']))

    if 'label' in transition:
        if cost_invest is not None:
            cost_invest = cost_invest.reindex(lcc_transition_seg.index.get_level_values('Energy performance'), axis=0)
            cost_invest = cost_invest.reindex(lcc_transition_seg.columns.get_level_values('Energy performance'), axis=1)
        if cost_intangible is not None:
            levels = [lvl for lvl in lcc_transition_seg.index.names if lvl in cost_intangible.index.names]
            cost_intangible = cost_intangible.reorder_levels(lcc_transition_seg.index.names)
            cost_intangible = reindex_mi(cost_intangible, lcc_transition_seg.index, levels)
            cost_intangible = cost_intangible.reindex(lcc_transition_seg.columns.get_level_values('Energy performance'),
                                                      axis=1)
            cost_intangible.fillna(0, inplace=True)

    if 'energy' in transition:
        if cost_switch_fuel is not None:
            cost_switch_fuel = cost_switch_fuel.reindex(lcc_transition_seg.index.get_level_values('Heating energy'),
                                                        axis=0)
            cost_switch_fuel = cost_switch_fuel.reindex(lcc_transition_seg.columns.get_level_values('Heating energy'),
                                                        axis=1)

    if transition == 'label-energy':
        if cost_intangible is None:
            lcc_transition_seg = pd.DataFrame(
                lcc_transition_seg.values + cost_invest.values + cost_switch_fuel.values,
                index=lcc_transition_seg.index, columns=lcc_transition_seg.columns)
        else:
            lcc_transition_seg = pd.DataFrame(
                lcc_transition_seg.values + cost_invest.values + cost_switch_fuel.values + cost_intangible.values,
                index=lcc_transition_seg.index, columns=lcc_transition_seg.columns)
    elif transition == 'label':
        if cost_intangible is None:
            lcc_transition_seg = pd.DataFrame(
                lcc_transition_seg.values + cost_invest.values,
                index=lcc_transition_seg.index, columns=lcc_transition_seg.columns)
        else:
            lcc_transition_seg = pd.DataFrame(
                lcc_transition_seg.values + cost_invest.values + cost_intangible.values,
                index=lcc_transition_seg.index, columns=lcc_transition_seg.columns)
    elif transition == 'energy':
        lcc_transition_seg = pd.DataFrame(
            lcc_transition_seg.values + cost_switch_fuel.values,
            index=lcc_transition_seg.index, columns=lcc_transition_seg.columns)

    lcc_transition_seg = lcc_transition_seg.reorder_levels(energy_lcc_seg.index.names)

    return lcc_transition_seg


def lcc2market_share(lcc_df, nu=8):
    """Returns market share for each segment based on lcc_df.

    Parameters
    ----------
    lcc_df : pd.DataFrame or pd.Series

    nu: int, optional

    Returns
    -------
    DataFrame if lcc_df is DataFrame or Series if lcc_df is Series.
    """

    lcc_reverse_df = lcc_df.apply(lambda x: x**-nu)
    if isinstance(lcc_df, pd.DataFrame):
        return pd.DataFrame((lcc_reverse_df.values.T / lcc_reverse_df.sum(axis=1).values).T,
                                       index=lcc_reverse_df.index, columns=lcc_reverse_df.columns)
    elif isinstance(lcc_df, pd.Series):
        return lcc_reverse_df / lcc_reverse_df.sum()


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
                        k=parameters_dict['rate_max'])


def segments2renovation_rate(segments, yr, cost_invest_df, cost_intangible_df, rho):
    """Routine calculating renovation rate from segments for a particular yr.

    Cost (energy, investment) & rho parameter are also required.
    """
    energy_lcc_ds = segments2energy_lcc(segments, yr)
    lcc_df = cost2lcc(energy_lcc_ds, cost_invest=cost_invest_df, cost_intangible=cost_intangible_df, transition='label')
    lcc_df = lcc_df.reorder_levels(energy_lcc_ds.index.names)
    market_share_df = lcc2market_share(lcc_df)
    pv_df = (market_share_df * lcc_df).sum(axis=1).replace(0, float('nan'))

    segments_initial = pv_df.index
    energy_initial_lcc_ds = segments2energy_lcc(segments_initial, yr)
    npv_df = energy_initial_lcc_ds.iloc[:, 0] - pv_df
    renovation_rate_df = npv_df.reset_index().apply(renov_rate_func, rho=rho, axis=1)
    renovation_rate_df.index = npv_df.index

    return {'Market share': market_share_df, 'NPV': pv_df, 'Renovation rate': renovation_rate_df,
            'Energy LCC': energy_lcc_ds}


def stock_mobile2flow_destroyed(stock_mobile, stock_mobile_ini, stock_remaining, type_housing_destroyed, logging):
    """ Returns stock_destroyed -  segmented housing number demolition.

    Buildings to destroy are chosen in stock_mobile.
    1. type_housing_destroyed is respected to match decision-maker proportion; - type_housing_destroyed_reindex
    2. income_class, income_class_owner, heating_energy match stock_remaining proportion; - type_housing_destroyed_wo_performance
    3. worst energy_performance_label for each segment are targeted. - stock_destroyed

    Parameters
    ----------
    stock_mobile : pd.Series

    stock_mobile_ini: pd.Series

    Returns
    -------
    pd.Series
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
    logging.debug('Number of buildings demolition: {:,.0f}'.format(type_housing_destroyed.sum()))
    type_housing_destroyed_reindex = reindex_mi(type_housing_destroyed,
                                                prop_housing_remaining_decision.index, levels)
    type_housing_destroyed_wo_performance = type_housing_destroyed_reindex * prop_housing_remaining_decision
    np.testing.assert_almost_equal(type_housing_destroyed.sum(), type_housing_destroyed_wo_performance.sum(),
                                   err_msg='Not normal')

    logging.debug('De-aggregate buildings demolition by labels')
    # we don't have the information about which labels are going to be destroyed first
    prop_stock_worst_label = stock_mobile.loc[idx_worst_label_list] / stock_mobile_ini.loc[
        idx_worst_label_list]
    nb_housing_destroyed_ini = reindex_mi(type_housing_destroyed_wo_performance, prop_stock_worst_label.index,
                                          levels_wo_performance)

    # initialize nb_housing_destroyed_theo for worst label based on how much have been destroyed so far
    nb_housing_destroyed_theo = prop_stock_worst_label * nb_housing_destroyed_ini
    flow_destroyed = pd.Series(0, index=stock_mobile.index, dtype='float64')

    logging.debug('Start while loop!')
    # TODO clean these lines, and create a function that returns flow_destroyed

    # Returns flow_destroyed for each segment
    # we start with the worst label and we stop when nb_housing_destroyed_theo == 0
    for segment in segments_mobile:
        label = worst_label_dict[segment]
        num = language_dict['energy_performance_list'].index(label)
        idx_tot = (segment[0], segment[1], label, segment[2], segment[3], segment[4])

        while nb_housing_destroyed_theo.loc[idx_tot] != 0:
            # stock_destroyed cannot be sup to stock_mobile and to nb_housing_destroyed_theo
            flow_destroyed.loc[idx_tot] = min(stock_mobile.loc[idx_tot], nb_housing_destroyed_theo.loc[idx_tot])
            if label != 'A':
                num += 1
                label = language_dict['energy_performance_list'][num]
                labels = language_dict['energy_performance_list'][:num + 1]
                idx = (segment[0], segment[1], segment[2], segment[3], segment[4])
                idx_tot = (segment[0], segment[1], label, segment[2], segment[3], segment[4])
                idxs_tot = [(segment[0], segment[1], label, segment[2], segment[3], segment[4]) for label in labels]

                # nb_housing_destroyed_theo is the remaining number of housing that need to be destroyed for this segment
                nb_housing_destroyed_theo[idx_tot] = type_housing_destroyed_wo_performance.loc[idx] - \
                                                     flow_destroyed.loc[idxs_tot].sum()

            else:
                nb_housing_destroyed_theo[idx_tot] = 0

    logging.debug('Number of buildings demolition {:,.0f}'.format(flow_destroyed.sum()))
    # check if nb_housing_destroyed is constant
    logging.debug('End while loop!')

    return flow_destroyed


def renovation_label2renovation_label_energy(energy_lcc_ds, cost_switch_fuel_df, flow_renovation_label):
    """De-aggregate stock_renovation_label by final heating energy.

    stock_renovation columns segmented by final label and final heating energy.

    Parameters
    ----------
    energy_lcc_ds : pd.DataFrame
        The

    cost_switch_fuel_df: pd.DataFrame

    stock_renovation_label: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    lcc_energy_transition = cost2lcc(energy_lcc_ds, cost_switch_fuel=cost_switch_fuel_df, transition='energy')
    lcc_energy_transition = lcc_energy_transition.reorder_levels(energy_lcc_ds.index.names)
    market_share_energy = lcc2market_share(lcc_energy_transition)
    ms_temp = pd.concat([market_share_energy.T] * len(language_dict['energy_performance_list']),
                        keys=language_dict['energy_performance_list'], names=['Energy performance'])
    sr_temp = pd.concat([flow_renovation_label.T] * len(language_dict['heating_energy_list']),
                        keys=language_dict['heating_energy_list'], names=['Heating energy'])
    flow_renovation_label_energy = (sr_temp * ms_temp).T
    return flow_renovation_label_energy


def flow_renovation2flow_remained(flow_renovation_label_energy_seg):
    """Calculate flow_remained for each segment.

    Returns: positive (+) flow for buildings segment reached by the renovation (final state),
             negative (-) flow for buildings segment (initial state) that have been renovated.
    """
    flow_renovation_initial_seg = flow_renovation_label_energy_seg.sum(axis=1)
    temp = flow_renovation_label_energy_seg.droplevel('Energy performance', axis=0).droplevel('Heating energy', axis=0)
    temp = temp.stack().stack()
    flow_renovation_final_seg = temp.reorder_levels(language_dict['levels_names'])
    flow_renovation_final_seg = flow_renovation_final_seg.groupby(flow_renovation_final_seg.index).sum()
    flow_renovation_final_seg.index = pd.MultiIndex.from_tuples(flow_renovation_final_seg.index)
    flow_renovation_final_seg.index.names = language_dict['levels_names']
    flow_renovation_final_seg = flow_renovation_final_seg.reindex(flow_renovation_initial_seg.index, fill_value=0)
    flow_remained_seg = flow_renovation_final_seg - flow_renovation_initial_seg
    np.testing.assert_almost_equal(flow_remained_seg.sum(), 0, err_msg='Not normal')
    return flow_remained_seg


def housing_need2housing_construction(nb_housing_need, nb_housing_construction, share_multi_family, yr, yr_ini):
    """Returns segmented (Occupancy status, Housing type) number of new construction for a year.


    Also returns share of Multi-family buildings in the total building parc.
    Using trend we calculate share of multi_family for the entire parc and the one in construction
    """
    trend_housing = (nb_housing_need[yr] - nb_housing_need[yr_ini]) / nb_housing_need[
        yr] * 100
    share_multi_family[yr] = 0.1032 * np.log(10.22 * trend_housing / 10 + 79.43) * parameters_dict[
        'factor_evolution_distribution']
    share_multi_family_construction = (nb_housing_need[yr] * share_multi_family[yr] - nb_housing_need[
        yr - 1] * share_multi_family[yr - 1]) / nb_housing_construction[yr]
    ht_share_tot_construction = pd.Series([share_multi_family_construction, 1 - share_multi_family_construction],
                                          index=['Multi-family', 'Single-family'])
    ht_share_tot_construction.index.set_names('Housing type', inplace=True)
    # share of occupancy status in housing type is constant
    dm_share_tot_construction = ds_mul_df(ht_share_tot_construction, parameters_dict['os_share_ht_construction']).stack()
    return dm_share_tot_construction * nb_housing_construction[yr], share_multi_family


def area_new_dynamic(average_area_new, yr, yr_ini):
    """Every year, average area of new buildings increase with available income.

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


def segments_new2lcc(segments_new, yr, cost_construction=cost_dict['cost_new'], cost_intangible=None):
    """Return lcc_dm for segments_new.
    """
    energy_lcc_new_seg = segments2energy_lcc(segments_new, yr, kind='new').iloc[:, 0]
    cost_new_reindex = reindex_mi(cost_construction, energy_lcc_new_seg.index,
                                  ['Heating energy', 'Housing type', 'Energy performance'])
    lcc_new_seg = energy_lcc_new_seg + cost_new_reindex
    lcc_new_dm = lcc_new_seg.to_frame().pivot_table(index=['Occupancy status', 'Housing type'],
                                                    columns=['Energy performance', 'Heating energy', 'Income class'])
    lcc_new_dm = lcc_new_dm.droplevel(None, axis=1)

    if cost_intangible is not None:
        pass

    return lcc_new_dm


def segments_new2market_share(segments_new, yr, cost_intangible=None):
    """Return market_share of construction for segments_new.
    """
    lcc_new_dm = segments_new2lcc(segments_new, yr, cost_construction=cost_dict['cost_new'],
                                  cost_intangible=cost_intangible)
    market_share_dm = lcc2market_share(lcc_new_dm, nu=parameters_dict['nu_new'])
    return market_share_dm


def segments_new2flow_constructed(flow_constructed_dm, segments_new, io_share_seg, yr, cost_intangible=None):
    """Returns flow of constructed buildings fully segmented.

    2. Calculate the market-share by decision-maker: market_share_dm;
    3. Multiply by flow_constructed_seg_dm;
    4. De-aggregate levels to add income class owner information.
    """

    market_share_dm = segments_new2market_share(segments_new, yr, cost_intangible=cost_intangible)
    flow_constructed_seg = ds_mul_df(flow_constructed_dm, market_share_dm)
    flow_constructed_seg = flow_constructed_seg.stack(flow_constructed_seg.columns.names)
    # keep the same proportion for income class owner than in the initial parc
    flow_constructed_new = de_aggregate_series(flow_constructed_seg, io_share_seg)
    flow_constructed_new = flow_constructed_new[flow_constructed_new > 0]
    return flow_constructed_new


def seg_share_construction_func():
    """Calculate share of new construction by levels (Occupancy Status, Occupancy Status, Housing type, Energy Performance)

    1. Calculate share of Occupancy Status and Heating Energy by Housing type based on individual share.
    2. Calculate share of Occupancy Status, Heating Energy, Housing type based on total number of buildings.
    3. De-aggregate by adding the independent share of 'Energy performance'.
    """

    os_he_share_ht_construction = de_aggregate_columns(parameters_dict['os_share_ht_construction'],
                                                       parameters_dict['he_share_ht_construction'])
    os_he_ht_share_tot_construction = ds_mul_df(parameters_dict['ht_share_tot_construction'],
                                                os_he_share_ht_construction).stack().stack()

    seg_share_construction = de_aggregating_series(os_he_ht_share_tot_construction,
                                                   parameters_dict['ep_share_tot_construction'],
                                                   level='Energy performance')
    seg_share_os_ht_construction = val2share(seg_share_construction, ['Occupancy status', 'Housing type'],
                                             option='column')

    seg_share_os_ht_construction = seg_share_os_ht_construction.droplevel(None, axis=1)

    return seg_share_os_ht_construction


def information_rate_func(knowldege_normalize, kind='remaining'):
    """Returns information rate. More info_rate is high, more intangible_cost are low.

    Intangible renovation costs decrease according to a logistic curve with the same cumulative
    production so as to capture peer effects and knowledge diffusion.
    intangible_cost[yr] = intangible_cost[calibration_year] * info_rate with info rate [1-info_rate_max ; 1]
    This function calibrate a logistic function, so rate of decrease is set at  25% for a doubling of cumulative
    production.
    """
    if kind == 'remaining':
        sh = technical_progress_dict['information_rate_max']
        alpha = technical_progress_dict['information_rate_intangible']
    elif kind == 'new':
        sh = technical_progress_dict['information_rate_max_new']
        alpha = technical_progress_dict['information_rate_intangible_new']
    else:
        raise ValueError

    def equations(p, sh=sh, alpha=alpha):
        a, r = p
        return (1 + a * np.exp(-r)) ** -1 - sh, (1 + a * np.exp(-2 * r)) ** -1 - sh - (1 - alpha) * sh + 1

    a, r = fsolve(equations, (1, -1))

    return logistic(knowldege_normalize, a=a, r=r) + 1 - sh


def learning_by_doing_func(knowledge_normalize, learning_rate, yr, cost_new, cost_new_lim_ds, calibration_yr):
    """ Calculate new cost after considering learning-by-doing effect.

    Investment costs decrease exponentially with the cumulative sum of operations so as to capture
    the classical “learning-by-doing” process.
    """
    learning_by_doing_new = knowledge_normalize ** (np.log(1 + learning_rate) / np.log(2))
    learning_by_doing_new_reindex = reindex_mi(learning_by_doing_new, cost_new_lim_ds.index,
                                               ['Heating energy', 'Energy performance'])
    cost_new[yr] = cost_new[calibration_yr] * learning_by_doing_new_reindex + cost_new_lim_ds * (
            1 - learning_by_doing_new_reindex)
    return cost_new


def calibration_market_share(lcc_df, logging):
    """Returns intangible costs by calibrating market_share.

    TODO: Calibration of intangible cost could be based on absolue value instead of market share.
    """
    # remove idx when label = 'A' (no transition) and label = 'B' (intangible_cost = 0)
    lcc_useful = remove_rows(lcc_df, 'Energy performance', 'A')
    lcc_useful = remove_rows(lcc_useful, 'Energy performance', 'B')

    market_share_temp = lcc2market_share(lcc_useful)
    market_share_objective = reindex_mi(calibration_dict['market_share'], market_share_temp.index, ['Energy performance'])
    market_share_objective = market_share_objective.reindex(market_share_temp.columns, axis=1)

    logging.debug('Approximation of market-share objective')

    def approximate_ms_objective(ms_obj, ms):
        """Treatment of market share objective to facilitate resolution.
        """
        if isinstance(ms, pd.DataFrame):
            ms = ms.loc[ms_obj.name, :]

        idx = (ms.index[ms < 0.005]).intersection(ms_obj.index[ms_obj < 0.005])
        ms_obj[idx] = ms[idx]
        idx = ms_obj[ms_obj > 10 * ms].index
        ms_obj[idx] = 5 * ms[idx]
        idx = ms_obj[ms_obj < ms / 10].index
        ms_obj[idx] = ms[idx] / 5

        return ms_obj / ms_obj.sum()

    ms_obj_approx = market_share_objective.apply(approximate_ms_objective, args=[market_share_temp], axis=1)
    ms_obj_approx.to_pickle(os.path.join(folder['calibration_intermediate'], 'ms_obj_approx.pkl'))

    def solve_intangible_cost(factor, lcc_np, ms_obj, ini=0):
        """Try to solve the equation with lambda=factor.
        """
        def func(intangible_cost_np, lcc, ms, factor):
            """Functions of intangible_cost that are equal to 0.

            Returns a vector that should converge toward 0 as intangible cost converge toward optimal.
            """
            result = np.empty(lcc.shape[0])
            market_share_np = (lcc + intangible_cost_np ** 2) ** -parameters_dict['nu_intangible_cost'] / np.sum(
                (lcc + intangible_cost_np ** 2) ** -parameters_dict['nu_intangible_cost'])
            result[:-1] = market_share_np[:-1] - ms[:-1]
            result[-1] = np.sum(intangible_cost_np ** 2) / np.sum(lcc + intangible_cost_np ** 2) - factor
            return result

        x0 = lcc_np * ini
        root, info_dict, ier, message = fsolve(func, x0, args=(lcc_np, ms_obj, factor), full_output=True)

        if ier == 1:
            return ier, root

        else:
            # logging.debug(message)
            return ier, None

    logging.debug('Calibration of intangible cost')

    lambda_min = 0.01
    lambda_max = 0.6
    step = 0.01

    idx_list, lambda_list, intangible_list = [], [], []
    num_label = list(lcc_df.index.names).index('Energy performance')
    for idx in lcc_useful.index:
        num_ini = language_dict['energy_performance_list'].index(idx[num_label])
        labels_final = language_dict['energy_performance_list'][num_ini + 1:]
        # intangible cost would be for index = idx, and labels_final.
        for lambda_current in range(int(lambda_min * 100), int(lambda_max * 100), int(step * 100)):
            lambda_current = lambda_current / 100
            lcc_row_np = lcc_df.loc[idx, labels_final].to_numpy()
            ms_obj_np = ms_obj_approx.loc[idx, labels_final].to_numpy()
            ier, root = solve_intangible_cost(lambda_current, lcc_row_np, ms_obj_np)
            if ier == 1:
                lambda_list += [lambda_current]
                idx_list += [idx]
                intangible_list += [pd.Series(root ** 2, index=labels_final)]
                # func(root, lcc_row_np, ms_obj_np, lambda_current)
                break

    intangible_cost = pd.concat(intangible_list, axis=1).T
    intangible_cost.index = pd.MultiIndex.from_tuples(idx_list)
    intangible_cost.index.names = lcc_df.index.names

    assert len(lcc_useful.index) == len(idx_list), "Calibration didn't work for all segments"

    intangible_cost.to_pickle(os.path.join(folder['calibration_intermediate'], 'intangible_cost.pkl'))
    logging.debug('Average lambda factor: {:.0f}%'. format(sum(lambda_list) / len(lambda_list) * 100))
    intangible_cost_mean = intangible_cost.groupby('Energy performance', axis=0).mean()
    intangible_cost_mean = intangible_cost_mean.loc[intangible_cost_mean.index[::-1], :]
    logging.debug('Intangible cost (€/m2): \n {}'.format(intangible_cost_mean))
    return intangible_cost


def calibration_market_share_construction(lcc_construction, logging):
    """Returns intangible costs construction by calibrating market_share.

    In Scilab intangible cost are calculated with conventional consumption.
    """

    seg_share_os_ht_construction = seg_share_construction_func()
    seg_share_os_ht_construction.sort_index(inplace=True)
    lcc_construction = lcc_construction.reorder_levels(seg_share_os_ht_construction.index.names)
    lcc_construction.sort_index(inplace=True)

    """denum = (lcc_construction ** -parameters_dict['nu_new']).sum(axis=1) ** -1
    ms_seg_construction = ds_mul_df(denum, lcc_construction ** -parameters_dict['nu_new'])"""

    def approximate_ms_objective(ms_obj):
        """Treatment of market share objective to facilitate resolution.
        """
        ms_obj[ms_obj == 0] = 0.001
        return ds_mul_df(ms_obj.sum(axis=1) ** -1, ms_obj)

    seg_share_os_ht_construction = approximate_ms_objective(seg_share_os_ht_construction)

    def solve_intangible_cost(factor, lcc_np, ms_obj, ini=0.0):
        """Try to solve the equation with lambda=factor.
        """
        def func(intangible_cost_np, lcc, ms, factor):
            """Functions of intangible_cost that are equal to 0.

            Returns a vector that should converge toward 0 as intangible cost converge toward optimal.
            """
            result = np.empty(lcc.shape[0])
            market_share_np = (lcc + intangible_cost_np ** 2) ** -parameters_dict['nu_new'] / np.sum(
               (lcc + intangible_cost_np ** 2) ** -parameters_dict['nu_new'])
            result[:-1] = market_share_np[:-1] - ms[:-1]
            result[-1] = np.sum(intangible_cost_np ** 2) / np.sum(lcc + intangible_cost_np ** 2) - factor
            return result

        x0 = lcc_np * ini
        root, info_dict, ier, message = fsolve(func, x0, args=(lcc_np, ms_obj, factor), full_output=True)

        if ier == 1:
            return ier, root

        else:
            # logging.debug(message)
            return ier, None

    logging.debug('Calibration of intangible cost')

    lambda_min = 0.01
    lambda_max = 0.6
    step = 0.01

    assert (lcc_construction.index == seg_share_os_ht_construction.index).all()

    labels_final = lcc_construction.columns
    idx_list, lambda_list, intangible_list = [], [], []
    for idx in lcc_construction.index:
        for lambda_current in range(int(lambda_min * 100), int(lambda_max * 100), int(step * 100)):
            lambda_current = lambda_current / 100
            lcc_row_np = lcc_construction.loc[idx, :].to_numpy()
            ms_obj_np = seg_share_os_ht_construction.loc[idx, :].to_numpy()
            ier, root = solve_intangible_cost(lambda_current, lcc_row_np, ms_obj_np)
            if ier == 1:
                lambda_list += [lambda_current]
                idx_list += [idx]
                intangible_list += [pd.Series(root ** 2, index=labels_final)]
                break

    intangible_cost = pd.concat(intangible_list, axis=1).T
    intangible_cost.index = pd.MultiIndex.from_tuples(idx_list)
    intangible_cost.index.names = lcc_construction.index.names

    intangible_cost.to_pickle(os.path.join(folder['calibration_intermediate'], 'intangible_cost_construction.pkl'))
    logging.debug('Lambda factor: {:.0f}%'. format(lambda_current * 100))
    return intangible_cost


def calibration_renovation_rate(npv_df, stock_ini_seg):
    renovation_rate_calibration = reindex_mi(calibration_dict['renovation_rate_decision_maker'], npv_df.index,
                                             ['Occupancy status', 'Housing type'])
    rho = (np.log(parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1) - np.log(
        parameters_dict['rate_max'] / renovation_rate_calibration - 1)) / (npv_df - parameters_dict['npv_min'])

    stock_ini_wooner = stock_ini_seg.groupby(
        [lvl for lvl in language_dict['levels_names'] if lvl != 'Income class owner']).sum()
    seg_stock_dm = stock_ini_wooner.to_frame().pivot_table(index=['Occupancy status', 'Housing type'],
                                                        columns=['Energy performance', 'Heating energy', 'Income class'])
    seg_stock_dm = seg_stock_dm.droplevel(None, axis=1)

    weight_dm = ds_mul_df((stock_ini_wooner.groupby(['Occupancy status', 'Housing type']).sum()) ** -1, seg_stock_dm)
    rho = rho.droplevel('Income class owner', axis=0)
    rho = rho[~rho.index.duplicated()]
    rho_df = rho.to_frame().pivot_table(index=weight_dm.index.names, columns=weight_dm.columns.names)
    rho_df = rho_df.droplevel(None, axis=1)
    rho_dm = (rho_df * weight_dm).sum(axis=1)
    rho_seg = reindex_mi(rho_dm, stock_ini_seg.index, rho_dm.index.names)

    return rho_seg


def area2knowledge_renovation(flow_area_seg, stock_knowledge_ep, yr, yr_ini):
    """Returns knowledge and stock_knowledge based on the area stock renovated.

    flow_area_seg must be
    If yr == yr_ini, initialize stock_knowledge_ep.
    Stock_knowledge_ini is defined as the renovation rate (2.7%/yr) x number of years (10 yrs) x renovated area (m2).
    """

    if stock_knowledge_ep == {}:
        renovation_rate_dm = reindex_mi(calibration_dict['renovation_rate_decision_maker'], flow_area_seg.index,
                                        calibration_dict['renovation_rate_decision_maker'].index.names)
        flow_area_renovated_seg_ini = renovation_rate_dm * flow_area_seg * technical_progress_dict['learning_year']
        flow_area_renovated_ep = flow_area_renovated_seg_ini.groupby(['Energy performance']).sum()
    else:
        flow_area_renovated_ep = flow_area_seg.groupby('Energy performance', axis=1).sum().sum()

    # knowledge_renovation_ini depends on energy performance final
    flow_knowledge_renovation = pd.Series(dtype='float64',
                                          index=[ep for ep in language_dict['energy_performance_list'] if ep != 'G'])
    flow_knowledge_renovation.loc['A'] = flow_area_renovated_ep.loc['A'] + flow_area_renovated_ep.loc['B']
    flow_knowledge_renovation.loc['B'] = flow_area_renovated_ep.loc['A'] + flow_area_renovated_ep.loc['B']
    flow_knowledge_renovation.loc['C'] = flow_area_renovated_ep.loc['C'] + flow_area_renovated_ep.loc['D']
    flow_knowledge_renovation.loc['D'] = flow_area_renovated_ep.loc['C'] + flow_area_renovated_ep.loc['D']
    flow_knowledge_renovation.loc['E'] = flow_area_renovated_ep.loc['E'] + flow_area_renovated_ep.loc['F']
    flow_knowledge_renovation.loc['F'] = flow_area_renovated_ep.loc['E'] + flow_area_renovated_ep.loc['F']

    if stock_knowledge_ep != {}:
        stock_knowledge_ep[yr] = stock_knowledge_ep[yr - 1] + flow_knowledge_renovation
    else:
        stock_knowledge_ep[yr] = flow_knowledge_renovation
    knowledge = stock_knowledge_ep[yr] / stock_knowledge_ep[yr_ini + 1]

    return knowledge, stock_knowledge_ep


def output2csv(dict_output, val, logging):
    name_file = os.path.join(folder['output'], val.replace(' ', '_') + '.csv')
    first_element = list(dict_output[val].items())[0][1]
    if isinstance(first_element, pd.Series):
        pd.concat(dict_output[val], axis=1).to_csv(name_file)
        logging.debug('Output: {}'.format(name_file))
    elif isinstance(first_element, pd.DataFrame):
        temp = [item.stack(item.columns.names) for key, item in dict_output[val].items()]
        temp = pd.concat(temp, axis=1)
        temp.columns = dict_output[val].keys()
        temp.to_csv(name_file)
        logging.debug('Output: {}'.format(name_file))