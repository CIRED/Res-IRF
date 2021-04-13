
import logging
import os
import time
import pickle

from scipy.optimize import fsolve

from input import language_dict, parameters_dict, index_year, folder, exogenous_dict, cost_dict, calibration_dict
from func import *
from function_pandas import *


if __name__ == '__main__':

    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug('Start Res-IRF')

    calibration_year = index_year[0]
    logging.debug('Calibration year: {}'.format(calibration_year))

    name_file = os.path.join(folder['middle'], 'parameter_dict.pkl')
    logging.debug('Dumping parameter_dict pickle file {}'.format(name_file))
    with open(name_file, 'wb') as file:
        pickle.dump(parameters_dict, file)

    name_file = os.path.join(folder['middle'], 'language_dict.pkl')
    logging.debug('Dumping language_dict pickle file {}'.format(name_file))
    with open(name_file, 'wb') as file:
        pickle.dump(language_dict, file)

    # loading cost
    cost_invest_df = cost_dict['cost_inv']
    cost_invest_df.replace({0: float('nan')}, inplace=True)
    cost_switch_fuel_df = cost_dict['cost_switch_fuel']
    cost_intangible_df = cost_dict['cost_intangible']

    # loading parc
    name_file = os.path.join(folder['middle'], 'parc.pkl')
    logging.debug('Loading parc pickle file {}'.format(name_file))
    dsp = pd.read_pickle(name_file)
    logging.debug('Total number of housing in this study {:,.0f}'.format(dsp.sum()))
    segments = dsp.index
    logging.debug('Total number of segments in this study {:,}'.format(len(segments)))

    # distribution housing by type
    nb_decision_maker = dsp.groupby(['Occupancy status', 'Housing type']).sum()
    distribution_decision_maker = nb_decision_maker / dsp.sum()

    # recalibrating number of housing based on total population
    nb_population = exogenous_dict['population_total_series'] * (dsp.sum() / exogenous_dict['stock_ini'])
    nb_population_housing_ini = nb_population.loc[calibration_year] / dsp.sum()

    # surface
    surface = reindex_mi(parameters_dict['surface'], segments, ['Occupancy status', 'Housing type'])
    stock_surface_ds = surface * dsp
    stock_surface_ini = stock_surface_ds.sum()
    average_surface_ini = stock_surface_ini / dsp.sum()

    # initializing knowledge
    dsp_new = pd.concat((dsp.xs('A', level='Energy performance'), dsp.xs('B', level='Energy performance')))
    surface = reindex_mi(parameters_dict['surface'], dsp_new.index, ['Occupancy status', 'Housing type'])
    knowledge_ini = pd.Series([2.5 * (surface * dsp_new).sum(), 2 * (surface * dsp_new).sum()],
                              index=['BBC', 'BEPOS'])
    technical_progress_dict = parameters_dict['technical_progress_dict']

    # initializing investment cost new
    cost_new_ds = pd.concat([cost_dict['cost_new']] * 2, keys=['Homeowners', 'Landlords'], names=['Occupancy status'])
    cost_new_ds.sort_index(inplace=True)
    cost_new_lim_ds = pd.concat([cost_dict['cost_new_lim']] * len(language_dict['energy_performance_new_list']),
                                           keys=language_dict['energy_performance_new_list'], names=['Energy performance'])
    cost_new_lim_ds = pd.concat([cost_new_lim_ds] * len(language_dict['heating_energy_list']),
                                           keys=language_dict['heating_energy_list'], names=['Heating energy'])
    cost_new_lim_ds = cost_new_lim_ds.reorder_levels(cost_new_ds.index.names)
    cost_new_lim_ds.sort_index(inplace=True)

    energy_prices_df = exogenous_dict['energy_price_data']

    logging.debug('Calculate life cycle cost for each possible transition')
    energy_lcc_ds = segments2energy_lcc(segments, energy_prices_df, calibration_year)
    lcc_df = lcc_func(energy_lcc_ds, cost_invest_df, cost_switch_fuel_df, cost_intangible_df, transition='label')
    lcc_df = lcc_df.reorder_levels(energy_lcc_ds.index.names)

    if False:
        for occ_status, housing_type, decile in language_dict['decision_maker_income_list']:
            break

        market_share_temp = lcc2market_share(lcc_df)
        market_share_temp_grouped = market_share_df.groupby(by='Energy performance', axis=0).mean().groupby(
            by='Energy performance', axis=1).mean()

        tuple_index = (slice(None), slice(None), occ_status, housing_type, decile, slice(None))
        market_share_temp.loc[tuple_index, :]
        lambda_min = 0.15
        lambda_max = 0.6
        step = 0.01

        def solve_intangible_cost(factor):

            def func(intangible_cost_np, lcc_ds, factor):
                intangible_cost_ds = pd.Series(intangible_cost_np, index=lcc_ds.index, name=lcc_ds.name)
                market_share_ds = lcc2market_share(lcc_ds + intangible_cost_ds**2)
                result0 = market_share_ds - calibration_dict['market_share'].loc[market_share_ds.name, market_share_ds.index]
                result1 = (intangible_cost_ds**2).sum() / (lcc_ds + intangible_cost_ds**2).sum() - factor
                result0.iloc[-1] = result1
                return result0

            # treatment of market share to facilitate resolution
            # market_share_temp = calibration_dict['market_share'].loc[market_share_ds.name, market_share_ds.index]
            for k in range(0, len()):
                pass

            x0 = lcc_ds.to_numpy() * 0
            root, info_dict, ier, message = fsolve(func, x0, args=(lcc_ds, factor), full_output=True)
            logging.debug(message)

            if ier == 1:
                # checking if solution solve the system
                intangible_cost_ds = pd.Series(root, index=lcc_ds.index, name=lcc_ds.name)
                # print(lcc_ds.sum() / (lcc_ds + intangible_cost_ds).sum())
                # print(lcc2market_share(lcc_ds + intangible_cost_ds))
                # print(calibration_dict['market_share'].loc[lcc_ds.name, :])
                return ier, intangible_cost_ds

            else:
                return ier, None

        logging.debug('Calibration of intangible cost')

        lambda_current = lambda_min
        for lambda_current in range(int(lambda_min * 100), int(lambda_max * 100), int(step * 100)):
            ier, intangible_cost_ds = solve_intangible_cost(lambda_current)
            if ier == 1:
                break

        lcc_ds = lcc_grouped_df.iloc[0]
        lcc_ds = lcc_ds[lcc_ds.index < lcc_ds.name]

    logging.debug('Calculate market share for each possible transition')
    market_share_label = lcc2market_share(lcc_df)

    logging.debug('Calculate net present value for each segment')
    pv_df = (market_share_label * lcc_df).sum(axis=1)
    pv_df = pv_df.replace({0: float('nan')})

    segments_initial = pv_df.index
    energy_initial_lcc_ds = segments2energy_lcc(segments_initial, energy_prices_df, calibration_year)
    npv_df = energy_initial_lcc_ds.iloc[:, 0] - pv_df

    if True:
        # concatenate energy performance and decision market renovation rate
        logging.debug('Calibration of rho by renovation rate')

        renovation_rate_calibration = reindex_mi(calibration_dict['renovation_rate_decision_maker'], npv_df.index,
                                                 ['Occupancy status', 'Housing type'])
        renovation_rate_calibration = renovation_rate_calibration.iloc[:, 0]

        rho = (np.log(parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1) - np.log(
            parameters_dict['rate_max'] / renovation_rate_calibration - 1)) / (npv_df - parameters_dict['npv_min'])

        rho.to_pickle(os.path.join(folder['calibration_middle'], 'rho.pkl'))
        logging.debug('End of calibration and dumping rho.pkl')

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

    logging.debug('Calculate renovation rate for each segment')
    renovation_rate_df = npv_df.reset_index().apply(func, rho=rho, axis=1)
    renovation_rate_df.index = npv_df.index
    stock_renovation = renovation_rate_df * dsp
    logging.debug('Nombre total de rénovations: {:,.0f}'.format(stock_renovation.sum()))

    stock_renovation_label = ds_mul_df(stock_renovation, market_share_label)
    logging.debug('Nombre total de rénovations: {:,.0f}'.format(stock_renovation_label.sum().sum()))

    logging.debug('Heating energy renovation')
    lcc_energy_transition = lcc_func(energy_lcc_ds, cost_invest_df, cost_switch_fuel_df, cost_intangible_df, transition='energy')
    lcc_energy_transition = lcc_energy_transition.reorder_levels(energy_lcc_ds.index.names)
    market_share_energy = lcc2market_share(lcc_energy_transition)
    ms_temp = pd.concat([market_share_energy.T] * len(language_dict['energy_performance_list']),
                                  keys=language_dict['energy_performance_list'], names=['Energy performance'])
    sr_temp = pd.concat([stock_renovation_label.T] * len(language_dict['heating_energy_list']),
                                  keys=language_dict['heating_energy_list'], names=['Heating energy'])
    stock_renovation_label_energy = (sr_temp * ms_temp).T
    logging.debug('Nombre total de rénovations: {:,.0f}'.format(stock_renovation_label_energy.sum().sum()))

    logging.debug('New construction')
    segments_new = segments.droplevel('Energy performance')
    segments_new = segments_new.drop_duplicates()
    segments_new = pd.concat([pd.Series(index=segments_new, dtype='float64')] * len(language_dict['energy_performance_new_list']),
                                           keys=language_dict['energy_performance_new_list'], names=['Energy performance'])
    segments_new = segments_new.reorder_levels(language_dict['properties_names']).index

    lcc_new_ds = segments_new2lcc(segments_new, calibration_year, energy_prices_df, cost_new=cost_dict['cost_new'])

    construction_share_new = val2share(lcc_new_ds, ['Occupancy status', 'Housing type'], func=lambda x: x**-parameters_dict['nu_new'])

    logging.debug('Initialization of variables')
    nb_population_housing, nb_housing_need, stock_housing_remaining, stock_housing_new, type_housing_remaining = dict(), dict(), dict(), dict(), dict()
    nb_population_housing[calibration_year] = nb_population_housing_ini
    nb_housing_need[calibration_year] = dsp.sum()
    stock_housing_remaining[calibration_year] = dsp.sum()
    stock_housing_new[calibration_year] = 0

    stock_surface, stock_surface_new, average_surface_new = dict(), dict(), dict()
    stock_surface[calibration_year] = parameters_dict['surface'] * nb_decision_maker
    stock_surface_new[calibration_year] = 0
    average_surface_new[calibration_year] = parameters_dict['surface_new']

    elasticity_surface_new = dict()
    temp = parameters_dict['elasticity_surface_new'].reindex(
        parameters_dict['surface_new'].index.get_level_values('Housing type'))
    temp.index = parameters_dict['surface_new'].index
    elasticity_surface_new[calibration_year] = temp

    share_multi_family = dict()
    share_multi_family[calibration_year] = parameters_dict['distribution_housing'].loc['Multi-family']

    knowledge_stock_new, knowledge_stock_remaining = dict(), dict()
    knowledge_stock_new[calibration_year] = knowledge_ini
    knowledge_stock_remaining[calibration_year] = 0

    cost_new = dict()
    cost_new[calibration_year] = cost_new_ds

    nb_housing_construction, type_housing_construction = dict(), dict()
    eps_nb_population_housing, factor_population_housing = dict(), dict()
    nb_housing_destroyed, type_housing_destroyed = dict(), dict()
    trend_housing = dict()

    dsp_ts = dsp

    stock_mobile, stock_remaining, stock_renovation, stock_renovation_df = dict(), dict(), dict(), dict()

    stock_residual = dsp_ts * parameters_dict['destruction_rate']

    stock_remaining[calibration_year] = dsp_ts
    stock_mobile[calibration_year] = stock_remaining[calibration_year] - stock_residual

    renovation_rate = dict()
    renovation_rate[calibration_year] = renovation_rate_df

    logging.debug('Start of dynamic evolution of housing from year {} to year {}'.format(index_year[0], index_year[-1]))
    for year in index_year[1:4]:
        logging.debug('Year: {}'.format(year))

        logging.debug('Number of persons by housing')
        nb_population_housing[year] = nb_population.loc[year] / nb_housing_need[year - 1]
        eps_nb_population_housing[year] = (nb_population_housing[year] - parameters_dict[
            'nb_population_housing_min']) / (nb_population_housing_ini - parameters_dict['nb_population_housing_min'])
        eps_nb_population_housing[year] = max(0, min(1, eps_nb_population_housing[year]))
        factor_population_housing[year] = parameters_dict['factor_population_housing_ini'] * eps_nb_population_housing[year]
        nb_population_housing[year] = max(parameters_dict['nb_population_housing_min'], nb_population_housing[year] * (
                    1 + factor_population_housing[year]))

        logging.debug('Number of housing')
        nb_housing_need[year] = nb_population.loc[year] / nb_population_housing[year]
        nb_housing_destroyed[year] = stock_housing_remaining[year - 1] * parameters_dict['destruction_rate']
        stock_housing_remaining[year] = stock_housing_remaining[year - 1] - nb_housing_destroyed[year]
        nb_housing_construction[year] = nb_housing_need[year] - stock_housing_remaining[year]
        stock_housing_new[year] = stock_housing_new[year - 1] + nb_housing_construction[year]

        logging.debug('Type of housing')
        type_housing_destroyed[year] = distribution_decision_maker * nb_housing_destroyed[year]
        type_housing_remaining[year] = distribution_decision_maker * stock_housing_remaining[year]

        trend_housing[year] = (nb_housing_need[year] - nb_housing_need[calibration_year]) / nb_housing_need[
            calibration_year] * 100
        share_multi_family[year] = 0.1032 * np.log(10.22 * trend_housing[year] / 10 + 79.43) * parameters_dict[
            'factor_evolution_distribution']
        share_multi_family_construction = (nb_housing_need[year] * share_multi_family[year] - nb_housing_need[
            year - 1] * share_multi_family[year - 1]) / nb_housing_construction[year]
        share_type_housing_new = pd.Series([share_multi_family_construction, 1 - share_multi_family_construction],
                                           index=['Multi-family', 'Single-family'])
        share_type_housing_new = reindex_mi(share_type_housing_new, parameters_dict['distribution_type'].index,
                                            ['Housing type'])
        share_type_new = parameters_dict['distribution_type'] * share_type_housing_new
        type_housing_construction[year] = share_type_new * nb_housing_construction[year]

        logging.debug('Surface housing')
        type_surface_remaining = type_housing_remaining[year] * parameters_dict['surface']
        type_surface_destroyed = type_housing_destroyed[year] * parameters_dict['surface']

        eps_surface_new = (parameters_dict['surface_new_max'] - average_surface_new[year - 1]) / (
                    parameters_dict['surface_new_max'] - average_surface_new[calibration_year])
        eps_surface_new = eps_surface_new.apply(lambda x: max(0, min(1, x)))

        elasticity_surface_new[year] = eps_surface_new.multiply(elasticity_surface_new[calibration_year])
        # TODO factor_surface
        factor_surface_new = elasticity_surface_new[year] * 0

        average_surface_new[year] = pd.concat(
            [parameters_dict['surface_new_max'], average_surface_new[year - 1] * (1 + factor_surface_new)], axis=1).min(
            axis=1)
        type_surface_construction = average_surface_new[year] * type_housing_construction[year]
        stock_surface_new[year] = type_surface_construction + stock_surface_new[year - 1]
        stock_surface[year] = type_surface_remaining + stock_surface_new[year]
        average_surface = stock_surface[year].sum() / nb_housing_need[year]
        surface_population_housing = stock_surface[year].sum() / nb_population.loc[year]

        logging.debug('Technical progress: Learning by doing - New stock')
        knowledge_stock_new[year] = knowledge_stock_new[year - 1] + type_surface_construction.sum()
        knowledge_normalize_new = knowledge_stock_new[year] / knowledge_stock_new[calibration_year]

        def learning_by_doing_func(knowledge_normalize, learning_rate, yr, cost_new):

            learning_by_doing_new = knowledge_normalize ** (np.log(1 + learning_rate) / np.log(2))
            learning_by_doing_new_reindex = reindex_mi(learning_by_doing_new, cost_new_lim_ds.index, ['Energy performance'])
            cost_new[yr] = cost_new[calibration_year] * learning_by_doing_new_reindex + cost_new_lim_ds * (
                        1 - learning_by_doing_new_reindex)
            return cost_new

        learning_rate = technical_progress_dict['learning-by-doing-new']
        cost_new = learning_by_doing_func(knowledge_normalize_new, learning_rate, year, cost_new)

        logging.debug('Information acceleration - New stock')

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

        information_rate_new = information_rate_func(knowledge_normalize_new, kind='new')
        knowledge_stock_remaining[year] = knowledge_stock_remaining[year - 1] + type_surface_construction.sum()

        # TODO calculate knowledge_normalize_remaining based on renovation previous year
        logging.debug('Technical progress: Learning by doing - remaining stock')
        logging.debug('Information acceleration - remaining stock')
        information_rate = information_rate_func(knowledge_normalize_new, kind='remaining')

        logging.debug('destroyed process')
        stock_mobile[year] = stock_remaining[year - 1] - stock_residual

        logging.debug('Catching less efficient label for each segment')

        stock_destroyed = stock_mobile2stock_destroyed(stock_mobile[year], stock_mobile[calibration_year],
                                                       stock_remaining[year - 1],
                                                       type_housing_destroyed[year], logging)

        stock_remaining[year] = stock_mobile[year] - stock_destroyed
        logging.debug('Renovation choice')
        result_dict = segments2renovation_rate(segments, year, energy_prices_df, cost_invest_df, cost_switch_fuel_df,
                                 cost_intangible_df, rho, transition='label')
        renovation_rate[year] = result_dict['Renovation rate']
        """renovation_rate[year] = renovation_rate[year].groupby(
            ['Occupancy status', 'Housing type', 'Energy performance', 'Heating energy', 'Income class']).sum()"""
        stock_renovation[year] = renovation_rate[year] * stock_remaining[year]
        nb_renovation = stock_renovation[year].sum()
        nb_renovation / stock_remaining[calibration_year].sum()

        market_share = result_dict['Market share']
        """"market_share = market_share.groupby(
            ['Occupancy status', 'Housing type', 'Energy performance', 'Heating energy', 'Income class']).sum()"""

        stock_renovation_df[year] = pd.DataFrame(market_share.values.T * stock_renovation[year].values,
                                                 index=market_share.columns, columns=market_share.index).T

        nb_renovation_label = dict()
        for column in stock_renovation_df[year].columns:
            nb_renovation_label[column] = stock_renovation_df[year].loc[:, column].sum()

        print('pause')

        logging.debug('Dynamic of new')


    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
