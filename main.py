import pandas as pd
import numpy as np
import logging
import os
import time
import pickle
from scipy.optimize import fsolve

from input import language_dict, parameters_dict, index_year, folder, exogenous_dict, cost_dict, calibration_dict
from func import *


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

    cost_invest_df = cost_dict['cost_inv']
    cost_invest_df.replace({0: float('nan')}, inplace=True)
    cost_switch_fuel_df = cost_dict['cost_switch_fuel']
    cost_intangible_df = cost_dict['cost_intangible']

    name_file = os.path.join(folder['middle'], 'parc.pkl')
    logging.debug('Loading parc pickle file {}'.format(name_file))
    dsp = pd.read_pickle(name_file)
    logging.debug('Total number of housing in this study {:,.0f}'.format(dsp.sum()))
    segments = pd.MultiIndex.from_tuples(dsp.index.tolist())
    logging.debug('Total number of segments in this study {:,}'.format(len(segments)))

    # distribution housing by type
    nb_decision_maker = dsp.groupby(['Occupancy status', 'Housing type']).sum()
    distribution_decision_maker = nb_decision_maker / dsp.sum()

    # recalibrating number of housing based on total population
    population_series = exogenous_dict['population_total_series'] * (dsp.sum() / exogenous_dict['stock_ini'])
    nb_population_housing_ini = population_series.at[calibration_year] / dsp.sum()

    # surface
    idx_occ = dsp.index.get_level_values('Occupancy status')
    idx_housing = dsp.index.get_level_values('Housing type')
    idx_surface = pd.MultiIndex.from_tuples(list(zip(list(idx_occ), list(idx_housing))))
    surface_ds = parameters_dict['surface'].reindex(idx_surface)
    surface_ds.index = dsp.index
    surface_total_ds = surface_ds * dsp
    surface_total_ini = surface_total_ds.sum()
    surface_average_ini = surface_total_ini / dsp.sum()

    # to initialize knowledge
    dsp_new = pd.concat((dsp.xs('A', level='Energy performance'), dsp.xs('B', level='Energy performance')))
    idx_occ = dsp_new.index.get_level_values('Occupancy status')
    idx_housing = dsp_new.index.get_level_values('Housing type')
    idx_surface = pd.MultiIndex.from_tuples(list(zip(list(idx_occ), list(idx_housing))))
    surface_ds = parameters_dict['surface'].reindex(idx_surface)
    surface_ds.index = dsp_new.index
    knowledge_ini = pd.Series([2.5 * (surface_ds * dsp_new).sum(), 2 * (surface_ds * dsp_new).sum()], index=['BBC', 'BEPOS'])

    technical_progress_dict = parameters_dict['technical_progress_dict']

    # initializing investment cost new
    cost_new_ds = pd.concat([cost_dict['cost_new']] * 2,
                                           keys=['Homeowners', 'Landlords'], names=['Occupancy status'])

    cost_new_lim_ds = pd.concat([cost_dict['cost_new_lim']] * len(language_dict['energy_performance_new_list']),
                                           keys=language_dict['energy_performance_new_list'], names=['Energy performance'])
    cost_new_lim_ds = pd.concat([cost_new_lim_ds] * len(language_dict['heating_energy_list']),
                                           keys=language_dict['heating_energy_list'], names=['Heating energy'])
    cost_new_lim_ds = cost_new_lim_ds.reorder_levels(cost_new_ds.index.names)


    energy_prices_df = exogenous_dict['energy_price_data']

    logging.debug('Calculate life cycle cost for each possible transition')
    discount_factor = discount_factor_func(segments)
    energy_cost_ts_df = energy_cost_func(segments, energy_prices_df)[3]
    energy_lcc_ds = energy_cost_ts_df.sum(axis=1)
    energy_discount_lcc_ds = pd.DataFrame(discount_factor.values * energy_lcc_ds.values, index=segments)
    energy_discount_lcc_ds.index.names = language_dict['properties_names']
    energy_discount_lcc_ds.columns = ['Values']

    logging.debug('Calculate life cycle cost for each possible transition')
    lcc_df = lcc_func(energy_discount_lcc_ds, cost_invest_df, cost_switch_fuel_df, cost_intangible_df)
    # lcc_grouped_df = lcc_df.groupby(by='Energy performance', axis=0).mean().groupby(by='Energy performance', axis=1).mean()

    if False:
        for occ_status, housing_type, decile in language_dict['decision_maker_income_list']:
            break

        market_share_temp = market_share_func(lcc_df)
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
                market_share_ds = market_share_func(lcc_ds + intangible_cost_ds**2)
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
                # print(market_share_func(lcc_ds + intangible_cost_ds))
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
    market_share_df = market_share_func(lcc_df)
    market_share_grouped_df = market_share_df.groupby(by='Energy performance', axis=1).sum().groupby(by='Energy performance', axis=0).mean()

    logging.debug('Calculate net present value for each segment')
    pv_df = (market_share_df * lcc_df).sum(axis=1)
    npv_df = energy_discount_lcc_ds.iloc[:, 0] - pv_df

    if True:
        # concatenate energy performance and decision market renovation rate
        logging.debug('Calibration of rho by renovation rate')
        renovation_rate_calibration = calibration_dict['renovation_rate_decision_maker'].copy()

        idx_occ = npv_df.index.get_level_values('Occupancy status')
        idx_housing = npv_df.index.get_level_values('Housing type')
        idx_renovation = pd.MultiIndex.from_tuples(list(zip(list(idx_occ), list(idx_housing))))
        renovation_rate_calibration = renovation_rate_calibration.reindex(idx_renovation)
        renovation_rate_calibration.index = npv_df.index
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

    logging.debug('Initialization of variables')
    # initialization
    nb_population_housing = dict()
    nb_population_housing[calibration_year] = nb_population_housing_ini
    nb_total_housing = dict()
    nb_total_housing[calibration_year] = dsp.sum()
    nb_existing_housing = dict()
    nb_existing_housing[calibration_year] = dsp.sum()
    nb_new_housing = dict()
    nb_new_housing[calibration_year] = 0

    surface_total = dict()
    surface_total[calibration_year] = parameters_dict['surface'] * nb_decision_maker
    surface_total_new = dict()
    surface_total_new[calibration_year] = 0
    surface_average_new = dict()
    surface_average_new[calibration_year] = parameters_dict['surface_new']

    elasticity_surface_new = dict()
    temp = parameters_dict['elasticity_surface_new'].reindex(
        parameters_dict['surface_new'].index.get_level_values('Housing type'))
    temp.index = parameters_dict['surface_new'].index
    elasticity_surface_new[calibration_year] = temp

    share_multi_family = dict()
    share_multi_family[calibration_year] = parameters_dict['distribution_housing'].loc['Multi-family']

    knowledge_stock_new = dict()
    knowledge_stock_new[calibration_year] = knowledge_ini
    knowledge_stock_remaining = dict()
    knowledge_stock_remaining[calibration_year] = 0

    cost_new = dict()
    cost_new[calibration_year] = cost_new_ds

    nb_construction = dict()
    eps_nb_population_housing = dict()
    factor_population_housing = dict()
    nb_destruction = dict()
    evolution_housing = dict()

    dsp_ts = dsp
    logging.debug('Start of dynamic evolution of housing from year {} to year {}'.format(index_year[0], index_year[-1]))
    for year in index_year[1:]:
        logging.debug('Year: {}'.format(year))

        # number of persons by housing
        logging.debug('Number of persons by housing')
        nb_population_housing[year] = population_series.loc[year] / nb_total_housing[year - 1]

        eps_nb_population_housing[year] = (nb_population_housing[year] - parameters_dict[
            'nb_population_housing_min']) / (nb_population_housing_ini - parameters_dict['nb_population_housing_min'])
        eps_nb_population_housing[year] = max(0, min(1, eps_nb_population_housing[year]))

        factor_population_housing[year] = parameters_dict['factor_population_housing_ini'] * eps_nb_population_housing[year]
        nb_population_housing[year] = max(parameters_dict['nb_population_housing_min'], nb_population_housing[year] * (
                    1 + factor_population_housing[year]))

        # number of housing
        logging.debug('Number of housing')
        nb_total_housing[year] = population_series.loc[year] / nb_population_housing[year]
        nb_destruction[year] = nb_existing_housing[year - 1] * parameters_dict['destruction_rate']
        nb_existing_housing[year] = nb_existing_housing[year - 1] - nb_destruction[year]
        nb_construction[year] = (nb_total_housing[year] - nb_existing_housing[year]) - nb_destruction[year]
        nb_new_housing[year] = nb_new_housing[year - 1] + nb_construction[year]

        # type of housing
        logging.debug('Type of housing')
        # remaining
        nb_type_destruction = distribution_decision_maker * nb_construction[year]
        nb_type_remaining = distribution_decision_maker * nb_existing_housing[year]

        # new
        evolution_housing[year] = (nb_total_housing[year] - nb_total_housing[calibration_year]) / nb_total_housing[
            calibration_year] * 100
        share_multi_family[year] = 0.1032 * np.log(10.22 * evolution_housing[year] / 10 + 79.43) * parameters_dict[
            'factor_evolution_distribution']
        share_multi_family_construction = (nb_total_housing[year] * share_multi_family[year] - nb_total_housing[
            year - 1] * share_multi_family[year - 1]) / nb_construction[year]
        share_housing_type_new = pd.Series([share_multi_family_construction, 1 - share_multi_family_construction],
                                           index=['Multi-family', 'Single-family'])
        share_housing_type_new = share_housing_type_new.reindex(
            parameters_dict['distribution_type'].index.get_level_values('Housing type'))
        share_housing_type_new.index = parameters_dict['distribution_type'].index
        share_type_new = parameters_dict['distribution_type'] * share_housing_type_new

        nb_type_construction = share_type_new * nb_construction[year]

        # surface housing
        logging.debug('Surface housing')
        # remaining
        surface_type_remaining = nb_type_remaining * parameters_dict['surface']
        surface_type_destruction = nb_type_destruction * parameters_dict['surface']

        # new
        eps_surface_new = (parameters_dict['surface_new_max'] - surface_average_new[year - 1]) / (
                    parameters_dict['surface_new_max'] - surface_average_new[calibration_year])
        eps_surface_new = eps_surface_new.apply(lambda x: max(0, min(1, x)))

        elasticity_surface_new[year] = eps_surface_new.multiply(elasticity_surface_new[calibration_year])
        # TODO factor_surface
        factor_surface_new = elasticity_surface_new[year] * 0

        surface_average_new[year] = pd.concat(
            [parameters_dict['surface_new_max'], surface_average_new[year - 1] * (1 + factor_surface_new)], axis=1).min(
            axis=1)

        surface_construction = surface_average_new[year] * nb_type_construction
        surface_total_new[year] = surface_construction + surface_total_new[year - 1]

        surface_total[year] = surface_type_remaining + surface_total_new[year]
        surface_average = surface_total[year].sum() / nb_total_housing[year]
        surface_population_housing = surface_total[year].sum() / population_series.loc[year]

        # technical progress
        knowledge_stock_new[year] = knowledge_stock_new[year - 1] + surface_construction.sum()
        factor_lbd_new = knowledge_stock_new[year] / knowledge_stock_new[calibration_year]
        learning_by_doing_new = factor_lbd_new ** (np.log(1 + technical_progress_dict['learning-by-doing-new']) / np.log(2))
        learning_by_doing_new_reindex = learning_by_doing_new.reindex(
            cost_new_lim_ds.index.get_level_values('Energy performance'))
        learning_by_doing_new_reindex.index = cost_new_lim_ds.index
        cost_new[year] = (cost_new[calibration_year] - cost_new_lim_ds) * learning_by_doing_new_reindex + cost_new_lim_ds

        # information acceleration
        def information_rate_func(knwoldege, ):
            """
            Calibration of information rate.
            """
            pass


    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
