
import logging
import os
import time
import pickle


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

    # calculating owner income distribution in the parc
    levels = [lvl for lvl in dsp.index.names if lvl not in ['Income class owner', 'Energy performance']]
    ds_income_owner_prop = val2share(dsp, levels, option='column')
    ds_income_owner_prop = ds_income_owner_prop.groupby('Income class owner', axis=1).sum().stack()

    # distribution housing by decision-maker
    nb_decision_maker = dsp.groupby(['Occupancy status', 'Housing type']).sum()
    distribution_decision_maker = nb_decision_maker / dsp.sum()

    # recalibrating number of housing based on total population
    nb_population = exogenous_dict['population_total_ds'] * (dsp.sum() / exogenous_dict['stock_ini'])
    nb_population_housing_ini = nb_population.loc[calibration_year] / dsp.sum()

    # initializing area
    area = reindex_mi(parameters_dict['area'], segments, ['Occupancy status', 'Housing type'])
    stock_area_ds = area * dsp
    stock_area_ini = stock_area_ds.sum()
    average_area_ini = stock_area_ini / dsp.sum()

    # initializing knowledge
    dsp_new = pd.concat((dsp.xs('A', level='Energy performance'), dsp.xs('B', level='Energy performance')))
    area = reindex_mi(parameters_dict['area'], dsp_new.index, ['Occupancy status', 'Housing type'])
    knowledge_ini = pd.Series([2.5 * (area * dsp_new).sum(), 2 * (area * dsp_new).sum()],
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
    lcc_df = cost2lcc(energy_lcc_ds, cost_invest_df, cost_switch_fuel_df, cost_intangible_df, transition='label')
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

    logging.debug('Calculate renovation rate for each segment')
    renovation_rate_df = npv_df.reset_index().apply(renov_rate_func, rho=rho, axis=1)
    renovation_rate_df.index = npv_df.index
    stock_renovation = renovation_rate_df * dsp
    logging.debug('Renovation number: {:,.0f}'.format(stock_renovation.sum()))

    stock_renovation_label = ds_mul_df(stock_renovation, market_share_label)
    logging.debug('Renovation number: {:,.0f}'.format(stock_renovation_label.sum().sum()))
    logging.debug('Heating energy renovation')
    stock_renovation_label_energy = renovation_label2renovation_label_energy(energy_lcc_ds, cost_invest_df,
                                                                             cost_switch_fuel_df, cost_intangible_df,
                                                                             stock_renovation_label)
    logging.debug('Renovation number: {:,.0f}'.format(stock_renovation_label_energy.sum().sum()))

    logging.debug('New construction')

    logging.debug('Initialization of variables')
    nb_population_housing, flow_needed, stock_remained, stock_constructed, flow_remained_seg_dm = dict(), dict(), dict(), dict(), dict()
    nb_population_housing[calibration_year] = nb_population_housing_ini
    flow_needed[calibration_year] = dsp.sum()
    stock_remained[calibration_year] = dsp.sum()
    stock_constructed[calibration_year] = 0

    stock_area, stock_area_constructed, dict_area_avg_new_seg_dm = dict(), dict(), dict()
    stock_area[calibration_year] = parameters_dict['area'] * nb_decision_maker
    stock_area_constructed[calibration_year] = 0
    dict_area_avg_new_seg_dm[calibration_year] = parameters_dict['area_new']

    distribution_multi_family = dict()
    distribution_multi_family[calibration_year] = parameters_dict['distribution_housing'].loc['Multi-family']

    stock_knowledge_new, knowledge_stock_remained = dict(), dict()
    stock_knowledge_new[calibration_year] = knowledge_ini
    knowledge_stock_remained[calibration_year] = 0

    cost_new = dict()
    cost_new[calibration_year] = cost_new_ds

    flow_constructed, flow_constructed_seg_dm = dict(), dict()
    flow_destroyed, flow_destroyed_seg_dm = dict(), dict()
    trend_housing = dict()

    stock_mobile, stock_remained_seg, stock_renovation, stock_renovation_df = dict(), dict(), dict(), dict()
    stock_residual = dsp * parameters_dict['destruction_rate']
    stock_remained_seg[calibration_year] = dsp
    stock_mobile[calibration_year] = stock_remained_seg[calibration_year] - stock_residual
    stock_remained = dict()
    stock_remained[calibration_year] = dsp.sum()

    renovation_rate = dict()
    renovation_rate[calibration_year] = renovation_rate_df

    logging.debug('Start of dynamic evolution of housing from year {} to year {}'.format(index_year[0], index_year[-1]))
    for year in index_year[1:4]:
        logging.debug('YEAR: {}'.format(year))

        logging.debug('Number of persons by housing')
        nb_population_housing[year] = nb_population_housing_dynamic(nb_population_housing[year - 1],
                                                                    nb_population_housing[calibration_year])

        logging.debug('Dynamic of housing non-segmented')
        flow_needed[year] = nb_population.loc[year] / nb_population_housing[year]
        flow_destroyed[year] = stock_remained[year - 1] * parameters_dict['destruction_rate']
        stock_remained[year] = stock_remained[year - 1].sum() - flow_destroyed[year]
        flow_constructed[year] = flow_needed[year] - stock_remained[year]
        stock_constructed[year] = stock_constructed[year - 1] + flow_constructed[year]

        logging.debug('Distribution of buildings by decision-maker')
        flow_destroyed_seg_dm[year] = distribution_decision_maker * flow_destroyed[year]
        flow_remained_seg_dm[year] = distribution_decision_maker * stock_remained[year]
        flow_constructed_seg_dm[year], distribution_multi_family = housing_need2housing_construction(flow_needed,
                                                                                                     flow_constructed,
                                                                                                     distribution_multi_family,
                                                                                                     year,
                                                                                                     calibration_year)
        logging.debug('area housing')
        flow_area_remained_seg_dm = flow_remained_seg_dm[year] * parameters_dict['area']
        flow_area_destroyed_seg_dm = flow_destroyed_seg_dm[year] * parameters_dict['area']
        dict_area_avg_new_seg_dm = area_new_dynamic(dict_area_avg_new_seg_dm, year, calibration_year)
        flow_area_constructed_seg_dm = dict_area_avg_new_seg_dm[year] * flow_constructed_seg_dm[year]
        stock_area_constructed[year] = flow_area_constructed_seg_dm + stock_area_constructed[year - 1]
        stock_area[year] = flow_area_remained_seg_dm + stock_area_constructed[year]
        average_area = stock_area[year].sum() / flow_needed[year]
        area_population_housing = stock_area[year].sum() / nb_population.loc[year]

        logging.debug('Technical progress: Learning by doing - New stock')
        stock_knowledge_new[year] = stock_knowledge_new[year - 1] + flow_area_constructed_seg_dm.sum()
        knowledge_normalize_new = stock_knowledge_new[year] / stock_knowledge_new[calibration_year]
        learning_rate = technical_progress_dict['learning-by-doing-new']
        cost_new = learning_by_doing_func(knowledge_normalize_new, learning_rate, year, cost_new, cost_new_lim_ds,
                                          calibration_year)
        logging.debug('Information acceleration - New stock')
        information_rate_new = information_rate_func(knowledge_normalize_new, kind='new')

        # TODO calculate knowledge_normalize_remaining based on renovation previous year
        logging.debug('Technical progress: Learning by doing - remaining stock')
        knowledge_stock_remained[year] = knowledge_stock_remained[year - 1] + flow_area_constructed_seg_dm.sum()
        logging.debug('Information acceleration - remaining stock')
        information_rate = information_rate_func(knowledge_normalize_new, kind='remaining')

        logging.debug('Demolition process')
        stock_mobile[year] = stock_remained[year - 1] - stock_residual
        logging.debug('Catching less efficient label for each segment')
        stock_destroyed_seg = stock_mobile2stock_destroyed(stock_mobile[year], stock_mobile[calibration_year],
                                                           stock_remained[year - 1],
                                                           flow_destroyed_seg_dm[year], logging)
        stock_remained[year] = stock_mobile[year] - stock_destroyed_seg

        logging.debug('Renovation choice')
        result_dict = segments2renovation_rate(segments, year, energy_prices_df, cost_invest_df, cost_switch_fuel_df,
                                               cost_intangible_df, rho, transition='label')
        renovation_rate[year] = result_dict['Renovation rate']
        stock_renovation[year] = renovation_rate[year] * stock_remained[year]
        nb_renovation = stock_renovation[year].sum()
        logging.debug('Renovation number: {:,.0f} buildings'.format(nb_renovation))
        logging.debug('Renovation rate: {:.0f}%'.format((nb_renovation / stock_remained[calibration_year].sum()) * 100))
        market_share = result_dict['Market share']
        stock_renovation_label = ds_mul_df(stock_renovation[year], market_share)
        energy_lcc_ds = result_dict['Energy LCC']
        stock_renovation_label_energy = renovation_label2renovation_label_energy(energy_lcc_ds, cost_invest_df,
                                                                                 cost_switch_fuel_df,
                                                                                 cost_intangible_df,
                                                                                 stock_renovation_label)
        logging.debug('Renovation number: {:,.0f} buildings'.format(stock_renovation_label_energy.sum().sum()))
        stock_renovation_by_label = stock_renovation_label_energy.groupby('Energy performance', axis=1).sum().sum()

        logging.debug('Dynamic of new')
        segments_new = segments2segments_new(segments)
        flow_constructed_new_seg = segments_new2flow_constructed(flow_constructed_seg_dm[year], segments_new,
                                                                   energy_prices_df, ds_income_owner_prop,
                                                                   calibration_year)

        print('pause')

    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
