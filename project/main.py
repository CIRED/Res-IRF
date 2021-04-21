
import logging
import os
import time
import pickle


from input import folder
from func import *
from function_pandas import *


# TODO: calibration, adding public policies, myopic behavior of agent, documentation, unit test
# TODO: sensitivity analysis


if __name__ == '__main__':

    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug('Start Res-IRF')

    # output is a dict storing data useful for the end-user but not necessary for the script.
    output = dict()

    calibration_year = index_year[0]
    logging.debug('Calibration year: {}'.format(calibration_year))

    name_file = os.path.join(folder['intermediate'], 'parameter_dict.pkl')
    logging.debug('Dump parameter_dict pickle file {}'.format(name_file))
    with open(name_file, 'wb') as file:
        pickle.dump(parameters_dict, file)

    name_file = os.path.join(folder['intermediate'], 'language_dict.pkl')
    logging.debug('Dump language_dict pickle file {}'.format(name_file))
    with open(name_file, 'wb') as file:
        pickle.dump(language_dict, file)

    # loading cost
    cost_invest_df = cost_dict['cost_inv']
    cost_invest_df.replace({0: float('nan')}, inplace=True)
    cost_switch_fuel_df = cost_dict['cost_switch_fuel']

    # loading parc
    name_file = os.path.join(folder['intermediate'], 'parc.pkl')
    logging.debug('Loading parc pickle file {}'.format(name_file))
    stock_ini_seg = pd.read_pickle(name_file)
    logging.debug('Total number of housing in this study {:,.0f}'.format(stock_ini_seg.sum()))
    segments = stock_ini_seg.index
    logging.debug('Total number of segments in this study {:,}'.format(len(segments)))

    # calculating owner income distribution in the parc
    levels = [lvl for lvl in stock_ini_seg.index.names if lvl not in ['Income class owner', 'Energy performance']]
    ds_income_owner_prop = val2share(stock_ini_seg, levels, option='column')
    ds_income_owner_prop = ds_income_owner_prop.groupby('Income class owner', axis=1).sum().stack()

    # distribution housing by decision-maker
    # TODO: clarifying val2share & proportion --> Jupyter tutorials
    nb_decision_maker = stock_ini_seg.groupby(['Occupancy status', 'Housing type']).sum()
    distribution_decision_maker = nb_decision_maker / stock_ini_seg.sum()

    # recalibrating number of housing based on total population
    stock_population = exogenous_dict['population_total_ds'] * (stock_ini_seg.sum() / exogenous_dict['stock_ini'])
    nb_population_housing_ini = stock_population.loc[calibration_year] / stock_ini_seg.sum()

    # initializing area
    # TODO: no usage for now
    stock_area_seg = buildings_number2area(stock_ini_seg)
    logging.debug('Total buildings surface: {:,.0f} Mm2'.format(stock_area_seg.sum() / 1000000))

    # initializing knowledge construction - [2.5, 2] * surface A & B existing buildings
    stock_ini_seg_new_existing = pd.concat(
        (stock_ini_seg.xs('A', level='Energy performance'), stock_ini_seg.xs('B', level='Energy performance')), axis=0)
    stock_area_new_existing_seg = buildings_number2area(stock_ini_seg_new_existing)
    knowledge_construction = pd.Series([2.5 * stock_area_new_existing_seg.sum(), 2 * stock_area_new_existing_seg.sum()],
                                  index=['BBC', 'BEPOS'])
    seg_share_tot_construction = seg_share_construction_func()

    technical_progress_dict = parameters_dict['technical_progress_dict']
    # technical_progress_dict['learning_year']
    
    # initializing investment cost new
    cost_new_seg = pd.concat([cost_dict['cost_new']] * 2, keys=['Homeowners', 'Landlords'], names=['Occupancy status'])
    cost_new_seg.sort_index(inplace=True)
    cost_new_lim_seg = pd.concat([cost_dict['cost_new_lim']] * len(language_dict['energy_performance_new_list']),
                                 keys=language_dict['energy_performance_new_list'], names=['Energy performance'])
    cost_new_lim_seg = pd.concat([cost_new_lim_seg] * len(language_dict['heating_energy_list']),
                                 keys=language_dict['heating_energy_list'], names=['Heating energy'])
    cost_new_lim_seg = cost_new_lim_seg.reorder_levels(cost_new_seg.index.names)
    cost_new_lim_seg.sort_index(inplace=True)

    energy_prices_df = exogenous_dict['energy_price_data']
    logging.debug('Calculate life cycle cost for constructions')
    segments_new = segments2segments_new(segments)
    lcc_new_ds = segments_new2lcc(segments_new, calibration_year, energy_prices_df, cost_new=cost_dict['cost_new'])

    logging.debug('Calculate life cycle cost for each possible transition wo intangible cost')
    energy_lcc_ds = segments2energy_lcc(segments, energy_prices_df, calibration_year)
    lcc_df = cost2lcc(energy_lcc_ds, cost_invest=cost_invest_df, transition='label')
    # lcc_df = lcc_df.reorder_levels(energy_lcc_ds.index.names)
    
    market_share_wo_intangible = lcc2market_share(lcc_df)
    if parameters_dict['intangible_cost_source'] == 'pickle':
        name_file = os.path.join(folder['calibration_intermediate'], 'intangible_cost.pkl')
        logging.debug('Loading intangible_cost from {}'.format(name_file))
        cost_intangible_seg = pd.read_pickle(name_file)
        cost_intangible_seg.index.names = language_dict['levels_names']
    elif parameters_dict['intangible_cost_source'] == 'cost_dict':
        cost_intangible_seg = cost_dict['cost_intangible']
    else:
        logging.debug('Calibration of market share with intangible cost'.format(name_file))
        cost_intangible_seg = calibration_market_share(lcc_df, logging)

    logging.debug('Calculate net present value for each segment')
    lcc_df = cost2lcc(energy_lcc_ds, cost_invest=cost_invest_df, cost_intangible=cost_intangible_seg,
                      transition='label')
    logging.debug('Calculate market share for each possible transition')
    market_share_label = lcc2market_share(lcc_df)
    pv_df = (market_share_label * lcc_df).sum(axis=1)
    pv_df = pv_df.replace({0: float('nan')})
    segments_initial = pv_df.index
    energy_initial_lcc_ds = segments2energy_lcc(segments_initial, energy_prices_df, calibration_year)
    npv_df = energy_initial_lcc_ds.iloc[:, 0] - pv_df

    logging.debug('Calibration of renovation rate with rho param')
    rho_seg = calibration_renovation_rate(npv_df)
    rho_seg.to_pickle(os.path.join(folder['calibration_intermediate'], 'rho.pkl'))
    logging.debug('End of calibration and dumping rho.pkl')

    logging.debug('Calculate renovation rate for each segment')
    renovation_rate_seg = npv_df.reset_index().apply(renov_rate_func, rho=rho_seg, axis=1)
    renovation_rate_seg.index = npv_df.index
    flow_renovation_seg = renovation_rate_seg * stock_ini_seg
    logging.debug('Renovation number: {:,.0f}'.format(flow_renovation_seg.sum()))
    output['Renovation rate seg'] = dict()
    output['Renovation rate seg'][calibration_year] = renovation_rate_seg

    flow_renovation_label_seg = ds_mul_df(flow_renovation_seg, market_share_label)
    logging.debug('Renovation number: {:,.0f}'.format(flow_renovation_label_seg.sum().sum()))
    logging.debug('Heating energy renovation')
    stock_renovation_label_energy_seg = renovation_label2renovation_label_energy(energy_lcc_ds,
                                                                                 cost_switch_fuel_df,
                                                                                 flow_renovation_label_seg)
    np.testing.assert_almost_equal(flow_renovation_label_seg.sum().sum(), flow_renovation_seg.sum(), err_msg='Not normal')
    logging.debug('Renovation number: {:,.0f}'.format(stock_renovation_label_energy_seg.sum().sum()))

    logging.debug('Initialization of variables')
    nb_population_housing = dict()
    # TODO: nb_population_housing can be exogeneous
    nb_population_housing[calibration_year] = nb_population_housing_ini

    flow_needed = dict()
    # TODO: flow_needed may not need to be a dict
    flow_needed[calibration_year] = stock_ini_seg.sum()

    stock_constructed_seg, stock_constructed, flow_constructed, flow_constructed_seg_dm = dict(), dict(), dict(), dict()
    # TODO: flow_constructed may be unuseful
    # TODO: flow_constructed_seg_dm may not need to be a dict
    segments_new = segments2segments_new(segments)
    stock_constructed[calibration_year] = 0
    stock_constructed_seg[calibration_year] = pd.Series(0, index=segments_new)

    flow_destroyed, flow_destroyed_seg_dm = dict(), dict()

    stock_remained_seg, stock_remained, flow_remained_seg_dm = dict(), dict(), dict()
    stock_remained_seg[calibration_year] = stock_ini_seg
    stock_remained[calibration_year] = stock_ini_seg.sum()

    stock_residual_seg = stock_ini_seg * parameters_dict['destruction_rate']

    stock_mobile_seg = dict()
    stock_mobile_seg[calibration_year] = stock_remained_seg[calibration_year] - stock_residual_seg

    stock_area, stock_area_constructed, dict_area_avg_new_seg_dm = dict(), dict(), dict()
    stock_area[calibration_year] = parameters_dict['area'] * nb_decision_maker
    stock_area_constructed[calibration_year] = 0
    dict_area_avg_new_seg_dm[calibration_year] = parameters_dict['area_new']

    distribution_multi_family = dict()
    distribution_multi_family[calibration_year] = parameters_dict['ht_share_tot'].loc['Multi-family']

    stock_knowledge_new, stock_knowledge_remained = dict(), dict()
    stock_knowledge_new[calibration_year] = knowledge_construction
    stock_knowledge_remained[calibration_year] = 0

    cost_new = dict()
    cost_new[calibration_year] = cost_new_seg

    logging.debug('Start  dynamic evolution of buildings stock from year {} to year {}'.format(index_year[0], index_year[-1]))
    for year in index_year[1:3]:
        logging.debug('YEAR: {}'.format(year))

        logging.debug('Number of persons by housing')
        nb_population_housing[year] = nb_population_housing_dynamic(nb_population_housing[year - 1],
                                                                    nb_population_housing[calibration_year])

        logging.debug('Dynamic of non-segmented buildings stock')
        flow_needed[year] = stock_population.loc[year] / nb_population_housing[year]
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
        logging.debug('Dynamic area')
        flow_area_remained_seg_dm = flow_remained_seg_dm[year] * parameters_dict['area']
        flow_area_destroyed_seg_dm = flow_destroyed_seg_dm[year] * parameters_dict['area']
        dict_area_avg_new_seg_dm = area_new_dynamic(dict_area_avg_new_seg_dm, year, calibration_year)
        flow_area_constructed_seg_dm = dict_area_avg_new_seg_dm[year] * flow_constructed_seg_dm[year]
        stock_area_constructed[year] = flow_area_constructed_seg_dm + stock_area_constructed[year - 1]
        stock_area[year] = flow_area_remained_seg_dm + stock_area_constructed[year]
        average_area = stock_area[year].sum() / flow_needed[year]
        area_population_housing = stock_area[year].sum() / stock_population.loc[year]

        logging.debug('Technical progress: Learning by doing - New stock')
        stock_knowledge_new[year] = stock_knowledge_new[year - 1] + flow_area_constructed_seg_dm.sum()
        knowledge_normalize_new = stock_knowledge_new[year] / stock_knowledge_new[calibration_year]
        learning_rate = technical_progress_dict['learning-by-doing-new']
        cost_new = learning_by_doing_func(knowledge_normalize_new, learning_rate, year, cost_new, cost_new_lim_seg,
                                          calibration_year)
        logging.debug('Information acceleration - New stock')
        information_rate_new = information_rate_func(knowledge_normalize_new, kind='new')

        # TODO calculate knowledge_normalize_remaining based on renovation previous year
        logging.debug('Technical progress: Learning by doing - remaining stock')
        stock_knowledge_remained[year] = stock_knowledge_remained[year - 1] + flow_area_constructed_seg_dm.sum()
        logging.debug('Information acceleration - remaining stock')
        information_rate = information_rate_func(knowledge_normalize_new, kind='remaining')

        logging.debug('Demolition dynamic')
        stock_mobile_seg[year] = stock_remained_seg[year - 1] - stock_residual_seg
        logging.debug('Catching less efficient label for each segment')
        stock_destroyed_seg = stock_mobile2stock_destroyed(stock_mobile_seg[year], stock_mobile_seg[calibration_year],
                                                           stock_remained_seg[year - 1],
                                                           flow_destroyed_seg_dm[year], logging)
        stock_remained_seg[year] = stock_mobile_seg[year] - stock_destroyed_seg

        logging.debug('Renovation dynamic')
        result_dict = segments2renovation_rate(segments, year, energy_prices_df, cost_invest_df, cost_intangible_seg,
                                               rho_seg)
        renovation_rate_seg = result_dict['Renovation rate']
        flow_renovation_seg = renovation_rate_seg * stock_remained_seg[year]
        nb_renovation = flow_renovation_seg.sum()
        logging.debug('Renovation number: {:,.0f} buildings'.format(nb_renovation))
        logging.debug('Renovation rate: {:.1f}%'.format((nb_renovation / stock_remained_seg[calibration_year].sum()) * 100))
        market_share_seg = result_dict['Market share']
        flow_renovation_label_seg = ds_mul_df(flow_renovation_seg, market_share_seg)
        energy_lcc_ds = result_dict['Energy LCC']
        flow_renovation_label_energy_seg = renovation_label2renovation_label_energy(energy_lcc_ds,
                                                                                    cost_switch_fuel_df,
                                                                                    flow_renovation_label_seg)
        np.testing.assert_almost_equal(nb_renovation, flow_renovation_label_energy_seg.sum().sum(), err_msg='Not normal')

        area_renovation_seg = buildings_number2area(flow_renovation_label_energy_seg)
        area_renovation_by_label_seg = area_renovation_seg.groupby('Energy performance', axis=1).sum().sum()
        # flow_renovation_by_label = flow_renovation_label_energy_seg.groupby('Energy performance', axis=1).sum().sum()

        flow_remained_seg = flow_renovation2flow_remained(flow_renovation_label_energy_seg)
        stock_remained_seg[year] = stock_remained_seg[year - 1] + flow_remained_seg

        logging.debug('Construction dynamic')
        segments_new = segments2segments_new(segments)
        flow_constructed_new_seg = segments_new2flow_constructed(flow_constructed_seg_dm[year], segments_new,
                                                                 energy_prices_df, ds_income_owner_prop, year)
        flow_constructed_new_seg = flow_constructed_new_seg.droplevel(None, axis=0)
        flow_constructed_new_seg = flow_constructed_new_seg.reorder_levels(language_dict['levels_names'])
        stock_constructed_seg[year] = stock_constructed_seg[year - 1] + flow_constructed_new_seg

        logging.debug('Storing interesting data in output dict')
        output['Renovation rate seg'][year] = renovation_rate_seg

        logging.debug('End of YEAR: {}'.format(year))

    end = time.time()
    logging.debug('Time for the module: {} seconds.'.format(end - start))
    logging.debug('End')
