
import logging
import time
import pickle
import datetime

# from project.input import folder
from project.old_src.func import *
from project.function_pandas import *


# TODO: calibration, adding public policies, documentation, unittest
# TODO: sensitivity analysis


if __name__ == '__main__':

    start = time.time()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.debug('Start Res-IRF')

    folder['output'] = os.path.join(folder['output'], datetime.datetime.today().strftime('%Y%m%d_%H%M%S'))
    if not os.path.isdir(folder['output']):
        os.mkdir(folder['output'])

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
    cost_switch_fuel_df = cost_dict['cost_switch_fuel']

    # loading parc
    name_file = os.path.join(folder['intermediate'], 'parc.pkl')
    logging.debug('Loading parc pickle file {}'.format(name_file))
    stock_ini_seg = pd.read_pickle(name_file)
    name_file = os.path.join(folder['output'], 'parc.csv')
    stock_ini_seg.to_csv(name_file)
    logging.debug('Writing parc in .csv file: {}'.format(name_file))
    logging.debug('Total number of housing in this study {:,.0f}'.format(stock_ini_seg.sum()))
    segments = stock_ini_seg.index
    logging.debug('Total number of segments in this study {:,}'.format(len(segments)))

    # calculating owner income distribution in the parc
    levels = [lvl for lvl in stock_ini_seg.index.names if lvl not in ['Income class owner', 'Energy performance']]
    temp = val2share(stock_ini_seg, levels, option='column')
    io_share_seg = temp.groupby('Income class owner', axis=1).sum()
    # TODO: val2share creates share of levels based on other levels io_share_ht? Can it do io_share_tot?

    # distribution housing by decision-maker
    # TODO: clarifying val2share & proportion --> Jupyter tutorials
    stock_ini_dm = stock_ini_seg.groupby(['Occupancy status', 'Housing type']).sum()
    dm_share_tot = stock_ini_dm / stock_ini_seg.sum()

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

    technical_progress_dict = parameters_dict['technical_progress_dict']

    # initializing investment cost new
    # TODO understand why do we have to do that
    cost_new_seg = pd.concat([cost_dict['cost_new']] * 2, keys=['Homeowners', 'Landlords'], names=['Occupancy status'])
    cost_new_seg.sort_index(inplace=True)
    cost_new_lim_seg = pd.concat([cost_dict['cost_new_lim']] * len(language_dict['energy_performance_new_list']),
                                 keys=language_dict['energy_performance_new_list'], names=['Energy performance'])
    cost_new_lim_seg = pd.concat([cost_new_lim_seg] * len(language_dict['heating_energy_list']),
                                 keys=language_dict['heating_energy_list'], names=['Heating energy'])
    cost_new_lim_seg = cost_new_lim_seg.reorder_levels(cost_new_seg.index.names)
    cost_new_lim_seg.sort_index(inplace=True)

    logging.debug('Calculate life cycle cost for constructions')
    # remove income class levels as it is not useful to calibrate intangible cost
    segments_new = segments2segments_new(segments)
    lcc_new_dm = segments_new2lcc(segments_new, calibration_year)
    lcc_new_dm = lcc_new_dm.droplevel('Income class', axis=1)
    lcc_new_dm = lcc_new_dm.loc[:, ~lcc_new_dm.columns.duplicated()]
    intangible_cost_construction = calibration_market_share_construction(lcc_new_dm, logging)

    logging.debug('Calculate life cycle cost for each possible transition wo intangible cost')
    energy_lcc_ds = segments2energy_lcc(segments, calibration_year)
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
    elif parameters_dict['intangible_cost_source'] == 'function':
        logging.debug('Calibration of market share with intangible cost'.format(name_file))
        cost_intangible_seg = calibration_market_share(lcc_df, logging)
    else:
        raise ValueError

    logging.debug('Calculate net present value for each segment')
    lcc_df = cost2lcc(energy_lcc_ds, cost_invest=cost_invest_df, cost_intangible=cost_intangible_seg,
                      transition='label')
    logging.debug('Calculate market share for each possible transition')
    market_share_label = lcc2market_share(lcc_df)
    pv_df = (market_share_label * lcc_df).sum(axis=1)
    pv_df = pv_df.replace({0: float('nan')})
    segments_initial = pv_df.index
    energy_initial_lcc_ds = segments2energy_lcc(segments_initial, calibration_year)
    npv_df = energy_initial_lcc_ds.iloc[:, 0] - pv_df

    logging.debug('Calibration of renovation rate with rho param')
    rho_seg = calibration_renovation_rate(npv_df, stock_ini_seg)
    rho_seg.to_pickle(os.path.join(folder['calibration_intermediate'], 'rho.pkl'))
    logging.debug('End of calibration and dumping rho.pkl')

    logging.debug('Calculate renovation rate for each segment')
    renovation_rate_seg = npv_df.reset_index().apply(renov_rate_func, rho=rho_seg, axis=1)
    renovation_rate_seg.index = npv_df.index

    flow_renovation_seg = renovation_rate_seg * stock_ini_seg
    logging.debug('Renovation number: {:,.0f}'.format(flow_renovation_seg.sum()))
    logging.debug('Renovation rate: {:,.1f}%'.format(flow_renovation_seg.sum() * 100 / stock_ini_seg.sum()))

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
    nb_population_housing[calibration_year] = nb_population_housing_ini

    flow_needed = dict()
    # TODO absurd
    flow_needed[calibration_year] = stock_ini_seg.sum()

    stock_constructed_seg, stock_constructed, flow_constructed, flow_constructed_dm = dict(), dict(), dict(), dict()
    segments_new = segments2segments_new(segments)
    stock_constructed[calibration_year] = 0
    flow_constructed[calibration_year] = 0

    flow_destroyed, flow_destroyed_seg_dm = dict(), dict()

    stock_remained_seg, stock_remained, flow_remained_seg_dm = dict(), dict(), dict()
    stock_remained_seg[calibration_year] = stock_ini_seg
    stock_remained[calibration_year] = stock_ini_seg.sum()

    stock_residual_seg = stock_ini_seg * parameters_dict['destruction_rate']

    stock_mobile_seg = dict()
    stock_mobile_seg[calibration_year] = stock_remained_seg[calibration_year] - stock_residual_seg

    stock_area_dm, stock_area_constructed_dm, dict_area_avg_new_dm = dict(), dict(), dict()
    stock_area_dm[calibration_year] = parameters_dict['area'] * stock_ini_dm
    stock_area_constructed_dm[calibration_year] = 0
    dict_area_avg_new_dm[calibration_year] = parameters_dict['area_new']

    share_multi_family = dict()
    share_multi_family[calibration_year] = parameters_dict['ht_share_tot'].loc['Multi-family']

    # knowledge is learnt at the end of the year i.e. for first dynamic year knowledge initialization
    stock_knowledge_construction_he_ep, stock_knowledge_renovation = dict(), dict()
    # stock_knowledge_construction_he_ep[calibration_year] = knowledge_construction
    flow_area_renovation_seg, flow_area_constructed_he_ep = dict(), dict()
    flow_area_renovation_seg[calibration_year] = stock_area_seg

    flow_area_constructed_ep = pd.Series(
        [2.5 * stock_area_new_existing_seg.sum(), 2 * stock_area_new_existing_seg.sum()], index=['BBC', 'BEPOS'])
    flow_area_constructed_ep.index.names = ['Energy performance']
    temp = add_level(flow_area_constructed_ep,
                     pd.Index(language_dict['heating_energy_list'], name='Heating energy'))
    flow_area_constructed_he_ep[calibration_year] = temp

    cost_new = dict()
    cost_new[calibration_year] = cost_new_seg

    output['Intangible cost construction'], output['Investment cost envelope'] = dict(), dict()
    output['Intangible cost renovation'] = dict()
    output['Intangible cost construction'][calibration_year] = intangible_cost_construction
    output['Intangible cost renovation'][calibration_year] = cost_intangible_seg
    output['Investment cost envelope'][calibration_year] = cost_invest_df

    logging.debug('Start  dynamic evolution of buildings stock from year {} to year {}'.format(index_year[0], index_year[-1]))
    for year in index_year[1:3]:
        logging.debug('YEAR: {}'.format(year))

        logging.debug('Number of persons by housing')
        nb_population_housing[year] = population_housing_dynamic(nb_population_housing[year - 1],
                                                                 nb_population_housing[calibration_year])

        logging.debug('Dynamic of non-segmented buildings stock')
        flow_needed[year] = stock_population.loc[year] / nb_population_housing[year]
        flow_destroyed[year] = stock_remained[year - 1] * parameters_dict['destruction_rate']
        stock_remained[year] = stock_remained[year - 1].sum() - flow_destroyed[year]
        flow_constructed[year] = flow_needed[year] - stock_remained[year]
        stock_constructed[year] = stock_constructed[year - 1] + flow_constructed[year]

        logging.debug('Distribution of buildings by decision-maker')
        flow_destroyed_seg_dm[year] = dm_share_tot * flow_destroyed[year]
        flow_remained_seg_dm[year] = dm_share_tot * stock_remained[year]
        flow_constructed_dm[year], share_multi_family = housing_need2housing_construction(flow_needed,
                                                                                          flow_constructed,
                                                                                          share_multi_family,
                                                                                          year,
                                                                                          calibration_year)

        logging.debug('Dynamic area')
        flow_area_remained_seg_dm = flow_remained_seg_dm[year] * parameters_dict['area']
        flow_area_destroyed_seg_dm = flow_destroyed_seg_dm[year] * parameters_dict['area']
        dict_area_avg_new_dm = area_new_dynamic(dict_area_avg_new_dm, year, calibration_year)
        flow_area_constructed_dm = flow_constructed_dm[year] * dict_area_avg_new_dm[year]
        stock_area_constructed_dm[year] = flow_area_constructed_dm + stock_area_constructed_dm[year - 1]
        stock_area_dm[year] = flow_area_remained_seg_dm + stock_area_constructed_dm[year]

        logging.debug('Knowledge dynamic construction')
        #
        if stock_knowledge_construction_he_ep != {}:
            stock_knowledge_construction_he_ep[year] = stock_knowledge_construction_he_ep[year - 1] + \
                                                       flow_area_constructed_he_ep[year - 1]
        else:
            stock_knowledge_construction_he_ep[year] = flow_area_constructed_he_ep[year - 1]
        knowledge_norm_new = stock_knowledge_construction_he_ep[year] / stock_knowledge_construction_he_ep[
            calibration_year + 1]

        logging.debug('Learning by doing - construction')
        learning_rate = technical_progress_dict['learning-by-doing-new']
        cost_new = learning_by_doing_func(knowledge_norm_new, learning_rate, year, cost_new, cost_new_lim_seg,
                                          calibration_year)
        logging.debug('Information acceleration - construction')
        information_rate_new = information_rate_func(knowledge_norm_new, kind='new')
        intangible_cost_construction = ds_mul_df(information_rate_new, intangible_cost_construction.T).T

        logging.debug('Knowledge dynamic renovation')
        knowledge_norm_renovation, stock_knowledge_renovation = area2knowledge_renovation(flow_area_renovation_seg[year - 1],
                                                                                          stock_knowledge_renovation,
                                                                                          year,
                                                                                          calibration_year)

        logging.debug('Learning by doing - renovation')
        learning_rate = technical_progress_dict['learning-by-doing-renovation']
        learning_by_doing_construction = knowledge_norm_renovation ** (np.log(1 + learning_rate) / np.log(2))
        cost_invest_df = cost_invest_df * learning_by_doing_construction

        logging.debug('Information acceleration - renovation')
        information_rate = information_rate_func(knowledge_norm_renovation, kind='remaining')
        cost_intangible_seg = ds_mul_df(information_rate, cost_intangible_seg.T).T

        logging.debug('Demolition dynamic')
        stock_mobile_seg[year] = stock_remained_seg[year - 1] - stock_residual_seg
        logging.debug('Catching less efficient label for each segment')
        flow_destroyed_seg = stock_mobile2flow_destroyed(stock_mobile_seg[year], stock_mobile_seg[calibration_year],
                                                         stock_remained_seg[year - 1],
                                                         flow_destroyed_seg_dm[year], logging)
        np.testing.assert_almost_equal(flow_destroyed_seg_dm[year].sum(), flow_destroyed_seg.sum(),
                                       err_msg='Not normal')
        # stock remained segmented before renovation
        stock_remained_seg[year] = stock_remained_seg[year - 1] - flow_destroyed_seg
        logging.debug('Stock remained buildings: {:,.0f}'.format(stock_remained_seg[year].sum()))

        logging.debug('Renovation dynamic')
        result_dict = segments2renovation_rate(segments, year, cost_invest_df, cost_intangible_seg, rho_seg)
        renovation_rate_seg = result_dict['Renovation rate']
        flow_renovation_seg = renovation_rate_seg * stock_remained_seg[year]
        nb_renovation = flow_renovation_seg.sum()
        logging.debug('Renovation number: {:,.0f} buildings'.format(nb_renovation))
        logging.debug('Renovation rate: {:.1f}%'.format((nb_renovation / stock_remained_seg[year].sum()) * 100))
        market_share_seg = result_dict['Market share']
        flow_renovation_label_seg = ds_mul_df(flow_renovation_seg, market_share_seg)
        energy_lcc_ds = result_dict['Energy LCC']
        flow_renovation_label_energy_seg = renovation_label2renovation_label_energy(energy_lcc_ds,
                                                                                    cost_switch_fuel_df,
                                                                                    flow_renovation_label_seg)
        np.testing.assert_almost_equal(nb_renovation, flow_renovation_label_energy_seg.sum().sum(), err_msg='Not normal')

        flow_remained_seg = flow_renovation2flow_remained(flow_renovation_label_energy_seg)
        # stock remained segmented after renovation
        stock_remained_seg[year] = stock_remained_seg[year] + flow_remained_seg
        logging.debug('Stock remained buildings: {:,.0f}'.format(stock_remained_seg[year].sum()))

        # flow area renovation seg used to determine flow knowledge
        flow_area_renovation_seg[year] = buildings_number2area(flow_renovation_label_energy_seg)

        logging.debug('Construction dynamic')
        segments_new = segments2segments_new(segments)
        flow_constructed_seg = segments_new2flow_constructed(flow_constructed_dm[year], segments_new, io_share_seg,
                                                             year, cost_intangible=intangible_cost_construction)
        flow_constructed_seg = flow_constructed_seg.reorder_levels(language_dict['levels_names'])
        logging.debug('Constructed buildings: {}'.format(flow_constructed_seg))
        if stock_constructed_seg == {}:
            stock_constructed_seg[year] = flow_constructed_seg
        else:
            stock_constructed_seg[year] = stock_constructed_seg[year - 1] + flow_constructed_seg

        # variable used to determine construction knowledge in the learning-by-doing process.
        # TODO: area? doesn't use dynamic at all
        flow_area_constructed_he_ep[year] = flow_constructed_seg.groupby(['Energy performance', 'Heating energy']).sum()

        logging.debug('Output: storing important results')
        output['Renovation rate seg'][year] = renovation_rate_seg
        output['Intangible cost construction'][year] = intangible_cost_construction
        output['Intangible cost renovation'][year] = cost_intangible_seg
        output['Investment cost envelope'][year] = cost_invest_df

        logging.debug('End of YEAR: {}'.format(year))

    for key in output.keys():
        output2csv(output, key, logging)

    # write csv file from dict pd Series
    to_output = {'Investment_cost_construction.csv': cost_new,
                 'Stock_remained_segmented.csv': stock_remained_seg,
                 'Stock_constructed_segmented.csv': stock_constructed_seg}
    for name_file, item in to_output.items():
        name_file = os.path.join(folder['output'], name_file)
        pd.concat(item, axis=1).to_csv(name_file)
        logging.debug('Output: {}'.format(name_file))

    # write csv file from dict float - concatenate all data in output_detailed.csv
    to_output = {'Population': stock_population,
                 'Population by buildings': nb_population_housing,
                 'Buildings needed': flow_needed,
                 'Buildings destroyed': flow_destroyed,
                 'Stock buildings remained': stock_remained,
                 'Buildings constructed': flow_constructed,
                 'Stock constructed': stock_constructed}

    output_detailed = pd.concat([pd.Series(item, name=val) for val, item in to_output.items()], axis=1)

    stock_remained_seg_ts = pd.concat(stock_remained_seg, axis=1)
    result = segments2energy_consumption(stock_remained_seg_ts.index, exogenous_dict['energy_price_forecast'],
                                         kind='remaining')
    lst = []
    for key in result.keys():
        name_file = os.path.join(folder['output'], key.replace(' ', '_') + '.csv')
        logging.debug('Output: {}'.format(name_file))
        if key == 'Consumption-actual':
            temp = result[key] * stock_remained_seg_ts
            temp.to_csv(name_file)
            lst += [pd.Series(temp.sum(axis=0) / stock_remained_seg_ts.sum(), name='Average {}'.format(key))]
            lst += [pd.Series(temp.sum(axis=0), name='Total {}'.format(key))]
        elif key == 'Consumption-conventional':
            temp = ds_mul_df(result[key], stock_remained_seg_ts)
            temp.to_csv(name_file)
            lst += [pd.Series(temp.sum(axis=0) / stock_remained_seg_ts.sum(), name='Average {}'.format(key))]
            lst += [pd.Series(temp.sum(axis=0), name='Total {}'.format(key))]
        elif key == 'Budget share':
            result[key].to_csv(name_file)
            lst += [pd.Series((result[key] * stock_remained_seg_ts).sum(axis=0) / stock_remained_seg_ts.sum(), name=key)]
        elif key == 'Use intensity':
            result[key].to_csv(name_file)
            lst += [pd.Series((result[key] * stock_remained_seg_ts).sum(axis=0) / stock_remained_seg_ts.sum(), name=key)]
    name_file = os.path.join(folder['output'], 'output_detailed.csv')
    output_detailed = pd.concat((output_detailed, pd.concat(lst, axis=1)), axis=1)
    output_detailed.to_csv(name_file)
    logging.debug('Output: {}'.format(name_file))


    end = time.time()
    logging.debug('Time for the module: {:,.0f} seconds.'.format(end - start))
    logging.debug('End')
