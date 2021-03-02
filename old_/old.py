def add_multiindex(data, new_index_level, name):
    """
    Function that add all indexes of new_index_level in data, copying the value.
    """
    temp = pd.DataFrame()
    for new_index in new_index_level:
        a = pd.concat([data], keys=[new_index], names=[name])
        temp = pd.concat([temp, a], axis=0)

    idx = pd.MultiIndex.from_tuples(temp.index)
    temp.index = idx

    return temp


new_index_level = df.index.get_level_values(0)
name = 'Niveau rémunération'
data_temp = add_multiindex(data, new_index_level, name)

new_index_level = data_temp.index.get_level_values(2)
name = 'Type logement'
df_temp = add_multiindex(df, new_index_level, name)

# too long
new_index_level = data_temp.index.get_level_values(3)
name = 'Performance énergétique'
df_temp = add_multiindex(df_temp, new_index_level, name)


a = df.reindex(data_temp.index, method='pad')

names = ["Income class, Occupancy status", "Housing type", "Performance"]

s = pd.Series(dtype='float64')
grouped = data_income.groupby(['Income class'])
for name, group in grouped:
    gr = group.groupby('DECILE_PB')
    for n, g in gr:
        s = pd.concat((s, pd.DataFrame([[name, n, g.sum()]])), axis=0)

series_income_temp.rename(index={'Bois*': 'Autres', 'Fioul domestique': 'Autres'}, inplace=True)


"""data_income.sort_index(inplace=True)
data_income_prop = pd.DataFrame(dtype='float64')
for index in data_income.index.unique():
    total = data_income.loc[index, ['DECILE_PB', 'NB_LOG']].sum()
    s = pd.concat((data_income.loc[index, 'NB_LOG'] / total.loc['NB_LOG'], data_income.loc[index, 'DECILE_PB']), axis=1)
    data_income_prop = pd.concat((data_income_prop, s), axis=0)"""

def life_cycle_cost_func(segment, energy_price_df, all_segments, intangible_cost_data):
    """
    Calculate the life cycle cost of an investment for a specific initial segment, and its market share.
    segment: tuple that represents ("Occupancy status", "Housing type", "Energy performance", "Heating energy",
    "Income class", "Income class owner")
    """
    # TODO: surface_series, switch_fuel
    h = Housing(*segment)
    energy_performance_transition = [i for i in language_dict['energy_performance_list'] if i < h.energy_performance]

    life_cycle_cost_data = pd.DataFrame(index=all_segments, columns=language_dict['energy_performance_list'])
    market_share_data = pd.DataFrame(index=all_segments, columns=language_dict['energy_performance_list'])

    for i in energy_performance_transition:
        discounted_energy_price = h.discounted_energy_prices(energy_price_df)
        life_cycle_cost_data.loc[segment, i] = investment_cost_data.loc[h.energy_performance, i] + \
                                                   discounted_energy_price + \
                                                   intangible_cost_data.loc[h.energy_performance, i]

    total_cost = life_cycle_cost_data.loc[segment, :].apply(lambda x: x ** -1).sum()

    for i in energy_performance_transition:
        market_share_data.loc[segment, i] = life_cycle_cost_data.loc[segment, i] ** -1 / total_cost

    return life_cycle_cost_data, market_share_data


def market_share_func_bis(energy_price_df, all_segments, intangible_cost_data):
    life_cycle_cost_data = pd.DataFrame(index=all_segments, columns=language_dict['energy_performance_list'])
    market_share_data = pd.DataFrame(index=all_segments, columns=language_dict['energy_performance_list'])

    logging.debug('Calculation of life cycle cost and market share for each segment. Total number {:,}'.format(len(all_segments)))
    k = 0
    for segment in all_segments:
        if k % round(len(all_segments)/100) == 0:
            logging.debug('Loading {:.2f} %'.format(k/len(all_segments) * 100))
        lfc, ms = life_cycle_cost_func(segment, energy_price_df, all_segments, intangible_cost_data)
        life_cycle_cost_data.update(lfc)
        market_share_data.update(ms)
        k += 1
    return life_cycle_cost_data, market_share_data


def renovation_rate_func(data, all_segments, energy_price_df):
    logging.debug('Calculation of renovation rate for each segment')
    npv = pd.Series(index=all_segments, dtype='float64')
    renovation_rate = pd.Series(index=all_segments, dtype='float64')
    for segment in all_segments:
        discounted_energy_price = Housing(*segment).discounted_energy_prices(energy_price_df)
        npv.loc[segment] = discounted_energy_price - data[segment]

        renovation_rate.loc[segment] = logistic(npv.loc[segment] - parameters_dict['npv_min'],
                                                    a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                                                    r=parameters_dict['r'],
                                                    K=parameters_dict['rate_max'])
    return renovation_rate, npv

    """segments = dsp.droplevel('Income class owner').index[~dsp.droplevel('Income class owner').index.duplicated(keep='first')].tolist()
    segments = [i + ('income_owner', ) for i in segments]
    segments = segments[700:-700]
    segments = pd.MultiIndex.from_tuples(segments)"""

    """dsp_idx = dsp.index.names
    dsp = dsp.reset_index().replace(language_dict['dict_replace']).set_index(dsp_idx).iloc[:, 0]
    dsp.index = dsp.index.set_names('Income class owner', 'DECILE_PB')
    dsp = dsp.reorder_levels(language_dict['properties_names'])"""