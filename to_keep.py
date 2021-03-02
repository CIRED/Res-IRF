import pandas as pd




def energy2CO2(energy_ts_df, CO2_content, option='final'):
    if option == 'final':
        level = 'Heating energy final'
    elif option == 'initial':
        level = 'Heating energy'
    else:
        raise

    idx_energy = energy_ts_df.index.get_level_values(level)
    CO2_content_reindex = CO2_content.reindex(idx_energy)
    CO2_content_df = pd.Series(CO2_content_reindex.values)
    CO2_content_ts_df = pd.concat([CO2_content_df] * len(energy_ts_df.columns), axis=1)
    CO2_content_ts_df.index = energy_ts_df.index
    CO2_content_ts_df.columns = energy_ts_df.columns
    CO2_emission_saving_ts_df = energy_ts_df.multiply(CO2_content_ts_df, axis=0)

    return CO2_emission_saving_ts_df


def agregate(df, dsp):
    idx_occ = df.index.get_level_values('Occupancy status')
    idx_housing = df.index.get_level_values('Housing type')
    idx_performance = df.index.get_level_values('Energy performance')
    idx_energy = df.index.get_level_values('Heating energy')
    idx_inc = df.index.get_level_values('Income class')
    idx_df = pd.MultiIndex.from_tuples(list(zip(list(idx_occ), list(idx_housing), list(idx_performance), list(idx_energy), list(idx_inc))))
    dsp_reindex = dsp.reindex(idx_df)
    dsp_reindex = pd.Series(dsp_reindex.values, index=df.index)
    result = pd.concat((df, dsp_reindex), axis=1)
    if isinstance(df, pd.DataFrame):
        result.columns = df.columns.tolist() + ['Housing number']
    elif isinstance(df, pd.Series):
        result.columns = [df.name] + ['Housing number']

    return result


def switch_fuel(df, plot=True):
    """
    Return series with segments that switch fuel.
    """
    idx = df.index.get_level_values('Heating energy') != df.index.get_level_values('Heating energy final')
    df_switching_fuel = df[idx]
    if plot:
        make_stacked_barplot(df_switching_fuel, ['Heating energy', 'Heating energy final'], format_yaxis='comma')
    return df_switching_fuel