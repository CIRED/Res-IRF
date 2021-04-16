import pandas as pd


def multi_index2tuple(ds, list_levels):
    """
    Aggregate levels of a MultiIndex into a tuple.
    Example:
    df = pd.DataFrame({'col1': ['a1', 'b1'], 'col2': ['a2', 'b2']}, index=[('x1', 'x2', 'x3'), ('y1', 'y2', 'y3')])
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.index.names = ['1', '2', '3']
    dfp = DataFrameParc(df)
    dfp.multi_index2tuple(['1', '2'])
    """

    list_level_position = [ds.index.names.index(i) for i in list_levels]
    list_other_position = [i for i in range(len(ds.index.names)) if i not in list_level_position]

    list_other_name = [ds.index.names[i] for i in list_other_position]

    list_temp = []
    for i in ds.index.to_list():
        tuple_index = tuple([i[j] for j in list_level_position])
        if len(list_other_position) > 0:
            tuple_row_index = tuple([tuple_index] + [i[k] for k in list_other_position])
        else:
            tuple_row_index = tuple_index

        list_temp += [tuple_row_index]

    if len(list_other_position) > 0:
        ds.index = pd.MultiIndex.from_tuples(list_temp)
        ds.index.names = [tuple(list_levels)] + list_other_name
    else:
        ds.index = list_temp
        ds.index.names = [tuple(list_levels)]
    return ds


def de_aggregate_value(serie, ds_prop, val, level_val, list_val, level_key):
    """
    Replace val by list_val in dfp using ds_prop.
    """

    good_index = serie.index.get_level_values(level_val) == val
    ds_keep = serie[~good_index]
    ds_de_aggregate = serie[good_index]

    ds = pd.Series(dtype='float64')
    for v in list_val:
        ds = ds.append(replace_strings(ds_de_aggregate, {val: v}))

    # ds = ds.loc[ds.index.drop_duplicates(keep=False)]
    ds.index = pd.MultiIndex.from_tuples(ds.index)
    ds.index.names = serie.index.names

    ds = multi_index2tuple(ds, [level_val, level_key])
    idx_fuel = ds.index.get_level_values(0)
    ds_prop = multi_index2tuple(ds_prop, [level_val, level_key])
    ds_prop = ds_prop.reindex(idx_fuel)
    ds = pd.Series(ds.values * ds_prop.values, index=ds.index)

    df_index = pd.DataFrame(ds.index.get_level_values(0).tolist(), columns=[level_val, level_key])
    ds.index = ds.index.droplevel(level=0)
    ds = ds.reset_index()
    ds = pd.concat((df_index, ds), axis=1)
    ds.set_index(serie.index.names, drop=True, inplace=True)

    ds = ds.iloc[:, 0].append(ds_keep)
    return ds


def remove_rows(ds, level, value_to_drop):
    """
    Remove rows based on value in index.
    """
    bad_index = ds.index.get_level_values(level) == value_to_drop
    ds = ds[~bad_index]
    return ds


def serie_to_prop(serie, level):
    """
    Get proportion of one dimension.
    """
    by = [i for i in serie.index.names if i != level]
    grouped = serie.groupby(by)
    ds = pd.Series(dtype='float64')
    for name, group in grouped:
        ds = pd.concat((ds, group / group.sum()), axis=0)
    ds.index = pd.MultiIndex.from_tuples(ds.index)
    ds.index.names = serie.index.names
    return ds


def replace_strings(ds, replace_dict):
    """
    Example: replace_dict = {'old_val0': 'new_val0', 'old_val0': 'new_val0'}
    """
    def replace_string_data(data, to_replace, value):
        data_updated = data.replace(to_replace=to_replace, value=value, regex=True)
        return data_updated

    index_names = ds.index.names
    df = ds.reset_index()
    for key, val in replace_dict.items():
        df = replace_string_data(df, key, val)

    df.set_index(index_names, inplace=True)
    return df.iloc[:, 0]


def de_aggregating_series(ds, ds_prop, level):
    """
    De-aggregate ds based on ds_prop by adding level.
    ds_prop can be independent of ds, or some levels can match.
    """
    ds_index_names = ds.index.names
    new_indexes = list(set(ds_prop.index.get_level_values(level)))
    d_temp = pd.Series(dtype='float64')
    for new_index in new_indexes:
        d_temp = d_temp.append(pd.concat([ds], keys=[new_index], names=[level]))

    d_temp.index = pd.MultiIndex.from_tuples(d_temp.index)
    d_temp.index.names = [level] + ds_index_names
    # d_temp = d_temp.reorder_levels(ds_index_names + [level])

    # ds_prop = ds_prop.reindex(d_temp.index)
    levels_shared = [n for n in ds_prop.index.names if n in ds_index_names]
    ds_prop = ds_prop.reorder_levels([level] + levels_shared)
    ds_prop = reindex_mi(ds_prop, d_temp.index, [level] + levels_shared)

    return d_temp.multiply(ds_prop)


def add_level_nan(ds, level):
    """
    Add level with 'nan' as value
    """
    ds_index_names = ds.index.names
    ds = pd.concat([ds], keys=['nan'], names=[level])
    ds = ds.reorder_levels(ds_index_names + [level])
    return ds


def linear2series(value, rate, index):
    temp = [value * (1 + rate) ** (i - index[0]) for i in index]
    return pd.Series(temp, index=index)


def get_levels_values(multiindex, levels):
    tuple_idx = tuple()
    for level in levels:
        idx = list(multiindex.get_level_values(level))
        tuple_idx = tuple_idx + (idx,)
    idx_return = pd.MultiIndex.from_tuples(list(zip(*tuple_idx)))
    idx_return.names = levels
    return idx_return


def miiindex_loc(ds, slicing_midx):
    """
    Select multiindex slices based on slice_midx when all levels are not passed.
    Example:
        miiindex_loc(ds, slicing_midx)
    """
    # The levels to slice on, in sorted order
    slicing_levels = list(slicing_midx.names)
    # The levels not to slice on
    non_slicing_levels = [level for level in ds.index.names if level not in slicing_levels]

    # Reset the unneeded index
    res = ds.reset_index(non_slicing_levels).loc[slicing_midx].set_index(non_slicing_levels, append=True)
    return res


def reindex_mi(df, miindex, levels):
    """
    Return reindexed dataframe based on miindex using only few labels.
    Levels order must match df.levels order.
    Example:
        reindex_mi(surface_ds, segments, ['Occupancy status', 'Housing type']))
        reindex_mi(cost_invest_ds, segments, ['Heating energy final', 'Heating energy']))
    """
    if len(levels) > 1:
        tuple_index = (miindex.get_level_values(level).tolist() for level in levels)
        new_miindex = pd.MultiIndex.from_tuples(list(zip(*tuple_index)))
    else:
        new_miindex = miindex.get_level_values(levels[0])
    df_reindex = df.reindex(new_miindex)
    df_reindex.index = miindex
    return df_reindex


def ds_mul_df(ds, df):
    """
    Multiply pd Series to each columns of pd Dataframe.
    """
    if isinstance(df.index, pd.MultiIndex):
        ds = ds.reorder_levels(df.index.names)
    ds.sort_index(inplace=True)
    df.sort_index(inplace=True)
    assert (ds.index == df.index).all(), "indexes don't match"
    ds = pd.concat([ds] * len(df.columns), axis=1)
    ds.columns = df.columns
    return ds * df


def val2share(ds, levels, func=lambda x: x, option='row'):
    """
    Returns the share of value based on levels.
    Get the proportion for every other levels than levels in ds.index.names.
    If option = 'columns', return pd DataFrame with levels in index. Sum of each row is equal to 1.
    """
    denum = reindex_mi(ds.apply(func).groupby(levels).sum(), ds.index, levels)
    num = ds.apply(func)
    prop = num/denum
    if option == 'row':
        return prop
    elif option == 'column':
        values = prop.name
        columns = [l for l in ds.index.names if l not in levels]
        prop = prop.reset_index()
        prop = pd.pivot_table(prop, values=values, index=levels,  columns=columns)
        # prop.droplevel(prop.columns.names[0], axis=1)
        return prop


