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


def remove_rows(ds, level, value_to_drop):
    """Remove rows based on value in index.
    """
    bad_index = ds.index.get_level_values(level) == value_to_drop
    ds = ds[~bad_index]
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


def linear2series(value, rate, index):
    temp = [value * (1 + rate) ** (i - index[0]) for i in index]
    return pd.Series(temp, index=index)


def get_levels_values(miindex, levels):
    """Returns MultiIndex for levels passed based on miiindex.

    """
    tuple_idx = tuple()
    for level in levels:
        idx = list(miindex.get_level_values(level))
        tuple_idx = tuple_idx + (idx,)
    idx_return = pd.MultiIndex.from_tuples(list(zip(*tuple_idx)))
    idx_return.names = levels
    return idx_return


def miiindex_loc(ds, slicing_midx):
    """Select multiindex slices based on slice_midx when all levels are not passed.

    slicing_midx must have at least on common level name with ds MultiIndex.
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
    """Return re-indexed DataFrame based on miindex using only few labels.

    Levels order must match df.levels order.
    Example:
        reindex_mi(surface_ds, segments, ['Occupancy status', 'Housing type']))
        reindex_mi(cost_invest_ds, segments, ['Heating energy final', 'Heating energy']))
    """

    if len(levels) > 1:
        tuple_index = (miindex.get_level_values(level).tolist() for level in levels)
        new_miindex = pd.MultiIndex.from_tuples(list(zip(*tuple_index)))
        df = df.reorder_levels(levels)
    else:
        new_miindex = miindex.get_level_values(levels[0])
    df_reindex = df.reindex(new_miindex)
    df_reindex.index = miindex
    return df_reindex


def ds_mul_df(ds, df, option='columns'):
    """Multiply pd.Series to each columns (or rows) of pd.Dataframe.

    """
    if option == 'columns':
        if isinstance(df.index, pd.MultiIndex):
            ds = ds.reorder_levels(df.index.names)
        ds.sort_index(inplace=True)
        df.sort_index(inplace=True)
        assert (ds.index == df.index).all(), "indexes don't match"
        ds = pd.concat([ds] * len(df.columns), axis=1)
        ds.columns = df.columns
        return ds * df
    elif option == 'rows':
        if isinstance(df.columns, pd.MultiIndex):
            ds = ds.reorder_levels(df.columns.names)
        ds.sort_index(inplace=True)
        df.sort_index(inplace=True, axis=1)
        assert (ds.index == df.columns).all(), "indexes don't match"
        ds = pd.concat([ds] * len(df.index), axis=1).T
        ds.index = df.index
        return ds * df


def add_level_nan(ds, level):
    """
    Add level with 'nan' as value index. Value of pd.Series don't change.
    """
    ds_index_names = ds.index.names
    ds = pd.concat([ds], keys=['nan'], names=[level])
    ds = ds.reorder_levels(ds_index_names + [level])
    return ds


def de_aggregating_series(ds_val, level_share_tot, level):
    """De-aggregate ds based on ds_prop by adding level.

    ds_prop can be independent of ds, or some levels can match.

    Example:

    """
    ds_index_names = ds_val.index.names
    levels_shared = [n for n in level_share_tot.index.names if n in ds_index_names]

    # unique values in level that need to be added to ds_val
    new_indexes = list(set(level_share_tot.index.get_level_values(level)))
    # ds_added append identical Series for each unique value
    ds_added = pd.Series(dtype='float64')
    for new_index in new_indexes:
        ds_added = ds_added.append(pd.concat([ds_val], keys=[new_index], names=[level]))

    ds_added.index = pd.MultiIndex.from_tuples(ds_added.index)
    ds_added.index.names = [level] + ds_index_names

    # ds_prop = ds_prop.reindex(d_temp.index)
    # TODO: no need to reorder as reindex_mi do it now
    if isinstance(level_share_tot.index, pd.MultiIndex):
        level_share_tot = level_share_tot.reorder_levels([level] + levels_shared)
    level_share_tot = reindex_mi(level_share_tot, ds_added.index, [level] + levels_shared)

    return ds_added.multiply(level_share_tot)


def de_aggregate_columns(df1, df2):
    """Cumulate 2 share df based on the same level to get a de-aggregate DataFrame.

    """

    level1 = df1.columns.names[0]
    df_temp = pd.DataFrame(dtype='float64')
    for column in df1.columns:
        df_temp = pd.concat((df_temp, pd.concat([df2], keys=[column], names=[level1], axis=1)), axis=1)
    df1_r = df1.reindex(df_temp.columns.get_level_values(level1), axis=1)
    df1_r.columns = df_temp.columns
    return df1_r * df_temp


def de_aggregate_series(ds_val, df_share):
    """Add new levels to ds_val using df_share.

    df_share has segment in index and share of new level in columns.
    ds_val has segment in index.
    ds_share and ds_val segment share some levels.

    Example:
        Function used to add Income class owner to a stock or a flow.
    """
    levels_shared = [n for n in df_share.index.names if n in ds_val.index.names]
    # reindex_mi add missing levels to df_share
    df_share_r = reindex_mi(df_share, ds_val.index, levels_shared)
    return ds_mul_df(ds_val, df_share_r).stack()


def serie_to_prop(serie, level):
    """Get proportion of one dimension.
    """
    # TODO: work on this function
    by = [i for i in serie.index.names if i != level]
    grouped = serie.groupby(by)
    ds = pd.Series(dtype='float64')
    for name, group in grouped:
        ds = pd.concat((ds, group / group.sum()), axis=0)
    ds.index = pd.MultiIndex.from_tuples(ds.index)
    ds.index.names = serie.index.names
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


def val2share(ds, levels, func=lambda x: x, option='row'):
    """Returns the share of value based on levels.

    Get the proportion for every other levels than levels in ds.index.names.
    If option = 'columns', return pd DataFrame with levels in index. Sum of each row is equal to 1.
    """
    # TODO: column option mandatory caused row is very confusing
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
    else:
        raise ValueError


def add_level(ds, index, axis=0):
    """Add index as a new level for ds index.

    Value of ds does not depend on the new level (i.e. only defined by other levels).
    """
    # ds_added append identical Series for each unique value
    ds_added = pd.Series(dtype='float64')
    for new_index in index:
        ds_added = ds_added.append(pd.concat([ds], keys=[new_index], names=[index.names[0]], axis=axis))

    if axis == 0:
        ds_added.index = pd.MultiIndex.from_tuples(ds_added.index)
        ds_added.index.names = [index.names[0]] + ds.index.names

    return ds_added

