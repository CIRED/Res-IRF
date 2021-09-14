import pandas as pd
import numpy as np


def logistic(x, a=1, r=1, k=1):
    return k / (1 + a * np.exp(- r * x))


def apply_linear_rate(value, rate, index):
    """Apply a linear rate for a value based on years index.

    Parameters
    ----------
    value: float or int
    rate: float
    index: list of int

    Returns
    -------
    pd.Series
    """
    temp = [value * (1 + rate) ** (i - index[0]) for i in index]
    return pd.Series(temp, index=index)


def multi_index2tuple(ds, list_levels):
    """Aggregate levels of a MultiIndex into a tuple based on specified levels.

    Parameters
    ----------
    ds: pd.Series
    list_levels: list

    Returns
    -------
    pd.Series
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
    """Remove rows based on value a level of pandas MultiIndex.

    Parameters
    ----------
    ds: pd.Series or pd.DataFrame
    level: str
        name of MultiIndex levels
    value_to_drop:

    Returns
    -------
    pd.Series

    """
    bad_index = ds.index.get_level_values(level) == value_to_drop
    return ds[~bad_index]


def replace(data, replace_dict):
    """Replace string in pandas Series (multiindexes included) with regex.

    Parameters
    ----------
    data: pd.Series or pd.DataFrame
    replace_dict: dict
        example: replace_dict = {'old_val0': 'new_val0', 'old_val0': 'new_val0'}

    Returns
    -------
    pd.Series
    """
    index_names = data.index.names
    df = data.reset_index()
    for key, val in replace_dict.items():
        df = df.replace(to_replace=key, value=val, regex=True)
    df.set_index(index_names, inplace=True)

    if isinstance(data, pd.Series):
        return df.iloc[:, 0]
    elif isinstance(data, pd.DataFrame):
        return df


def get_levels_values(miindex, levels):
    """Returns MultiIndex for levels passed based on miiindex.

    Parameters
    ----------
    miindex: pd.MultiIndex
    levels: list

    Returns
    -------
    pd.MultiIndex
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


def reindex_mi(df, miindex, levels=None, axis=0):
    """Return re-indexed DataFrame based on miindex using only few labels.

    Parameters
    -----------
    df: pd.DataFrame or pd.Series
        data to reindex
    miindex: pd.MultiIndex
        master to index to reindex df
    levels: list, default df.index.names
        list of levels to use to reindex df
    axis: {0, 1}, default 0
        axis to reindex df

    Returns
    --------
    pd.DataFrame or pd.Series

    Example:
    --------
        reindex_mi(surface_ds, segments, ['Occupancy status', 'Housing type']))
        reindex_mi(cost_invest_ds, segments, ['Heating energy final', 'Heating energy']))
    """

    if levels is None:
        if axis == 0:
            levels = df.index.names
        else:
            levels = df.columns.names

    if len(levels) > 1:
        tuple_index = (miindex.get_level_values(level).tolist() for level in levels)
        new_miindex = pd.MultiIndex.from_tuples(list(zip(*tuple_index)))
        if axis == 0:
            df = df.reorder_levels(levels)
        else:
            df = df.reorder_levels(levels, axis=1)
    else:
        new_miindex = miindex.get_level_values(levels[0])
    df_reindex = df.reindex(new_miindex, axis=axis)
    if axis == 0:
        df_reindex.index = miindex
    elif axis == 1:
        df_reindex.columns = miindex
    else:
        raise AttributeError('Axis can only be 0 or 1')

    return df_reindex


def add_level_nan(ds, level):
    """Add level to ds with 'nan' as value index.

    Value of pd.Series doesn't change.

    Parameters
    -----------
    ds: pd.Series
    level: str

    Returns
    --------
    pd.Series
    """
    ds_index_names = ds.index.names
    ds = pd.concat([ds], keys=['nan'], names=[level])
    ds = ds.reorder_levels(ds_index_names + [level])
    return ds


def add_level(data, index, axis=0):
    """Add index as a new level for ds.index or ds.columns.

    Values of data does not depend on the new level (i.e. only defined by other levels).

    Parameters
    -----------
    data: pd.Series
    index: pd.Index or list-like
    axis: {0, 1}, default 0

    Returns
    --------
    pd.Series
    """
    # ds_added append identical Series for each unique value
    if isinstance(data, pd.Series):
        ds_added = pd.Series(dtype='float64')
        for new_index in index:
            ds_added = ds_added.append(pd.concat([data], keys=[new_index], names=[index.names[0]], axis=axis))

        if axis == 0:
            ds_added.index = pd.MultiIndex.from_tuples(ds_added.index)
            ds_added.index.names = [index.names[0]] + data.index.names
        else:
            raise

        return ds_added

    elif isinstance(data, pd.DataFrame):
        df_temp = pd.DataFrame(dtype='float64')
        if axis == 1:
            for column in index:
                df_temp = pd.concat((df_temp, pd.concat([data], keys=[column], names=[index.names[0]], axis=1)), axis=1)
        elif axis == 0:
            for new_index in index:
                df_temp = pd.concat((df_temp, pd.concat([data], keys=[new_index], names=[index.names[0]])), axis=0)
        return df_temp


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
    """TOKEEP:Add new levels to ds_val using df_share.

    Parameters
    ----------
    ds_val:
        index: segments
    df_share: pd.DataFrame
        index: segments, columns: new level,
        ds_share and ds_val segment must share at least on level.

    Returns
    --------
    pd.Series
        MultiIndex series with
    """
    levels_shared = [n for n in df_share.index.names if n in ds_val.index.names]
    # reindex_mi add missing levels to df_share
    df_share_r = reindex_mi(df_share, ds_val.index, levels_shared)
    return(ds_val * df_share_r.T).T.stack()


def val2share(ds, levels, func=lambda x: x, option='row'):
    """Returns the share of value based on levels.

    Get the proportion for every other levels than levels in ds.index.names.
    If option = 'column', return pd DataFrame with levels in index. Sum of each row is equal to 1.
    """
    # TODO: column option mandatory caused row is very confusing
    denum = reindex_mi(ds.apply(func).groupby(levels).sum(), ds.index)
    num = ds.apply(func)
    prop = num / denum
    if option == 'row':
        return prop
    elif option == 'column':
        values = prop.name
        columns = [lvl for lvl in ds.index.names if lvl not in levels]
        prop = prop.reset_index()
        prop = pd.pivot_table(prop, values=values, index=levels, columns=columns)
        if None in prop.columns.names:
            prop = prop.droplevel(list(prop.columns.names).index(None), axis=1)
        return prop
    else:
        raise ValueError
