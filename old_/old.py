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