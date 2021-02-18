import os
import pandas as pd
import re

from input import folder


name_file = 'comptages_DPE.csv'
data = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0], encoding='latin-1',
                   index_col=[0, 1, 2, 3, 4], squeeze=True)
data_original = data

print('Total number of housing in this study {}'.format(data.sum()))

index_name = ['Housing type', 'Occupancy status', 'Income class', 'Energy', 'DPE']
data.index.names = index_name


def replace_strings(df, replace_dict):

    def replace_string_data(df, to_replace, value):
        df_updated = df.replace(to_replace=to_replace, value=value, regex=True)
        return df_updated

    index_names = df.index.names
    df = df.reset_index()
    for key, val in replace_dict.items():
        df = replace_string_data(df, key, val)

    df.set_index(index_names, inplace=True)

    return df


print("1st: Remove 'G' from Occupancy status")

bad_index = data.index.get_level_values('Occupancy status') == 'G'
data = data[~bad_index]

print('2nd: Desagregate Others to Wood fuel and Oil fuel')

name_file = 'fuel_oil_wood.xlsx'
energy_list = ['Bois*', 'Fioul domestique']
data_temp = pd.read_excel(os.path.join(folder['input'], name_file), header=[0], index_col=[1, 0])
data_temp = data_temp[data_temp.index.get_level_values('Energie principale de chauffage').isin(energy_list)]

data_temp.index = data_temp.index.set_levels(data_temp.index.levels[1].str.replace('Appartement', 'AP').str.replace('Maison', 'MA'), level=1)
data_temp = data_temp.loc[:, 'Taux du parc en %']


def desagregate_value(data, data_temp, level_name, val, column=None):
    """
    Create a new DataFrame for each row and then drop rows where value is.
    """
    if isinstance(data, pd.Series):
        data_add = pd.Series(dtype='float64')
    elif isinstance(data, pd.DataFrame):
        data_add = pd.DataFrame(dtype='float64')

    if isinstance(data, pd.Series):
        for index, value in data[data.index.get_level_values(level_name) == val].iteritems():
            for energy in energy_list:
                # todo: find something more general
                # position_level = [i for i, x in enumerate(data.index.names) if x == level_name][0]
                new_index = []
                for i in index:
                    if i == val:
                        new_index += [energy]
                    else:
                        new_index += [i]

                new_index = pd.MultiIndex.from_tuples([tuple(new_index)])
                data_add = pd.concat((data_add, pd.Series(value * data_temp.loc[(energy, index[0])], index=new_index)))

    elif isinstance(data, pd.DataFrame):
        for index, series in data[data.index.get_level_values(level_name) == val].iterrows():
            for energy in energy_list:
                new_index = pd.MultiIndex.from_tuples([(index[0], index[1], index[2], energy, index[4])])
                series.loc[column] = series.loc[column] * data_temp.loc[(energy, index[0])]
                new_series = pd.DataFrame([series.values], index=new_index, columns=series.index)
                data_add = pd.concat((data_add, new_series), axis=0)

    bad_index = data.index.get_level_values(level_name) == val
    data = data[~bad_index]
    data = pd.concat((data, data_add), axis=0)

    return data


data = desagregate_value(data, data_temp, 'Energy', 'Autres')

rename_dict = {'Appartement': 'AP', 'Maison': 'MA', 'Bois*': 'Wood fuel', 'Electricité': 'Power'}

print('Use quintile instead of decile')

# add level name
data.rename(index={'D9': 'C5', 'D10': 'C5', 'D8': 'C4', 'D7': 'C4', 'D6': 'C3', 'D5': 'C3', 'D4': 'C2', 'D3': 'C2', 'D2': 'C1', 'D1': 'C1'}, inplace=True)

print('Occupant income to owner income matrix')

name_file = 'parclocatifprive_post48_revenusPB.csv'
data_income = pd.read_csv(os.path.join(folder['input'], name_file), sep=',', header=[0], index_col=[2, 0, 3, 5, 6])
data_income.index.names = index_name

print('Use quintile instead of decile')


print('Remove rows where DECILE_PB is NaN')
data_income.dropna(axis=0, subset=['DECILE_PB'], inplace=True)
data_income.reset_index(inplace=True)
bad_index = data_income.index[data_income['DECILE_PB'] == 'NC']
data_income.drop(bad_index, inplace=True)

data_income.set_index(['Housing type', 'Occupancy status', 'Income class', 'Energy', 'DPE', 'DECILE_PB'], inplace=True)

print("Desagregate Others to Wood fuel and Oil fuel using the proportionnal table")
data_income = data_income.loc[:, 'NB_LOG']
data_income = desagregate_value(data_income, data_temp, 'Energy', 'Autres', column='NB_LOG')

def serie_to_prop(serie, level):
    """
    Get proportion of one dimension.
    """
    by = [i for i in serie.index.names if i != level]
    grouped = serie.groupby(by)
    s = pd.Series(dtype='float64')
    for name, group in grouped:
        s = pd.concat((s, group / group.sum()), axis=0)
    return s

data_income_prop = serie_to_prop(data_income, 'DECILE_PB')

print('pause')

def serieND_to_serie2D(df, dimension_list, dimension_val, graph=False):
    """
    n-dimensions pandas series --> graph of one variable depending of one other variable
    grouping and summing all values
    aggregate by occupant income class
    """
    serie_2D = df.groupby(dimension_list).sum()
    serie_2D.columns = dimension_list + [dimension_val]

    # todo: correct that
    if graph:
        decile_list = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
        s_pivot = serie_2D.pivot(index='Income class', columns='DECILE_PB', values='NB_LOG')
        s_pivot.loc[decile_list, decile_list].plot.bar(stacked=True)

    return serie_2D


serieND_to_serie2D(data_income, ['Income class', 'DECILE_PB'], 'NB_LOG', graph=True)


data_income.rename(index={'D9': 'C5', 'D10': 'C5', 'D8': 'C4', 'D7': 'C4', 'D6': 'C3', 'D5': 'C3', 'D4': 'C2', 'D3': 'C2', 'D2': 'C1', 'D1': 'C1'}, inplace=True)


replace_dict = {'.?lectricit.*': 'Power', 'Gaz': 'Natural Gas', 'Bois*': 'Wood fuel', 'Fioul domestique': 'Oil fuel'}
replace_strings(data, replace_dict)

data_lp = data[data.index.get_level_values('Occupancy status') == 'LP']
for (index, val) in data_lp.iteritems():
    break
    data_add = pd.DataFrame(data_income_prop[index], index=index)
    break


print('pause')

