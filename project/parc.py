import os
import logging
import time

from project.old_src.input import folder, language_dict
from project.function_pandas import *


start = time.time()

# todo: filename='res_irf.log'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

name_file = 'comptages_DPE.csv'
logging.info('Loading {}'.format(os.path.join(folder['input'], 'ds_parc', name_file)))
df_parc = pd.read_csv(os.path.join(folder['input'], 'ds_parc', name_file), sep=',', header=[0], encoding='latin-1',
                   index_col=[0, 1, 2, 3, 4], squeeze=True)

logging.debug('Total number of housing in this study {:,}'.format(df_parc.sum()))

index_names = ['Housing type', 'Occupancy status', 'Income class', 'Heating energy', 'Energy performance']
df_parc.index.names = index_names

logging.debug("Remove 'G' from Occupancy status")
df_parc = remove_rows(df_parc, 'Occupancy status', 'G')
logging.debug('Total number of housing at this point {:,}'.format(df_parc.sum()))

logging.debug("Desagregate 'Autres' to 'Wood fuel' and 'Oil fuel'")
name_file = 'fuel_oil_wood_2018.xlsx'
logging.info('Loading {}'.format(os.path.join(folder['input'], 'ds_parc', name_file)))
df_fuel = pd.read_excel(os.path.join(folder['input'], 'ds_parc', name_file), header=[0], index_col=[1, 0])
df_fuel.index.names = ['Heating energy', 'Housing type']
fuel_list = ['Bois*', 'Fioul domestique']
df_fuel = df_fuel[df_fuel.index.get_level_values('Heating energy').isin(fuel_list)]
df_fuel.index = df_fuel.index.set_levels(df_fuel.index.levels[1].str.replace('Appartement', 'AP').str.replace('Maison', 'MA'), level=1)
df_fuel = df_fuel.loc[:, 'Taux du parc en %']

df_parc = de_aggregate_value(df_parc, df_fuel.copy(), 'Autres', 'Heating energy', fuel_list, 'Housing type')
logging.debug('Total number of housing at this point {:,}'.format(df_parc.sum()))

logging.debug('Occupant income to owner income matrix')
name_file = 'parclocatifprive_post48_revenusPB.csv'
logging.info('Loading {}'.format(os.path.join(folder['input'], 'ds_parc', name_file)))
ds_income = pd.read_csv(os.path.join(folder['input'], 'ds_parc', name_file), sep=',', header=[0], index_col=[2, 0, 3, 5, 6])
ds_income.index.names = index_names

ds_income.reset_index(inplace=True)
bad_index = ds_income.index[ds_income['DECILE_PB'] == 'NC']
ds_income.drop(bad_index, inplace=True)
ds_income.set_index(index_names + ['DECILE_PB'], inplace=True)

logging.debug('Desagregate Others to Wood fuel and Oil fuel using the proportionnal table')
ds_income = ds_income.loc[:, 'NB_LOG']
ds_income = de_aggregate_value(ds_income, df_fuel.copy(), 'Autres', 'Heating energy', fuel_list, 'Housing type')
ds_income_prop = serie_to_prop(ds_income, 'DECILE_PB')
replace_dict = {'.?lectricit.*': 'Power', 'Gaz': 'Natural gas', 'Bois\*': 'Wood fuel', 'Fioul domestique': 'Oil fuel',
                'MA': 'Individual house', 'AP': 'Collective housing'}
ds_income_prop = replace_strings(ds_income_prop, replace_dict)

df_parc = replace_strings(df_parc, replace_dict)
logging.debug('Total number of housing at this point {:,}'.format(df_parc.sum()))

ds_lp = df_parc[df_parc.index.get_level_values('Occupancy status') == 'LP']
logging.debug('Number of landlords buildings {:,.0f}'.format(ds_lp.sum()))

ds_lp = de_aggregating_series(ds_lp, ds_income_prop, 'DECILE_PB')
logging.debug('Number of landlords buildings {:,.0f}'.format(ds_lp.sum()))

d_temp = remove_rows(df_parc, 'Occupancy status', 'LP')
d_temp = add_level_nan(d_temp, 'DECILE_PB')
df_parc = pd.concat((ds_lp, d_temp), axis=0)
logging.debug('Total number of housing at this point {:,}'.format(df_parc.sum()))

df_index = df_parc.index.names
df_parc = df_parc.reset_index().replace(language_dict['dict_replace']).set_index(df_index, drop=True).iloc[:, 0]
df_parc.index = df_parc.index.set_names('Income class owner', 'DECILE_PB')
df_parc = df_parc.reorder_levels(language_dict['levels_names'])

df_parc = df_parc.reset_index()
# setting income class owner = income class occupant when occupancy status = 'Homeowners'
df_parc.loc[df_parc.loc[:, 'Occupancy status'] == 'Homeowners', 'Income class owner'] = df_parc.loc[df_parc.loc[:, 'Occupancy status'] == 'Homeowners', 'Income class']

# setting income class owner = D10 when occupancy status = 'Social-housing'
temp = df_parc.loc[df_parc.loc[:, 'Occupancy status'] == 'Social-housing', 'Income class owner']
df_parc.loc[df_parc.loc[:, 'Occupancy status'] == 'Social-housing', 'Income class owner'] = ['D10'] * len(temp)

df_parc = df_parc.set_index(language_dict['levels_names']).iloc[:, 0]
df_parc.name = 'Housing numbers'


logging.debug('Saving df_parc as pickle in '.format(os.path.join(folder['intermediate'], 'parc.pkl')))
df_parc.to_pickle(os.path.join(folder['intermediate'], 'parc.pkl'))

ds_income_prop = replace_strings(ds_income_prop, replace_dict)
ds_income_prop.index = ds_income_prop.index.set_names('Income class owner', 'DECILE_PB')
logging.debug('Saving ds_income_prop as pickle in '.format(os.path.join(folder['intermediate'], 'ds_income_prop.pkl')))
ds_income_prop.to_pickle(os.path.join(folder['intermediate'], 'ds_income_prop.pkl'))

end = time.time()
logging.debug('Module time {:.1f} secondes.'.format(end - start))
logging.debug('End')
