import os
import pandas as pd
import logging
import time

from input import folder
from function_pandas import *
from class_parc import DataFrameParc

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

dfp = DataFrameParc(df_parc)

logging.debug("Remove 'G' from Occupancy status")
dfp.serie = remove_rows(dfp.serie, 'Occupancy status', 'G')
logging.debug('Total number of housing at this point {:,}'.format(dfp.serie.sum()))

logging.debug("Desagregate 'Autres' to 'Wood fuel' and 'Oil fuel'")
name_file = 'fuel_oil_wood_2018.xlsx'
logging.info('Loading {}'.format(os.path.join(folder['input'], 'ds_parc', name_file)))
df_fuel = pd.read_excel(os.path.join(folder['input'], 'ds_parc', name_file), header=[0], index_col=[1, 0])
df_fuel.index.names = ['Heating energy', 'Housing type']
fuel_list = ['Bois*', 'Fioul domestique']
df_fuel = df_fuel[df_fuel.index.get_level_values('Heating energy').isin(fuel_list)]
df_fuel.index = df_fuel.index.set_levels(df_fuel.index.levels[1].str.replace('Appartement', 'AP').str.replace('Maison', 'MA'), level=1)
df_fuel = df_fuel.loc[:, 'Taux du parc en %']

dfp.serie = de_aggregate_value(dfp.serie, df_fuel.copy(), 'Autres', 'Heating energy', fuel_list, 'Housing type')
logging.debug('Total number of housing at this point {:,}'.format(dfp.serie.sum()))

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
dfp.serie = replace_strings(dfp.serie, replace_dict)
logging.debug('Total number of housing at this point {:,}'.format(dfp.serie.sum()))

ds_lp = dfp.serie[dfp.serie.index.get_level_values('Occupancy status') == 'LP']
ds_lp = add_level_prop(ds_lp, ds_income_prop, 'DECILE_PB')
d_temp = remove_rows(dfp.serie, 'Occupancy status', 'LP')
d_temp = add_level_nan(d_temp, 'DECILE_PB')

dfp.serie = pd.concat((ds_lp, d_temp), axis=0)
logging.debug('Total number of housing at this point {:,}'.format(dfp.serie.sum()))

dfp.serie.to_pickle(os.path.join(folder['middle'], 'parc_2018.pkl'))

end = time.time()
logging.debug('Temps du module {} secondes.'.format(end - start))
logging.debug('End')
