import os
import pandas as pd
from input import folder

from function_pandas import de_aggregating_series

name_file = os.path.join(folder['middle'], 'parc.pkl')
dsp = pd.read_pickle(name_file)

# test add_level_prop


test1 = pd.Series([0.1, 0.9], index=['aaa', 'bbb'])
test2 = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], index=[['aaa', 'bbb'] * 3, ['Homeowners', 'Homeowners', 'Landlords', 'Landlords', 'Social-housing','Social-housing']])
test2.index.names = [0, 'Occupancy status']
# add levels without any correlation another variable
# a = add_level_prop(dsp, test1, 0)

# add levels with a correlation with another variable
b = de_aggregating_series(dsp, test2, 0)



# test val2share
"""
test = prop.groupby(['Occupancy status', 'Housing type']).sum()
assert np.allclose(test, pd.Series(1, index=range(test.shape[0])))
"""
# test stock_mobile2stock_destroyed

"""
from random import randrange
i_test = randrange(type_housing_destroyed_wo_performance.shape[0])
idx_test = type_housing_destroyed_wo_performance.index[i_test]
idx_test_tot = (idx_test[0], idx_test[1], 'G', idx_test[2], idx_test[3], idx_test[4])
print(type_housing_destroyed_wo_performance.loc[idx_test])
print(nb_housing_destroyed_ini.loc[idx_test_tot])
"""