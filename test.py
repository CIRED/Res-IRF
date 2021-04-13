
# test val2share

test = prop.groupby(['Occupancy status', 'Housing type']).sum()
assert np.allclose(test, pd.Series(1, index=range(test.shape[0])))

# test stock_mobile2stock_destroyed

"""
from random import randrange
i_test = randrange(type_housing_destroyed_wo_performance.shape[0])
idx_test = type_housing_destroyed_wo_performance.index[i_test]
idx_test_tot = (idx_test[0], idx_test[1], 'G', idx_test[2], idx_test[3], idx_test[4])
print(type_housing_destroyed_wo_performance.loc[idx_test])
print(nb_housing_destroyed_ini.loc[idx_test_tot])
"""