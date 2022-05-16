
import pandas as pd
from numpy.random import normal
from itertools import product

energy_prices = pd.read_csv('project/input/sdes_40/energy_prices_2018.csv', index_col=[0], header=[0])
year = 2020

lambda_1 = [0.6, 0.65, 0.7, 0.75]
lambda_2 = [0.85, 0.9, 0.95, 0.97]

scale = (energy_prices.loc[year, :] / 10)
mean = pd.Series(0, index=scale.index)
epsilon = pd.DataFrame(normal(loc=mean, scale=scale, size=(10, 4)), columns=scale.index)

idx = epsilon.index
scenarios = list(product(lambda_1, lambda_2, idx))

price = dict()
for scenario in scenarios:
    l_1 = scenario[0]
    l_2 = scenario[1]

    nu = energy_prices.loc[year, :] - l_1 * energy_prices.loc[year-1, :] - l_2 * energy_prices.loc[year-2, :]

    eps = epsilon.loc[scenario[2]]

    temp = energy_prices.loc[[year - 2, year - 1,  year], :]
    for y in range(year + 1, 2081):
        temp.loc[y, :] = l_1 * temp.loc[y - 1, :] + l_2 * temp.loc[y - 2, :] + nu + eps


    price[scenario] = temp
    break

print('break')
