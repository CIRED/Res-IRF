from math import log
from input import language_dict, parameters_dict


from itertools import product
import pandas as pd

# TODO: investment_horizon_series


class Housing:

    def __init__(self, housing_type, occupancy_status, energy_performance, heating_energy, income_class):

        self.housing_type = housing_type
        self.occupancy_status = occupancy_status

        self.decision_maker = (self.housing_type, self.occupancy_status)

        self.income_class = income_class
        self.income = parameters_dict['income_series'].loc[income_class]

        self.energy_performance = energy_performance
        self.energy_consumption_theoretical = parameters_dict['energy_consumption_series'].loc[self.energy_performance]

        self.heating_energy = heating_energy

        self.investment_horizon = parameters_dict['investment_horizon_series'].loc[self.decision_maker]
        self.interest_rate = parameters_dict['interest_rate_series'].loc[(self.housing_type, self.occupancy_status, self.income_class)]
        self.discount_factor = (1 - (1 + self.interest_rate)**-self.investment_horizon) / self.interest_rate

    def energy_cost(self, energy_price, surface=1, energy_performance='Current'):

        if energy_performance == 'Current':
            energy_consumption_theoretical = self.energy_consumption_theoretical
        elif energy_performance in language_dict['energy_performance_list']:
            energy_consumption_theoretical = parameters_dict['energy_consumption_series'].loc[energy_performance]
        else:
            raise

        budget_share = (energy_price * surface * energy_consumption_theoretical) / self.income
        use_intensity = -0.191 * log(budget_share) + 0.1105
        energy_consumption_actual = use_intensity * energy_consumption_theoretical
        energy_cost = energy_consumption_actual * energy_price

        return budget_share, use_intensity, energy_consumption_actual, energy_cost



