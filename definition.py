from math import log
from input import language_dict, parameters_dict


from itertools import product
import pandas as pd

# TODO: investment_horizon_series


class Housing:

    def __init__(self, housing_type, occupancy_status, energy_performance, heating_energy, income_class,
                 income_class_owner):

        self.housing_type = housing_type
        self.occupancy_status = occupancy_status

        self.decision_maker = (self.housing_type, self.occupancy_status)

        self.income_class = income_class
        self.income = parameters_dict['income_series'].loc[income_class]
        self.income_class_owner = income_class_owner
        self.income_owner = parameters_dict['income_series'].loc[income_class]

        self.energy_performance = energy_performance
        self.energy_consumption_theoretical = parameters_dict['energy_consumption_series'].loc[self.energy_performance]

        self.heating_energy = heating_energy

        self.investment_horizon = parameters_dict['investment_horizon_series'].loc[self.decision_maker]
        self.interest_rate = parameters_dict['interest_rate_series'].loc[(self.housing_type, self.occupancy_status, self.income_class)]
        self.discount_factor = (1 - (1 + self.interest_rate)**-self.investment_horizon) / self.interest_rate

        self.surface = parameters_dict['surface'].loc[self.housing_type, self.occupancy_status]

    def energy_cost(self, energy_price_series, energy_performance='Current'):
        """
        Calculate energy cost for a specific segment for all years.
        Intermediate calculation:
        - budget_share ;
        - use_intensity ;
        - actual energy consumption.

        Argument:
        - energy_price [!]: Do not necessary match self.heating_energy.

        Optional:
        energy_performance
        - 'Current'(default): calculate for the attribute of the housing object.
        - 'A' or 'B', ...: calculate the energy_cost for this new energy performance.

        Return pd.Series
        """
        # TODO: surface_series

        if energy_performance == 'Current':
            energy_consumption_theoretical = self.energy_consumption_theoretical
        elif energy_performance in language_dict['energy_performance_list']:
            energy_consumption_theoretical = parameters_dict['energy_consumption_series'].loc[energy_performance]
        else:
            raise

        budget_share = (energy_price_series * self.surface * energy_consumption_theoretical) / self.income
        use_intensity = -0.191 * budget_share.apply(log) + 0.1105
        energy_consumption_actual = use_intensity * energy_consumption_theoretical
        energy_cost = energy_consumption_actual * energy_price_series

        return budget_share, use_intensity, energy_consumption_actual, energy_cost

    def discounted_energy_prices(self, energy_prices_df):
        """
        Calculate NPV of energy cost based on energy_performance and energy_heating of the Housing object and the
        energy_prices_df passed as an argument.
        # TODO: Add year or range year as a parameter.
        """
        energy_prices_ds = energy_prices_df.loc[:, self.heating_energy]
        _, _, _, energy_cost = self.energy_cost(energy_prices_ds)
        return energy_cost.sum() * self.discount_factor



