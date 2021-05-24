import pandas as pd
import os
import numpy as np
from scipy.optimize import fsolve
from copy import deepcopy
from project.utils import *


class HousingStock:
    # detailed output:
    # TODO: add self.output: pd.DataFrame consumption conventional, energy_lcc_final.stack(), cost_renovation,
    #  intangible_cost, lcc_final.stack(), market_share.stack(), renovation_rate
    # TODO: calculate to consumption actual and CO2 emission each year based on stock_seg
    # TODO: calculate investment cost for realized renovation and private cost
    """Class represents housing stocks. Buildings and households archetype.

    Attributes
    __________
    stock : pd.Series

    label2xxx :
    - float (no segments, no time)
    - pd.Series (segments, no time)
    - pd.DataFrame (segments, time)
    - dict[scenario] - [float, pd.Series, pd.DataFrame]

    Methods
    _______
    """

    def __init__(self, stock_seg, levels_values, year=2018,
                 label2area=None,
                 label2horizon=None,
                 label2discount=None,
                 label2income=None,
                 label2consumption=None,
                 stock_type='simple',
                 agent_behavior='myopic'):
        # TODO: add pd.MultiIndex as input type for stock
        """Initialize Housing Stock object.

        Parameters
        ----------
        stock_seg : pd.Series
        MultiIndex levels describing buildings attributes. Values are number of buildings.

        levels_values: dict
        Possible values for levels.

        year: int

        label2area: float, pd.Series, pd.DataFrame, dict, optional
        Area by segments.

        label2horizon: float, pd.Series, pd.DataFrame, dict, optional
        Investment horizon by segments.

        label2discount: float, pd.Series, pd.DataFrame, dict, optional
        Interest rate by segments.

        label2income: float, pd.Series, pd.DataFrame, dict, optional
        Income by segments.

        label2consumption: float, pd.Series, pd.DataFrame, dict, optional
        Consumption by segments

        # TODO: use property to automatically update stock, segments, levels and dimensions
        """

        self._year = year
        self._calibration_year = year
        self._stock_type = stock_type
        self._agent_behavior = agent_behavior

        self._stock_seg = stock_seg
        self._stock_seg_dict = {self.year: self._stock_seg}
        self._segments = stock_seg.index

        self._levels = stock_seg.index.names
        self._dimension = len(self._stock_seg.index.names)

        # explains what kind of levels needs to be used
        self.levels_values = levels_values

        self.label2area = label2area
        self.label2horizon = label2horizon
        self.label2discount = label2discount
        self.label2income = label2income
        self.label2consumption = label2consumption

        self.consumption_conventional_dict = dict()
        self.consumption_info_dict = dict()
        self.energy_cost_dict = dict()

        transitions = [['Energy performance'], ['Heating energy'], ['Energy performance', 'Heating energy']]
        temp = dict()
        for t in transitions:
            temp[tuple(t)] = dict()
        self.consumption_final_info_dict = deepcopy(temp)
        self.consumption_saving_dict = deepcopy(temp)
        self.emission_saving_dict = deepcopy(temp)
        self.consumption_conventional_final_dict = deepcopy(temp)
        # energy_lcc depends on transition as investment horizon depends on it
        self.energy_lcc_dict = deepcopy(temp)
        self.energy_lcc_final_dict = deepcopy(temp)
        self.lcc_final_dict = deepcopy(temp)
        self.market_share_dict = deepcopy(temp)
        self.pv_dict = deepcopy(temp)
        self.npv_dict = deepcopy(temp)
        self.capex_total_dict = deepcopy(temp)
        self.subsidies_detailed_dict = deepcopy(temp)
        self.subsidies_total_dict = deepcopy(temp)

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, val):
        self._year = val

    @property
    def stock_seg(self):
        return self._stock_seg

    @stock_seg.setter
    def stock_seg(self, new_stock):
        self._stock_seg = new_stock
        self._stock_seg_dict = {self.year: self._stock_seg}
        self._segments = new_stock.index

    def add_flow(self, flow_seg):
        self.stock_seg = self.stock_seg + flow_seg

    def __label2(self, label2, scenario=None):
        """Returns segmented value based on self.segments and by using label2 table.

        Parameters
        ----------
        label2: float, pd.Series, pd.DataFrame, dict

        scenario: str, optional

        Returns
        -------
        pd.Series or pd.DataFrame
        """
        if isinstance(label2, dict):
            if scenario is None:
                val = label2[list(label2.keys())[0]]
                val = reindex_mi(val, self._segments, val.index.names)
            else:
                val = reindex_mi(label2[scenario], self._segments, label2[scenario].index.names)
        else:
            val = label2

        if isinstance(val, float) or isinstance(val, int):
            val = pd.Series(val, index=self._segments)
        elif isinstance(val, pd.Series) or isinstance(val, pd.DataFrame):
            val = reindex_mi(val, self._segments, val.index.names)

        return val

    def to_stock_area_seg(self, scenario=None):
        """Suppose that area_seg levels are included in self.levels.
        """
        if self.label2area is None:
            raise AttributeError('Need to define a table from labels2area')

        area = self.__label2(self.label2area, scenario=scenario)

        return area * self._stock_seg

    def discount_factor(self, scenario_horizon=None, scenario_discount=None):
        """Calculate discount factor for all segments.

        Discount factor can be used when agents doesn't anticipate prices evolution.
        Discount factor does not depend on the year it is calculated.
        """
        horizon = self.__label2(self.label2horizon, scenario=scenario_horizon)
        discount_rate = self.__label2(self.label2discount, scenario=scenario_discount)
        return (1 - (1 + discount_rate) ** -horizon) / discount_rate

    @staticmethod
    def interest_rate2series(interest_rate, idx_yr):
        return [(1 + interest_rate) ** -(yr - idx_yr[0]) for yr in idx_yr]

    def discount_rate_series(self, index_yr):
        """Return pd.DataFrame - partial segments in index and years in column - with value corresponding to discount rate.
        """

        if isinstance(self.label2discount, pd.Series):
            discounted_df = self.label2discount.apply(HousingStock.interest_rate2series, args=[index_yr])
            discounted_df = pd.DataFrame.from_dict(dict(zip(discounted_df.index, discounted_df.values))).T
            discounted_df.columns = index_yr
            discounted_df.index.names = self.label2discount.index.names
        elif isinstance(self.label2discount, float):
            discounted_df = pd.Series(HousingStock.interest_rate2series(self.label2discount, index_yr), index=index_yr)
        else:
            return ValueError
        return discounted_df

    def to_income(self, scenario=None):
        if self.label2income is None:
            raise AttributeError('Need to define a table from label2income')
        return self.__label2(self.label2income, scenario=scenario)

    def to_area(self, scenario=None):
        if self.label2area is None:
            raise AttributeError('Need to define a table from label2area')
        return self.__label2(self.label2area, scenario=scenario)

    def income_stats(self):
        """Returns total income and average income for households.
        """
        income_seg = self.to_income()
        total_income = (income_seg * self._stock_seg).sum()
        return total_income, total_income / self._stock_seg

    def to_horizon(self, scenario=None):
        if self.label2horizon is None:
            raise AttributeError('Need to define a table from label2horizon')
        return self.__label2(self.label2horizon, scenario=scenario)

    def to_consumption_conventional(self, scenario=None):
        if self.label2consumption is None:
            raise AttributeError('Need to define a table from label2consumption')
        consumption_conventional = self.__label2(self.label2consumption, scenario=scenario)
        self.consumption_conventional_dict[self.year] = consumption_conventional
        return consumption_conventional

    def to_consumption_actual(self, energy_prices):
        """
        energy_prices is a DataFrame with segments as rows, and years as columns.
        """
        area_seg = self.to_area()
        income_seg = self.to_income()
        energy_prices = reindex_mi(energy_prices, self._segments, energy_prices.index.names)
        consumption_conventional = self.to_consumption_conventional()
        budget_share_seg = ds_mul_df(area_seg * consumption_conventional, energy_prices / income_seg)
        use_intensity_seg = -0.191 * budget_share_seg.apply(np.log) + 0.1105
        consumption_actual = ds_mul_df(consumption_conventional, use_intensity_seg)
        result_dict = {'Area': area_seg,
                       'Income': income_seg,
                       'Energy prices': energy_prices,
                       'Budget share': budget_share_seg,
                       'Use intensity': use_intensity_seg,
                       'Consumption-conventional': consumption_conventional,
                       'Consumption-actual': consumption_actual
                       }

        self.consumption_info_dict[self.year] = {'Budget share': budget_share_seg.loc[:, self.year],
                                                 'Use intensity': use_intensity_seg.loc[:, self.year],
                                                 'Consumption-actual': area_seg * consumption_actual.loc[:, self.year],
                                                 'Consumption-conventional': area_seg * consumption_conventional,
                                                 }

        return result_dict

    @staticmethod
    def consumption2emission(consumption, co2_content):
        co2_content = reindex_mi(co2_content, consumption.index, levels=co2_content.index.names)
        if isinstance(consumption, pd.Series):
            return ds_mul_df(consumption, co2_content)
        elif isinstance(consumption, pd.DataFrame):
            return consumption * co2_content
    
    @staticmethod
    def consumption2cost(consumption, energy_prices):
        """Returns energy cost segmented and for every year based on energy consumption and energy prices.
        €/m2
        """
        energy_prices = reindex_mi(energy_prices, consumption.index, ['Heating energy'])
        if isinstance(energy_prices, pd.DataFrame):
            if isinstance(consumption, pd.DataFrame):
                return energy_prices * consumption
            elif isinstance(consumption, pd.Series):
                return ds_mul_df(consumption, energy_prices)
        elif isinstance(energy_prices, pd.Series):
            if isinstance(consumption, pd.DataFrame):
                return ds_mul_df(energy_prices, consumption)
            elif isinstance(consumption, pd.Series):
                return consumption * energy_prices

    def to_energy_lcc(self, energy_prices, transition=None, consumption='conventional'):
        """Return segmented energy-life-cycle-cost discounted from segments, and energy prices year=yr.

        Energy LCC is calculated on an segment-specific horizon, and using a segment-specific discount rate.
        Because, time horizon depends of type of renovation (label, or heating energy), lcc needs to know which transition.
        Energy LCC can also be calculated for new constructed buildings: kind='new'.
        Transition defined the investment horizon.
        """
        if transition is None:
            transition = ['Energy performance']

        if consumption == 'conventional':
            if self.year in self.consumption_conventional_dict.keys():
                consumption_seg = self.consumption_conventional_dict[self.year]
            else:
                consumption_seg = self.to_consumption_conventional()
        elif consumption == 'actual':
            consumption_seg = self.to_consumption_actual(energy_prices)['Consumption-' + consumption]
        else:
            raise AttributeError("Consumption must be in ['conventional', 'actual']")

        # TODO: energy_prices doesn't have to change each year as HousingStock will know it's myopic
        if self._agent_behavior == 'myopic':
            energy_cost_seg = HousingStock.consumption2cost(consumption_seg, energy_prices.loc[:, self.year])
        else:
            energy_cost_seg = HousingStock.consumption2cost(consumption_seg, energy_prices)
        self.energy_cost_dict[self.year] = energy_cost_seg

        if transition == ['Energy performance']:
            scenario = 'envelope'
        elif transition == ['Heating energy']:
            scenario = 'heater'
        elif transition == ['Energy performance', 'Heating energy']:
            scenario = None
        else:
            raise AttributeError("Transition can only be 'Energy performance' or 'Heating energy' for now")

        # to fasten the script special case where nothing depends on time
        if self._agent_behavior == 'myopic' and consumption == 'conventional':
            discount_factor = self.discount_factor(scenario_horizon=scenario)
            energy_lcc = energy_cost_seg * discount_factor
            energy_lcc.sort_index(inplace=True)
        else:
            # TODO: get the pd.Series with energy cost for each segment with the good horizon to calculate IRR and Payback
            discounted_seg = self.discount_rate_series(energy_prices.columns)
            if isinstance(discounted_seg, pd.Series):
                discounted_seg = pd.concat([discounted_seg] * len(energy_cost_seg.index), axis=1).T
                discounted_seg.index = energy_cost_seg.index
            elif isinstance(discounted_seg, pd.DataFrame):
                discounted_seg = reindex_mi(discounted_seg, energy_cost_seg.index, discounted_seg.index.names)
            energy_cost_discounted_seg = discounted_seg * energy_cost_seg

            def horizon2years(num, year_yr):
                """Return list of years based on a number of years and year.
                """
                return [year_yr + k for k in range(int(num))]

            invest_horizon_seg = self.to_horizon(scenario=scenario)
            invest_years = invest_horizon_seg.apply(horizon2years, args=[self.year])

            def time_series2sum(ds, years, levels):
                """Return sum of ds for each segment based on list of years in invest years.

                Parameters:
                -----------
                ds: pd.Series
                segments as index, time series as column

                years: pd.Series
                list of years to use for each segment

                levels: str
                levels used to catch idxyears

                Returns:
                --------
                float
                """
                idx_invest = [ds[lvl] for lvl in levels]
                idxyears = years.loc[tuple(idx_invest)]
                return ds.loc[idxyears].sum()

            energy_lcc = energy_cost_discounted_seg.reset_index().apply(time_series2sum, args=[invest_years, self._levels],
                                                                        axis=1)
            energy_lcc.index = energy_cost_discounted_seg.index

        self.energy_lcc_dict[tuple(transition)][self.year] = energy_lcc
        return energy_lcc

    def to_transition(self, ds, transition=None):
        """Create pd.DataFrame from pd.Series by duplicating column.

        Create a MultiIndex columns when transition is a list.
        """
        if transition is None:
            transition = ['Energy performance']

        if isinstance(transition, list):
            for t in transition:
                ds = pd.concat([ds] * len(self.levels_values[t]),
                               keys=self.levels_values[t], names=['{} final'.format(t)], axis=1)
            return ds
        else:
            raise AttributeError('transition should be a list')

    def to_energy_saving(self, transition=None, consumption='conventional', co2_content=None, discount_rate=0.04):

        if transition is None:
            transition = ['Energy performance']

        # TODO horizon depends on
        if transition == ['Energy performance']:
            horizon = 30
        elif transition == ['Heating energy']:
            horizon = 16
        else:
            raise
        index_yr = range(self._year, self._year+horizon, 1)

        # Needs to be in dict before if not need to calculate energy consumption
        if consumption == 'conventional':
            consumption_initial = self.consumption_conventional_dict[self.year]
            consumption_final = self.consumption_conventional_final_dict[tuple(transition)][self.year]
        elif consumption == 'actual':
            consumption_initial = self.consumption_info_dict[self.year]['Consumption-actual']
            consumption_final = self.consumption_final_info_dict[tuple(transition)][self.year]['Consumption-actual']
        else:
            raise AttributeError

        for t in transition:
            consumption_final.index.rename('{} final'.format(t),
                                           level=list(consumption_final.index.names).index('{}'.format(t)),
                                           inplace=True)
            if '{} initial'.format(t) in consumption_final.index.names:
                consumption_final.index.rename('{}'.format(t),
                                               level=list(consumption_final.index.names).index('{} initial'.format(t)),
                                               inplace=True)

        consumption_initial = reindex_mi(consumption_initial, consumption_final.index, consumption_initial.index.names)
        energy_saving = consumption_initial - consumption_final

        if consumption == 'conventional':
            energy_saving = pd.concat([energy_saving] * horizon, axis=1)
            energy_saving.columns = index_yr

        discounted_df = pd.Series(HousingStock.interest_rate2series(discount_rate, index_yr), index=index_yr)
        energy_saving_discounted = discounted_df * energy_saving

        if co2_content is not None:
            emission_initial = HousingStock.consumption2cost(consumption_initial, co2_content)
            emission_final = HousingStock.consumption2cost(consumption_final, co2_content)
            emission_saving = emission_initial - emission_final
            return energy_saving, emission_saving

        return energy_saving_discounted

    def to_energy_lcc_final(self, energy_prices, transition=None, consumption='conventional'):
        """Calculate energy life-cycle cost based on transition.
        """
        if transition is None:
            transition = ['Energy performance']

        stock_transition_seg = self.to_transition(pd.Series(dtype='float64', index=self._segments), transition)

        temp = stock_transition_seg.copy()
        for t in transition:
            if t in temp.index.names:
                temp.index.rename('{} initial'.format(t), level=list(temp.index.names).index('{}'.format(t)),
                                  inplace=True)
            temp = temp.stack(dropna=False)
            temp.index.rename('{}'.format(t), level=list(temp.index.names).index('{} final'.format(t)),
                              inplace=True)
        stock_seg_final = pd.Series(dtype='float64', index=temp.index)
        buildings_final = HousingStock(stock_seg_final, self.levels_values, self.year,
                                       label2area=self.label2area,
                                       label2horizon=self.label2horizon,
                                       label2discount=self.label2discount,
                                       label2income=self.label2income,
                                       label2consumption=self.label2consumption,
                                       agent_behavior=self._agent_behavior)

        # TODO: replace stock_type by energy_saving
        if self._stock_type == 'detailed':
            if consumption == 'actual':
                temp = deepcopy(buildings_final.to_consumption_actual(energy_prices))
                self.consumption_final_info_dict[tuple(transition)][self.year] = {
                     'Budget share': temp['Budget share'].loc[:, self.year],
                     'Use intensity': temp['Use intensity'].loc[:, self.year],
                     'Consumption-actual': temp['Area'] * temp['Consumption-actual'].loc[:, self.year],
                     'Consumption-conventional': temp['Area'] * temp['Consumption-conventional']
                     }
            elif consumption == 'conventional':
                self.consumption_conventional_final_dict[tuple(transition)][
                    self.year] = deepcopy(buildings_final.to_consumption_conventional())

            self.consumption_saving_dict[tuple(transition)][self.year] = self.to_energy_saving(transition=transition,
                                                                                               consumption=consumption)

        energy_final_lcc = buildings_final.to_energy_lcc(energy_prices, transition=transition, consumption=consumption)

        for t in transition:
            energy_final_lcc.index.rename('{} final'.format(t),
                                          level=list(energy_final_lcc.index.names).index('{}'.format(t)),
                                          inplace=True)
            energy_final_lcc = energy_final_lcc.unstack('{} final'.format(t))
            if '{} initial'.format(t) in energy_final_lcc.index.names:
                energy_final_lcc.index.rename('{}'.format(t),
                                              level=list(energy_final_lcc.index.names).index('{} initial'.format(t)),
                                              inplace=True)

        energy_final_lcc = energy_final_lcc.reorder_levels(stock_transition_seg.index.names)
        energy_final_lcc.sort_index(inplace=True)
        stock_transition_seg.sort_index(inplace=True)
        assert energy_final_lcc.index.equals(stock_transition_seg.index), 'Index should match'
        self.energy_lcc_final_dict[tuple(transition)][self.year] = energy_final_lcc
        return energy_final_lcc

    def to_lcc_final(self, energy_prices, cost_invest=None, cost_switch_fuel=None, cost_intangible=None,
                     cost_construction=None, transition=None, consumption='conventional', subsidies=None):
        """Calculate life-cycle-cost of home-energy retrofits for every segment and for every possible transition.

        Parameters
        ----------
        energy_prices: pd.Series

        cost_invest: pd.Series, optional
        Label initial, Label final.

        cost_switch_fuel: pd.DataFrame, optional
        Energy initial, Energy final.

        cost_intangible: pd.DataFrame, optional
        Label initial, Label final.

        cost_construction: pd.DataFrame, optional

        consumption: str
        ['conventionnal', 'actual']

        transition: list, optional
        ['label', 'energy', 'label-energy']
        Define transition. Transition can be defined as label transition, energy transition, or label-energy transition.

        subsidies: list, optional

        Returns
        -------
        pd.Series
        param: option - 'label-energy', 'label', 'energy'
        """

        if transition is None:
            transition = ['Energy performance']

        lcc_final_seg = self.to_energy_lcc_final(energy_prices, transition, consumption=consumption)

        if cost_invest is not None:
            cost_invest = reindex_mi(cost_invest, lcc_final_seg.index, cost_invest.index.names)
            cost_invest = reindex_mi(cost_invest, lcc_final_seg.columns, cost_invest.columns.names, axis=1)

        if cost_intangible is not None:
            cost_intangible = reindex_mi(cost_intangible, lcc_final_seg.index, cost_intangible.index.names)
            cost_intangible = reindex_mi(cost_intangible, lcc_final_seg.columns,
                                         cost_intangible.columns.names, axis=1)
            cost_intangible.fillna(0, inplace=True)

        if cost_switch_fuel is not None:
            cost_switch_fuel = reindex_mi(cost_switch_fuel, lcc_final_seg.index, cost_switch_fuel.index.names)
            cost_switch_fuel = reindex_mi(cost_switch_fuel, lcc_final_seg.columns,
                                          cost_switch_fuel.columns.names, axis=1)

        if cost_construction is not None:
            cost_construction = reindex_mi(cost_construction, lcc_final_seg.index, cost_construction.index.names)
            cost_construction = reindex_mi(cost_construction, lcc_final_seg.columns,
                                           cost_construction.columns.names, axis=1)

        lcc_transition_seg = lcc_final_seg
        capex = None
        if transition == ['Energy performance']:
            capex = cost_invest
            if cost_intangible is not None:
                capex += cost_intangible
        elif transition == ['Heating energy']:
            capex = cost_switch_fuel
        elif transition == ['Energy performance', 'Heating energy']:
            if cost_construction is not None:
                capex = cost_construction
            else:
                capex = cost_switch_fuel + cost_invest
            if cost_intangible is not None:
                capex += cost_intangible

        self.capex_total_dict[tuple(transition)][self.year] = capex

        self.subsidies_detailed_dict[tuple(transition)][self.year] = dict()
        total_subsidies = None
        if subsidies is not None:
            for subsidy in subsidies:
                if subsidy.transition == transition:
                    if subsidy.kind == '%':
                        s = subsidy.to_subsidy(cost=capex)
                        s.fillna(0, inplace=True)
                    elif subsidy.kind == '€/kWh':
                        energy_saving = self.consumption_saving_dict[tuple(transition)][self.year].sum(axis=1)
                        for t in transition:
                            energy_saving = energy_saving.unstack('{} final'.format(t))
                        s = subsidy.to_subsidy(energy_saving=energy_saving)
                        s[s < 0] = 0
                        # € -> €/m2
                        s = (s.T * (self.to_area() ** -1)).T

                    if total_subsidies is None:
                        total_subsidies = s
                    else:
                        total_subsidies += s

                    self.subsidies_detailed_dict[tuple(transition)][self.year][subsidy.name] = s
                    self.subsidies_total_dict[tuple(transition)][self.year] = total_subsidies

        if capex is not None:
            lcc_transition_seg += capex
        if total_subsidies is not None:
            lcc_transition_seg += total_subsidies

        self.lcc_final_dict[tuple(transition)][self.year] = lcc_transition_seg
        return lcc_transition_seg

    @staticmethod
    def lcc2market_share(lcc_df, nu=8.0):
        """Returns market share for each segment based on lcc_df.

        Parameters
        ----------
        lcc_df : pd.DataFrame or pd.Series

        nu: int, optional

        Returns
        -------
        DataFrame if lcc_df is DataFrame or Series if lcc_df is Series.
        """

        lcc_reverse_df = lcc_df.apply(lambda x: x ** -nu)
        if isinstance(lcc_df, pd.DataFrame):
            return ds_mul_df(lcc_reverse_df.sum(axis=1) ** -1, lcc_reverse_df)
        elif isinstance(lcc_df, pd.Series):
            return lcc_reverse_df / lcc_reverse_df.sum()

    def to_market_share(self,
                        energy_prices,
                        transition=None,
                        consumption='conventional',
                        cost_invest=None,
                        cost_switch_fuel=None,
                        cost_intangible=None,
                        cost_construction=None,
                        nu=8.0,
                        subsidies=None):

        if transition is None:
            transition = ['Energy performance']

        lcc_final_seg = self.to_lcc_final(energy_prices, cost_invest=cost_invest, cost_switch_fuel=cost_switch_fuel,
                                          cost_intangible=cost_intangible, cost_construction=cost_construction,
                                          transition=transition, consumption=consumption, subsidies=subsidies)

        ms = HousingStock.lcc2market_share(lcc_final_seg, nu=nu)
        # ms.columns.names = ['{} final'.format(transition)]
        self.market_share_dict[tuple(transition)][self.year] = ms
        return ms, lcc_final_seg

    def to_pv(self,
              energy_prices,
              transition=None,
              consumption='conventional',
              cost_invest=None,
              cost_switch_fuel=None,
              cost_intangible=None,
              cost_construction=None,
              subsidies=None,
              nu=8.0):

        if transition is None:
            transition = ['Energy performance']

        ms_final_seg, lcc_final_seg = self.to_market_share(energy_prices,
                                                           transition=transition,
                                                           consumption=consumption,
                                                           cost_invest=cost_invest,
                                                           cost_switch_fuel=cost_switch_fuel,
                                                           cost_intangible=cost_intangible,
                                                           cost_construction=cost_construction,
                                                           subsidies=subsidies,
                                                           nu=nu)

        pv = (ms_final_seg * lcc_final_seg).sum(axis=1)
        self.pv_dict[tuple(transition)][self.year] = pv
        return pv

    def to_npv(self,
               energy_prices,
               transition=None,
               consumption='conventional',
               cost_invest=None,
               cost_switch_fuel=None,
               cost_intangible=None,
               cost_construction=None,
               subsidies=None,
               nu=8.0):

        if transition is None:
            transition = ['Energy performance']

        energy_lcc_seg = self.to_energy_lcc(energy_prices, transition=transition, consumption=consumption)
        pv_seg = self.to_pv(energy_prices,
                            transition=transition,
                            consumption=consumption,
                            cost_invest=cost_invest,
                            cost_switch_fuel=cost_switch_fuel,
                            cost_intangible=cost_intangible,
                            cost_construction=cost_construction,
                            subsidies=subsidies, nu=nu)

        assert energy_lcc_seg.index.equals(pv_seg.index), 'Index should match'

        npv = energy_lcc_seg - pv_seg
        self.npv_dict[tuple(transition)][self.year] = npv
        return npv

    def calibration_market_share(self, energy_prices, market_share_objective, folder_output=None, cost_invest=None,
                                 consumption='conventional', subsidies=None):
        """Returns intangible costs by calibrating market_share.

        TODO: Calibration of intangible cost could be based on absolute value instead of market share.
        """

        lcc_final = self.to_lcc_final(energy_prices, consumption=consumption, cost_invest=cost_invest,
                                      transition=['Energy performance'], subsidies=subsidies)

        # remove idx when label = 'A' (no transition) and label = 'B' (intangible_cost = 0)
        lcc_useful = remove_rows(lcc_final, 'Energy performance', 'A')
        lcc_useful = remove_rows(lcc_useful, 'Energy performance', 'B')

        market_share_temp = HousingStock.lcc2market_share(lcc_useful)
        market_share_objective = reindex_mi(market_share_objective, market_share_temp.index,
                                            market_share_objective.index.names)
        market_share_objective = market_share_objective.reindex(market_share_temp.columns, axis=1)

        def approximate_ms_objective(ms_obj, ms):
            """Treatment of market share objective to facilitate resolution.
            """
            if isinstance(ms, pd.DataFrame):
                ms = ms.loc[ms_obj.name, :]

            idx = (ms.index[ms < 0.005]).intersection(ms_obj.index[ms_obj < 0.005])
            ms_obj[idx] = ms[idx]
            idx = ms_obj[ms_obj > 10 * ms].index
            ms_obj[idx] = 5 * ms[idx]
            idx = ms_obj[ms_obj < ms / 10].index
            ms_obj[idx] = ms[idx] / 5

            return ms_obj / ms_obj.sum()

        ms_obj_approx = market_share_objective.apply(approximate_ms_objective, args=[market_share_temp], axis=1)
        if folder_output is not None:
            ms_obj_approx.to_pickle(os.path.join(folder_output, 'ms_obj_approx.pkl'))

        def solve_intangible_cost(factor, lcc_np, ms_obj, ini=0):
            """Try to solve the equation with lambda=factor.
            """

            def func(intangible_cost_np, lcc, ms, factor, nu=8):
                """Functions of intangible_cost that are equal to 0.

                Returns a vector that should converge toward 0 as intangible cost converge toward optimal.
                """
                result = np.empty(lcc.shape[0])
                market_share_np = (lcc + intangible_cost_np ** 2) ** -nu / np.sum(
                    (lcc + intangible_cost_np ** 2) ** -nu)
                result[:-1] = market_share_np[:-1] - ms[:-1]
                result[-1] = np.sum(intangible_cost_np ** 2) / np.sum(lcc + intangible_cost_np ** 2) - factor
                return result

            x0 = lcc_np * ini
            root, info_dict, ier, message = fsolve(func, x0, args=(lcc_np, ms_obj, factor), full_output=True)

            if ier == 1:
                return ier, root

            else:
                return ier, None

        lambda_min = 0.01
        lambda_max = 0.6
        step = 0.01

        idx_list, lambda_list, intangible_list = [], [], []
        num_label = list(lcc_final.index.names).index('Energy performance')
        for idx in lcc_useful.index:
            num_ini = self.levels_values['Energy performance'].index(idx[num_label])
            labels_final = self.levels_values['Energy performance'][num_ini + 1:]
            # intangible cost would be for index = idx, and labels_final.
            for lambda_current in range(int(lambda_min * 100), int(lambda_max * 100), int(step * 100)):
                lambda_current = lambda_current / 100
                lcc_row_np = lcc_final.loc[idx, labels_final].to_numpy()
                ms_obj_np = ms_obj_approx.loc[idx, labels_final].to_numpy()
                ier, root = solve_intangible_cost(lambda_current, lcc_row_np, ms_obj_np)
                if ier == 1:
                    lambda_list += [lambda_current]
                    idx_list += [idx]
                    intangible_list += [pd.Series(root ** 2, index=labels_final)]
                    # func(root, lcc_row_np, ms_obj_np, lambda_current)
                    break

        intangible_cost = pd.concat(intangible_list, axis=1).T
        intangible_cost.index = pd.MultiIndex.from_tuples(idx_list)
        intangible_cost.index.names = lcc_final.index.names
        intangible_cost.columns.names = lcc_final.columns.names

        assert len(lcc_useful.index) == len(idx_list), "Calibration didn't work for all segments"

        if folder_output is not None:
            intangible_cost.to_pickle(os.path.join(folder_output, 'intangible_cost.pkl'))
        intangible_cost_mean = intangible_cost.groupby('Energy performance', axis=0).mean()
        intangible_cost_mean = intangible_cost_mean.loc[intangible_cost_mean.index[::-1], :]
        return intangible_cost

    def to_segments_construction(self, lvl2drop, new_lvl_dict):
        """Returns segments_new from segments.

        Segments_new doesn't get Income class owner at first and got other Energy performance value.
        """
        levels_wo = [lvl for lvl in self._levels if lvl not in lvl2drop]
        segments_new = get_levels_values(self._segments, levels_wo)
        segments_new = segments_new.drop_duplicates()
        for level in new_lvl_dict:
            segments_new = segments_new.droplevel(level)
            segments_new = segments_new.drop_duplicates()
            segments_new = pd.concat(
                [pd.Series(index=segments_new, dtype='float64')] * len(new_lvl_dict[level]),
                keys=new_lvl_dict[level], names=list(new_lvl_dict.keys()))
            segments_new = segments_new.reorder_levels(levels_wo).index
        return segments_new

    def to_io_share_seg(self):
        levels = [lvl for lvl in self.stock_seg.index.names if lvl not in ['Income class owner', 'Energy performance']]
        temp = val2share(self.stock_seg, levels, option='column')
        io_share_seg = temp.groupby('Income class owner', axis=1).sum()
        return io_share_seg

    @staticmethod
    def learning_by_doing(knowledge, cost, learning_rate, cost_lim=None):
        """ Calculate new cost after considering learning-by-doing effect.

        Investment costs decrease exponentially with the cumulative sum of operations so as to capture
        a “learning-by-doing” process.
        """
        # TODO: create function with add column level
        lbd = knowledge ** (np.log(1 + learning_rate) / np.log(2))
        if cost_lim is not None:
            cost_lim = cost_lim.unstack('Energy performance final')
            level = 'Heating energy final'
            indexes = lbd.index.get_level_values(level).unique()
            temp = add_level(cost_lim.copy(), indexes, axis=1)
            return (ds_mul_df(lbd, cost.T) + ds_mul_df(1 - lbd, temp.T)).T
        else:
            idx_union = lbd.index.union(cost.T.index)
            return ds_mul_df(lbd.reindex(idx_union), cost.T.reindex(idx_union)).T

    @staticmethod
    def information_rate(knowledge, info_max, info_param):
        """Returns information rate. More info_rate is high, more intangible_cost are low.
        Intangible renovation costs decrease according to a logistic curve with the same cumulative
        production so as to capture peer effects and knowledge diffusion.
        intangible_cost[yr] = intangible_cost[calibrationyear] * info_rate with info rate [1-info_rate_max ; 1]
        This function calibrate a logistic function, so rate of decrease is set at  25% for a doubling of cumulative
        production.
        """

        def equations(p, sh=info_max, alpha=info_param):
            a, r = p
            return (1 + a * np.exp(-r)) ** -1 - sh, (1 + a * np.exp(-2 * r)) ** -1 - sh - (1 - alpha) * sh + 1

        a, r = fsolve(equations, (1, -1))

        return logistic(knowledge, a=a, r=r) + 1 - info_max

    @staticmethod
    def acceleration_information(knowledge, cost_intangible, info_max, info_param):
        info_rate = HousingStock.information_rate(knowledge, info_max, info_param)

        temp = cost_intangible.T.copy()
        if isinstance(temp.index, pd.MultiIndex):
            info_rate = info_rate.reorder_levels(temp.index.names)
        cost_intangible = ds_mul_df(info_rate.loc[temp.index], temp).T
        return cost_intangible


class HousingStockRenovated(HousingStock):
    """Class that represents an existing buildings stock that can (i) renovate buildings, (ii) demolition buildings.

    Some stocks imply change for other stock: stock_master.
    Stock_master should be property as the setter methods need to change all dependencies stock: stock_slave.
    As they need to be initialize there is a private attribute in the init.
    Example:
         A modification of stock_seg will change stock_mobile_seg. stock_mobile_seg cannot change directly.

    Some stocks doesn't have stock_master and stock_slave: they are public.
    Example:

    Some stocks need to be stored in a dict. Their modification need to be updated in the dict.
    Theses stocks are properties as the setter method updates the stock_dict.
    As they need to be initialize there is a private attribute in the init.
    Dict cannot be change directly as they depend on stock (only when to change in the setter method of the stock)
    Example:
        _stock_knowledge_dict = _stock_knowledge_dict.append({year: stock_knowledge})
    These stocks can be a slave stock. In that case all updates are in the stock master setter methods.
    Example:
        _stock_seg_mobile & _stock_seg_mobile_dict
    """

    def __init__(self, stock_seg, levels_values, year=2018,
                 residual_rate=0.0, destruction_rate=0.0,
                 rate_renovation_ini=None, learning_year=None,
                 npv_min=None, rate_max=None, rate_min=None,
                 label2area=None,
                 label2horizon=None,
                 label2discount=None,
                 label2income=None,
                 label2consumption=None):

        super().__init__(stock_seg, levels_values, year,
                         label2area=label2area,
                         label2horizon=label2horizon,
                         label2discount=label2discount,
                         label2income=label2income,
                         label2consumption=label2consumption,
                         stock_type='detailed')

        self.residual_rate = residual_rate
        self._destruction_rate = destruction_rate

        # slave stock of stock_seg property
        self._stock_seg_mobile = stock_seg * (1 - residual_rate)
        self._stock_seg_mobile_dict = {year: stock_seg * (1 - residual_rate)}
        self._stock_seg_residual = stock_seg * residual_rate
        self._stock_area_seg = self.to_stock_area_seg()

        # initializing knowledge
        self.flow_area_renovated_seg = self.flow_area_renovated_seg_ini(rate_renovation_ini, learning_year)
        self._flow_knowledge_ep = self.to_flow_knowledge()
        self._stock_knowledge_ep = self.to_flow_knowledge()
        self._stock_knowledge_ep_dict = {year: self._stock_knowledge_ep}
        self._knowledge = self._stock_knowledge_ep / self._stock_knowledge_ep

        # share of decision-maker in the total stock
        self._dm_share_tot = stock_seg.groupby(['Occupancy status', 'Housing type']).sum() / stock_seg.sum()

        # calibration
        self.rho_seg = pd.Series()
        self._npv_min = npv_min
        self._rate_max = rate_max
        self._rate_min = rate_min

        # attribute to keep information bu un-necessary for the endogeneous
        self.flow_demolition_dict = {}
        self.flow_remained_dict = {}
        self.flow_renovation_label_energy_dict = {}
        self.flow_renovation_label_dict = {}

        transitions = [['Energy performance'], ['Heating energy'], ['Energy performance', 'Heating energy']]
        temp = dict()
        for t in transitions:
            temp[tuple(t)] = dict()
        self.renovation_rate_dict = deepcopy(temp)

    @property
    def stock_seg(self):
        return self._stock_seg

    @stock_seg.setter
    def stock_seg(self, new_stock_seg):
        """Master stock that implement modification for stock slave.
        """
        self._segments = new_stock_seg.index

        self._stock_seg = new_stock_seg
        self._stock_seg_dict[self.year] = new_stock_seg
        self._stock_seg_mobile = self.stock_seg * (1 - self.residual_rate)
        self._stock_seg_mobile_dict[self.year] = self._stock_seg_mobile
        self._stock_seg_residual = self.stock_seg * self.residual_rate
        self._stock_area_seg = self.to_stock_area_seg()

    @property
    def flow_knowledge_ep(self):
        return self._flow_knowledge_ep

    @flow_knowledge_ep.setter
    def flow_knowledge_ep(self, new_flow_knowledge_ep):
        self._flow_knowledge_ep = new_flow_knowledge_ep
        self._stock_knowledge_ep = self._stock_knowledge_ep + new_flow_knowledge_ep
        self._stock_knowledge_ep_dict[self.year] = self._stock_knowledge_ep
        self._knowledge = self._stock_knowledge_ep / self._stock_knowledge_ep_dict[self._calibration_year]

    @property
    def knowledge(self):
        return self._knowledge

    def flow_area_renovated_seg_ini(self, rate_renovation_ini, learning_year):
        """Initialize flow area renovated.

        Flow area renovation is defined as:
         renovation rate (2.7%/yr) x number of learning years (10 yrs) x renovated area (m2).
        """
        renovation_rate_dm = reindex_mi(rate_renovation_ini, self._stock_area_seg.index,
                                        rate_renovation_ini.index.names)
        return renovation_rate_dm * self._stock_area_seg * learning_year

    @property
    def stock_area_seg(self):
        return self._stock_area_seg

    @staticmethod
    def renovate_rate_func(lcc, rho, npv_min, rate_max, rate_min):
        if isinstance(rho, pd.Series):
            rho_f = rho.loc[tuple(lcc.iloc[:-1].tolist())]
        else:
            rho_f = rho

        if np.isnan(rho_f):
            return float('nan')
        else:
            return logistic(lcc.loc[0] - npv_min,
                            a=rate_max / rate_min - 1,
                            r=rho_f,
                            k=rate_max)

    def to_renovation_rate(self, energy_prices,
                           transition=None,
                           consumption='conventional',
                           cost_invest=None,
                           cost_switch_fuel=None,
                           cost_intangible=None,
                           subsidies=None):

        """Routine calculating renovation rate from segments for a particular yr.

        Cost (energy, investment) & rho parameter are also required.
        """
        if transition is None:
            transition = ['Energy performance']

        npv_seg = self.to_npv(energy_prices,
                              transition=transition,
                              consumption=consumption,
                              cost_invest=cost_invest,
                              cost_switch_fuel=cost_switch_fuel,
                              cost_intangible=cost_intangible,
                              subsidies=subsidies)
        renovation_rate_seg = npv_seg.reset_index().apply(HousingStockRenovated.renovate_rate_func,
                                                          args=[self.rho_seg, self._npv_min, self._rate_max,
                                                                self._rate_min], axis=1)
        renovation_rate_seg.index = npv_seg.index
        self.renovation_rate_dict[tuple(transition)][self.year] = renovation_rate_seg
        return renovation_rate_seg

    def to_flow_renovation_label(self, energy_prices,
                                 consumption='conventional',
                                 cost_invest=None,
                                 cost_intangible=None,
                                 subsidies=None):

        transition = ['Energy performance']
        renovation_rate_seg = self.to_renovation_rate(energy_prices,
                                                      transition=transition,
                                                      consumption=consumption,
                                                      cost_invest=cost_invest,
                                                      cost_intangible=cost_intangible,
                                                      subsidies=subsidies)

        flow_renovation_seg = renovation_rate_seg * self.stock_seg

        if self.year in self.market_share_dict[tuple(transition)]:
            market_share_seg_ep = self.market_share_dict[tuple(transition)][self.year]
        else:
            market_share_seg_ep = self.to_market_share(energy_prices,
                                                       transition=transition,
                                                       consumption=consumption,
                                                       cost_invest=cost_invest,
                                                       cost_intangible=cost_intangible,
                                                       subsidies=subsidies)[0]

        flow_renovation_seg_ep = ds_mul_df(flow_renovation_seg, market_share_seg_ep)
        self.flow_renovation_label_dict[self.year] = flow_renovation_seg_ep

        return flow_renovation_seg_ep

    def to_flow_renovation_label_energy(self, energy_prices, consumption='conventional',
                                        cost_switch_fuel=None,
                                        cost_invest=None,
                                        cost_intangible=None,
                                        subsidies=None):
        """De-aggregate stock_renovation_label by final heating energy.

        stock_renovation columns segmented by final label and final heating energy.

        Parameters
        ----------
        energy_prices: pd.DataFrame

        cost_switch_fuel: pd.DataFrame

        cost_invest: pd.DataFrame

        cost_intangible: pd.DataFrame

        consumption: str

        subsidy

        Returns
        -------
        pd.DataFrame
        """

        market_share_seg_he = self.to_market_share(energy_prices,
                                                   transition=['Heating energy'],
                                                   consumption=consumption,
                                                   cost_switch_fuel=cost_switch_fuel,
                                                   subsidies=subsidies)[0]

        ms_temp = pd.concat([market_share_seg_he.T] * len(self.levels_values['Energy performance']),
                            keys=self.levels_values['Energy performance'], names=['Energy performance final'])

        flow_renovation_seg_label = self.to_flow_renovation_label(energy_prices,
                                                                  consumption=consumption,
                                                                  cost_invest=cost_invest,
                                                                  cost_intangible=cost_intangible,
                                                                  subsidies=subsidies)

        sr_temp = pd.concat([flow_renovation_seg_label.T] * len(self.levels_values['Heating energy']),
                            keys=self.levels_values['Heating energy'], names=['Heating energy final'])
        flow_renovation_label_energy = (sr_temp * ms_temp).T
        self.flow_renovation_label_energy_dict[self.year] = flow_renovation_label_energy

        return flow_renovation_label_energy

    def to_flow_remained(self, energy_prices, consumption='conventional',
                         cost_switch_fuel=None,
                         cost_invest=None,
                         cost_intangible=None,
                         subsidies=None):
        """Calculate flow_remained for each segment.

        Returns: positive (+) flow for buildings segment reached by the renovation (final state),
                 negative (-) flow for buildings segment (initial state) that have been renovated.
        """

        flow_renovation_label_energy_seg = self.to_flow_renovation_label_energy(energy_prices,
                                                                                consumption=consumption,
                                                                                cost_switch_fuel=cost_switch_fuel,
                                                                                cost_invest=cost_invest,
                                                                                cost_intangible=cost_intangible,
                                                                                subsidies=subsidies)

        area_seg = reindex_mi(self.label2area, flow_renovation_label_energy_seg.index, self.label2area.index.names)
        flow_area_renovation_seg = ds_mul_df(area_seg, flow_renovation_label_energy_seg)

        flow_renovation_initial_seg = flow_renovation_label_energy_seg.sum(axis=1)
        temp = flow_renovation_label_energy_seg.droplevel('Energy performance', axis=0).droplevel('Heating energy',
                                                                                                  axis=0)
        temp = temp.stack().stack()
        temp.index.rename('Energy performance', 'Energy performance final', inplace=True)
        temp.index.rename('Heating energy', 'Heating energy final', inplace=True)

        flow_renovation_final_seg = temp.reorder_levels(self._levels)

        flow_renovation_final_seg = flow_renovation_final_seg.groupby(flow_renovation_final_seg.index).sum()
        flow_renovation_final_seg.index = pd.MultiIndex.from_tuples(flow_renovation_final_seg.index)
        flow_renovation_final_seg.index.names = self._levels
        union_index = flow_renovation_final_seg.index.union(flow_renovation_initial_seg.index)
        union_index = union_index[~union_index.duplicated()]

        flow_renovation_final_seg = flow_renovation_final_seg.reindex(union_index, fill_value=0)
        flow_renovation_initial_seg = flow_renovation_initial_seg.reindex(union_index, fill_value=0)
        flow_remained_seg = flow_renovation_final_seg - flow_renovation_initial_seg
        np.testing.assert_almost_equal(flow_remained_seg.sum(), 0, err_msg='Not normal')

        self.flow_remained_dict[self.year] = flow_remained_seg

        return flow_remained_seg, flow_area_renovation_seg

    def to_flow_demolition_dm(self):
        flow_demolition = self._stock_seg.sum() * self._destruction_rate
        flow_demolition_dm = self._dm_share_tot * flow_demolition
        # flow_area_demolition_seg_dm = flow_demolition_seg_dm * self.label2area
        return flow_demolition_dm

    def to_flow_demolition_seg(self):
        """ Returns stock_demolition -  segmented housing number demolition.

        Buildings to destroy are chosen in stock_mobile.
        1. type_housing_demolition is respected to match decision-maker proportion; - type_housing_demolition_reindex
        2. income_class, income_class_owner, heating_energy match stock_remaining proportion; - type_housing_demolition_wo_performance
        3. worst energy_performance_label for each segment are targeted. - stock_demolition

        Parameters
        ----------
        destruction_rate : float

        Returns
        -------
        pd.Series segmented
        """

        flow_demolition_dm = self.to_flow_demolition_dm()

        stock_mobile = self._stock_seg_dict[self.year - 1]
        stock_mobile_ini = self._stock_seg_dict[self._calibration_year]
        segments_mobile = stock_mobile.index
        segments_mobile = segments_mobile.droplevel('Energy performance')
        segments_mobile = segments_mobile.drop_duplicates()

        def worst_label(segments_mobile, stock_mobile):
            """Returns worst label for each segment with stock > 1.

            Parameters
            __________
            segments_mobile, pd.MultiIndex
            MultiIndex without Energy performance level.

            stock_mobile, pd.Series
            with Energy performance level

            Returns
            _______
            worst_label_idx, list
            Index with the worst Energy Performance value

            worst_label_dict, dict
            Worst label for each segment
            """
            worst_lbl_idx = []
            worst_lbl_dict = dict()
            for seg in segments_mobile:
                for lbl in self.levels_values['Energy performance']:
                    indx = (seg[0], seg[1], lbl, seg[2], seg[3], seg[4])
                    if stock_mobile.loc[indx] > 1:
                        worst_lbl_idx.append(indx)
                        worst_lbl_dict[seg] = lbl
                        break
            return worst_lbl_idx, worst_lbl_dict

        worst_label_idx, worst_label_dict = worst_label(segments_mobile, stock_mobile)

        # we know type_demolition, then we calculate nb_housing_demolition_ini based on proportion of stock remaining
        levels = ['Occupancy status', 'Housing type']
        levels_wo_performance = [lvl for lvl in self._levels if lvl != 'Energy performance']
        stock_remaining_woperformance = self._stock_seg.groupby(levels_wo_performance).sum()
        stock_share_dm = val2share(stock_remaining_woperformance, levels, option='column')
        flow_demolition_dm_re = reindex_mi(flow_demolition_dm, stock_share_dm.index, levels)
        flow_demolition_wo_ep = ds_mul_df(flow_demolition_dm_re, stock_share_dm)
        flow_demolition_wo_ep = flow_demolition_wo_ep.stack(
            flow_demolition_wo_ep.columns.names)
        np.testing.assert_almost_equal(flow_demolition_dm.sum(), flow_demolition_wo_ep.sum(),
                                       err_msg='Not normal')

        # we don't have the information about which labels are going to be destroyed first
        prop_stock_worst_label = stock_mobile.loc[worst_label_idx] / stock_mobile_ini.loc[
            worst_label_idx]

        flow_demolition_ini = reindex_mi(flow_demolition_wo_ep, prop_stock_worst_label.index,
                                         levels_wo_performance)
        # initialize nb_housing_demolition_theo for worst label based on how much have been demolition so far
        # TODO:bad initialization as flow_demolition_theo < flow_demolition_ini after year 1
        flow_demolition_theo = prop_stock_worst_label * flow_demolition_ini

        # we year with the worst label and we stop when nb_housing_demolition_theo == 0
        flow_demolition = pd.Series(0, index=stock_mobile.index, dtype='float64')
        for segment in segments_mobile:
            label = worst_label_dict[segment]
            num = self.levels_values['Energy performance'].index(label)
            idx_tot = (segment[0], segment[1], label, segment[2], segment[3], segment[4])

            while flow_demolition_theo.loc[idx_tot] != 0:
                # stock_demolition cannot be sup to stock_mobile and to flow_demolition_theo
                flow_demolition.loc[idx_tot] = min(stock_mobile.loc[idx_tot], flow_demolition_theo.loc[idx_tot])
                if label != 'A':
                    num += 1
                    label = self.levels_values['Energy performance'][num]
                    labels = self.levels_values['Energy performance'][:num + 1]
                    idx = (segment[0], segment[1], segment[2], segment[3], segment[4])
                    idx_tot = (segment[0], segment[1], label, segment[2], segment[3], segment[4])
                    idxs_tot = [(segment[0], segment[1], label, segment[2], segment[3], segment[4]) for label in labels]

                    # flow_demolition_theo: remaining housing that need to be destroyed for this segment
                    flow_demolition_theo[idx_tot] = flow_demolition_wo_ep.loc[idx] - \
                                                          flow_demolition.loc[idxs_tot].sum()

                else:
                    flow_demolition_theo[idx_tot] = 0

        self.flow_demolition_dict[self.year] = flow_demolition
        return flow_demolition

    def to_flow_knowledge(self):
        """Returns knowledge renovation.
        """
        if isinstance(self.flow_area_renovated_seg, pd.Series):
            flow_area_renovated_ep = self.flow_area_renovated_seg.groupby(['Energy performance']).sum()
        elif isinstance(self.flow_area_renovated_seg, pd.DataFrame):
            flow_area_renovated_ep = self.flow_area_renovated_seg.groupby('Energy performance', axis=1).sum().sum()
        else:
            raise ValueError('Flow area renovated segmented should be a DataFrame (Series for calibration year')
        # knowledge_renovation_ini depends on energy performance final
        flow_knowledge_renovation = pd.Series(dtype='float64',
                                              index=[ep for ep in self.levels_values['Energy performance'] if
                                                     ep != 'G'])

        def flow_area2knowledge(flow_knowledge_renovation, label1, label2):
            if label1 in flow_area_renovated_ep.index and label2 in flow_area_renovated_ep.index:
                flow_knowledge_renovation.loc[label1] = flow_area_renovated_ep.loc[label1] + flow_area_renovated_ep.loc[
                    label2]
                flow_knowledge_renovation.loc[label2] = flow_area_renovated_ep.loc[label1] + flow_area_renovated_ep.loc[
                    label2]
            elif label2 not in flow_area_renovated_ep.index:
                flow_knowledge_renovation.loc[label1] = flow_area_renovated_ep.loc[label1]
                flow_knowledge_renovation.loc[label2] = flow_area_renovated_ep.loc[label1]
            else:
                flow_knowledge_renovation.loc[label1] = flow_area_renovated_ep.loc[label2]
                flow_knowledge_renovation.loc[label2] = flow_area_renovated_ep.loc[label2]
            return flow_knowledge_renovation

        flow_knowledge_renovation = flow_area2knowledge(flow_knowledge_renovation, 'A', 'B')
        flow_knowledge_renovation = flow_area2knowledge(flow_knowledge_renovation, 'C', 'D')
        flow_knowledge_renovation = flow_area2knowledge(flow_knowledge_renovation, 'E', 'F')

        flow_knowledge_renovation.index.set_names('Energy performance final', inplace=True)
        return flow_knowledge_renovation

    def update_stock(self, flow_demolition_seg, flow_remained_seg, flow_area_renovation_seg=None):

        # update segmented stock considering demolition
        self.add_flow(- flow_demolition_seg)

        # update segmented stock  considering renovation
        self.add_flow(flow_remained_seg)

        if flow_area_renovation_seg is not None:
            self.flow_area_renovation_seg = flow_area_renovation_seg
            flow_knowledge_renovation = self.to_flow_knowledge()
            self.flow_knowledge_ep = flow_knowledge_renovation

    def calibration_renovation_rate(self, energy_prices, renovation_rate_obj, consumption='conventional',
                                    cost_invest=None, cost_intangible=None):
        # TODO: rho_seg must be determine for all future indexes
        # TODO: weight_dm doesn't make a lot of sense
        npv_df = self.to_npv(energy_prices,
                             transition=['Energy performance'],
                             consumption=consumption,
                             cost_invest=cost_invest,
                             cost_intangible=cost_intangible)

        renovation_rate_obj = reindex_mi(renovation_rate_obj, npv_df.index, renovation_rate_obj.index.names)
        rho = (np.log(self._rate_max / self._rate_min - 1) - np.log(
            self._rate_max / renovation_rate_obj - 1)) / (npv_df - self._npv_min)

        stock_ini_wooner = self.stock_seg.groupby(
            [lvl for lvl in self._levels if lvl != 'Income class owner']).sum()
        seg_stock_dm = stock_ini_wooner.to_frame().pivot_table(index=['Occupancy status', 'Housing type'],
                                                               columns=['Energy performance', 'Heating energy',
                                                                        'Income class'])
        seg_stock_dm = seg_stock_dm.droplevel(None, axis=1)

        weight_dm = ds_mul_df((stock_ini_wooner.groupby(['Occupancy status', 'Housing type']).sum()) ** -1,
                              seg_stock_dm)
        rho = rho.droplevel('Income class owner', axis=0)
        rho = rho[~rho.index.duplicated()]
        rho_df = rho.to_frame().pivot_table(index=weight_dm.index.names, columns=weight_dm.columns.names)
        rho_df = rho_df.droplevel(None, axis=1)
        rho_dm = (rho_df * weight_dm).sum(axis=1)
        rho_seg = reindex_mi(rho_dm, self._segments, rho_dm.index.names)

        return rho_seg


class HousingStockConstructed(HousingStock):
    def __init__(self, stock, levels_values, year, stock_needed_ts,
                 param_share_multi_family=None,
                 os_share_ht=None,
                 io_share_seg=None,
                 stock_area_existing_seg=None,
                 label2area=None,
                 label2horizon=None,
                 label2discount=None,
                 label2income=None,
                 label2consumption=None):

        super().__init__(stock, levels_values, year,
                         label2area=label2area,
                         label2discount=label2discount,
                         label2income=label2income,
                         label2consumption=label2consumption,
                         label2horizon=label2horizon, stock_type='simple')

        self._flow_constructed = 0
        self._flow_constructed_dict = {self.year: self._flow_constructed}

        self._flow_constructed_seg = None
        self._flow_constructed_seg_dict = {self.year: self._flow_constructed_seg}
        self._stock_constructed_seg_dict = {self.year: self._flow_constructed_seg}

        self._stock_needed_ts = stock_needed_ts
        self._stock_needed = stock_needed_ts.loc[self._calibration_year]
        # used to estimate share of housing type
        self._share_multi_family_tot_dict = HousingStockConstructed.to_share_multi_family_tot(stock_needed_ts,
                                                                                              param_share_multi_family)
        self._share_multi_family_tot = self._share_multi_family_tot_dict[self._calibration_year]

        # used to let share of occupancy status in housing type constant
        self._os_share_ht = os_share_ht

        # used to estimate share of income class owner
        self._io_share_seg = io_share_seg

        self._flow_knowledge_construction = None
        self._stock_knowledge_construction_dict = {}
        self._knowledge = None

        self._flow_area_constructed_he_ep = None
        if stock_area_existing_seg is not None:
            self._flow_area_constructed_he_ep = self.to_flow_area_constructed_ini(stock_area_existing_seg)
            # to initialize knowledge
            self.flow_area_constructed_he_ep = self._flow_area_constructed_he_ep
        self._area_construction_dict = {self.year: self.label2area}

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, val):
        self._year = val
        self._stock_needed = self._stock_needed_ts.loc[val]
        self._share_multi_family_tot = self._share_multi_family_tot_dict[val]

    @property
    def flow_constructed(self):
        return self._flow_constructed

    @flow_constructed.setter
    def flow_constructed(self, val):
        self._flow_constructed = val
        self._flow_constructed_dict[self.year] = val

    @property
    def flow_constructed_seg(self):
        return self._flow_constructed_seg

    @flow_constructed_seg.setter
    def flow_constructed_seg(self, val):
        self._flow_constructed_seg = val
        self._flow_constructed_seg_dict[self.year] = val
        if self._stock_constructed_seg_dict[self.year - 1] is not None:
            self._stock_constructed_seg_dict[self.year] = self._stock_constructed_seg_dict[self.year - 1] + val
        else:
            self._stock_constructed_seg_dict[self.year] = val

        flow_area_constructed_seg = HousingStockConstructed.data2area(self.label2area, val)
        self.flow_area_constructed_he_ep = flow_area_constructed_seg.groupby(
            ['Energy performance', 'Heating energy']).sum()

    @property
    def flow_area_constructed_he_ep(self):
        return self._flow_area_constructed_he_ep

    @flow_area_constructed_he_ep.setter
    def flow_area_constructed_he_ep(self, val):

        val.index.rename('Energy performance final', 'Energy performance', inplace=True)
        val.index.rename('Heating energy final', 'Heating energy', inplace=True)

        self._flow_area_constructed_he_ep = val
        self._flow_knowledge_construction = val
        if self._stock_knowledge_construction_dict != {}:
            self._stock_knowledge_construction_dict[self.year] = self._stock_knowledge_construction_dict[
                                                                      self.year - 1] + self._flow_knowledge_construction
            self._knowledge = self._stock_knowledge_construction_dict[self.year] / \
                              self._stock_knowledge_construction_dict[
                                  self._calibration_year]
        else:
            self._stock_knowledge_construction_dict[self.year] = self._flow_knowledge_construction
            self._knowledge = self._stock_knowledge_construction_dict[self.year] / \
                              self._stock_knowledge_construction_dict[
                                  self._calibration_year]

    @property
    def knowledge(self):
        return self._knowledge

    @staticmethod
    def data2area(l2area, ds_seg):
        area_seg = reindex_mi(l2area, ds_seg.index, l2area.index.names)
        return ds_seg * area_seg

    @staticmethod
    def to_share_multi_family_tot(stock_needed, param):

        def func(stock, stock_ini, p):
            trend_housing = (stock - stock_ini) / stock * 100
            share = 0.1032 * np.log(10.22 * trend_housing / 10 + 79.43) * p
            return share

        share_multi_family_tot = {}
        stock_needed_ini = stock_needed.iloc[0]
        for year in stock_needed.index:
            share_multi_family_tot[year] = func(stock_needed.loc[year], stock_needed_ini, param)

        return share_multi_family_tot

    def to_share_housing_type(self):

        # self._share_multi_family_tot must be updated first
        stock_need_prev = self._stock_needed_ts[self.year - 1]
        share_multi_family_prev = self._share_multi_family_tot_dict[self.year - 1]
        share_multi_family_construction = (self._stock_needed * self._share_multi_family_tot - stock_need_prev *
                                           share_multi_family_prev) / self.flow_constructed

        ht_share_tot_construction = pd.Series([share_multi_family_construction, 1 - share_multi_family_construction],
                                              index=['Multi-family', 'Single-family'])
        ht_share_tot_construction.index.set_names('Housing type', inplace=True)
        return ht_share_tot_construction

    def to_flow_constructed_dm(self):
        ht_share_tot_construction = self.to_share_housing_type()
        dm_share_tot_construction = ds_mul_df(ht_share_tot_construction, self._os_share_ht).stack()
        return self.flow_constructed * dm_share_tot_construction

    def to_flow_constructed_seg(self, energy_price, cost_intangible=None, cost_construction=None,
                                consumption='conventional', nu=8.0):
        """Returns flow of constructed buildings fully segmented.

        2. Calculate the market-share by decision-maker: market_share_dm;
        3. Multiply by flow_constructed_seg_dm;
        4. De-aggregate levels to add income class owner information.
        """

        market_share_dm = self.to_market_share(energy_price,
                                               cost_construction=cost_construction,
                                               cost_intangible=cost_intangible,
                                               transition=['Energy performance', 'Heating energy'],
                                               consumption=consumption, nu=nu)[0]

        flow_constructed_dm = self.to_flow_constructed_dm()
        flow_constructed_seg = ds_mul_df(flow_constructed_dm, market_share_dm)
        flow_constructed_seg = flow_constructed_seg.stack(flow_constructed_seg.columns.names)

        for t in ['Energy performance', 'Heating energy']:
            flow_constructed_seg.index.rename('{}'.format(t),
                                              level=list(flow_constructed_seg.index.names).index('{} final'.format(t)),
                                              inplace=True)

        # at this point flow_constructed is not segmented by income class as
        return flow_constructed_seg

    def de_aggregate_flow(self, energy_price, cost_intangible=None, cost_construction=None,
                          consumption=None, nu=8.0):
        """Add levels to flow_constructed.

        Specifically, adds Income class, Income class owner

        Parameters
        __________

        io_share_seg: pd.DataFrame
        for each segment (rows) distribution of income class owner decile (columns)

        ic_share_seg: pd.DataFrame
        for each segment (rows) distribution of income class owner decile (columns)
        """

        flow_constructed_seg = self.to_flow_constructed_seg(energy_price,
                                                            cost_intangible=cost_intangible,
                                                            cost_construction=cost_construction,
                                                            consumption=consumption, nu=nu)
        # same repartition of income class
        seg_index = flow_constructed_seg.index
        seg_names = flow_constructed_seg.index.names
        val = 1 / len(self.levels_values["Income class"])
        temp = pd.Series(val, index=self.levels_values["Income class"])
        ic_share_seg = pd.concat([temp] * len(seg_index), axis=1).T
        ic_share_seg.index = seg_index

        flow_constructed_seg = de_aggregate_series(flow_constructed_seg, ic_share_seg)
        flow_constructed_seg.index.names = seg_names + ['Income class']

        # keep the same proportion for income class owner than in the initial parc
        # io_share_seg = reindex_mi(self._io_share_seg, flow_constructed_seg.index, self._io_share_seg.index.names)
        flow_constructed_seg = de_aggregate_series(flow_constructed_seg, self._io_share_seg)
        flow_constructed_seg = flow_constructed_seg[flow_constructed_seg > 0]
        return flow_constructed_seg

    def update_flow_constructed_seg(self, energy_price, cost_intangible=None, cost_construction=None,
                                    consumption='conventional', nu=8.0):
        flow_constructed_seg = self.de_aggregate_flow(energy_price,
                                                      cost_intangible=cost_intangible,
                                                      cost_construction=cost_construction,
                                                      consumption=consumption, nu=nu)

        self.flow_constructed_seg = flow_constructed_seg

    def to_calibration_market_share(self, market_share_objective, energy_price, cost_construction=None):
        """Returns intangible costs construction by calibrating market_share.

        In Scilab intangible cost are calculated with conventional consumption.
        """

        lcc_final = self.to_lcc_final(energy_price,
                                      cost_construction=cost_construction,
                                      transition=['Energy performance', 'Heating energy'],
                                      consumption='conventional')

        market_share_objective.sort_index(inplace=True)
        lcc_final = lcc_final.reorder_levels(market_share_objective.index.names)
        lcc_final.sort_index(inplace=True)

        def approximate_ms_objective(ms_obj):
            """Treatment of market share objective to facilitate resolution.
            """
            ms_obj[ms_obj == 0] = 0.001
            return ds_mul_df(ms_obj.sum(axis=1) ** -1, ms_obj)

        market_share_objective = approximate_ms_objective(market_share_objective)

        def solve_intangible_cost(factor, lcc_np, ms_obj, ini=0.0, nu=8.0):
            """Try to solve the equation with lambda=factor.
            """

            def func(intangible_cost_np, lcc, ms, factor):
                """Functions of intangible_cost that are equal to 0.

                Returns a vector that should converge toward 0 as intangible cost converge toward optimal.
                """
                result = np.empty(lcc.shape[0])
                market_share_np = (lcc + intangible_cost_np ** 2) ** -nu / np.sum(
                    (lcc + intangible_cost_np ** 2) ** -nu)
                result[:-1] = market_share_np[:-1] - ms[:-1]
                result[-1] = np.sum(intangible_cost_np ** 2) / np.sum(lcc + intangible_cost_np ** 2) - factor
                return result

            x0 = lcc_np * ini
            root, info_dict, ier, message = fsolve(func, x0, args=(lcc_np, ms_obj, factor), full_output=True)

            if ier == 1:
                return ier, root

            else:
                return ier, None

        lambda_min = 0.01
        lambda_max = 0.6
        step = 0.01

        assert (lcc_final.index == market_share_objective.index).all()

        labels_final = lcc_final.columns
        idx_list, lambda_list, intangible_list = [], [], []
        for idx in lcc_final.index:
            for lambda_current in range(int(lambda_min * 100), int(lambda_max * 100), int(step * 100)):
                lambda_current = lambda_current / 100
                lcc_row_np = lcc_final.loc[idx, :].to_numpy()
                ms_obj_np = market_share_objective.loc[idx, :].to_numpy()
                ier, root = solve_intangible_cost(lambda_current, lcc_row_np, ms_obj_np)
                if ier == 1:
                    lambda_list += [lambda_current]
                    idx_list += [idx]
                    intangible_list += [pd.Series(root ** 2, index=labels_final)]
                    break

        intangible_cost = pd.concat(intangible_list, axis=1).T
        intangible_cost.index = pd.MultiIndex.from_tuples(idx_list)
        intangible_cost.index.names = lcc_final.index.names
        return intangible_cost

    @staticmethod
    def to_market_share_objective(os_share_ht_construction, he_share_ht_construction, ht_share_tot_construction,
                                  ep_share_tot_construction):
        """

        Market share is the same shape than LCC -
        rows = [Occupancy status, Housing type] - columns = [Heating energy, Energy performance]
        """
        os_he_share_ht_construction = de_aggregate_columns(os_share_ht_construction, he_share_ht_construction)
        os_he_ht_share_tot_construction = ds_mul_df(ht_share_tot_construction,
                                                    os_he_share_ht_construction).stack().stack()

        seg_share_construction = de_aggregating_series(os_he_ht_share_tot_construction,
                                                       ep_share_tot_construction,
                                                       level='Energy performance')
        market_share_objective = val2share(seg_share_construction, ['Occupancy status', 'Housing type'],
                                           option='column')
        # market_share_objective = market_share_objective.droplevel(None, axis=1)
        return market_share_objective

    def update_area_construction(self, elasticity_area_new_ini, available_income_real_pop_ds, area_max_construction):
        """Every year, average area of new buildings increase with available income.

        Trend is based on elasticity area / income.
        eps_area_new decrease over time and reduce elasticity while average area converge towards area_max.
        exogenous_dict['population_total_ds']
        """

        area_construction_ini = self._area_construction_dict[self._calibration_year]
        area_construction_prev = self._area_construction_dict[self.year - 1]
        area_max_construction = area_max_construction.reorder_levels(area_construction_ini.index.names)

        eps_area_new = (area_max_construction - area_construction_prev) / (
                area_max_construction - area_construction_ini)
        eps_area_new = eps_area_new.apply(lambda x: max(0, min(1, x)))
        elasticity_area_new = eps_area_new.multiply(elasticity_area_new_ini)

        available_income_real_pop_ini = available_income_real_pop_ds.loc[self._calibration_year]
        available_income_real_pop = available_income_real_pop_ds.loc[self.year]

        factor_area_new = elasticity_area_new * max(0, (
                available_income_real_pop / available_income_real_pop_ini - 1))

        area_construction = pd.concat([area_max_construction, area_construction_prev * (1 + factor_area_new)],
                                      axis=1).min(axis=1)
        self.label2area = area_construction
        self._area_construction_dict[self.year] = self.label2area

    def to_flow_area_constructed_ini(self, stock_area_existing_seg):
        """To initialize construction knowledge returns area of 2.5 A DPE and 2 B DPE.
        """
        stock_area_new_existing_seg = pd.concat(
            (stock_area_existing_seg.xs('A', level='Energy performance'),
             stock_area_existing_seg.xs('B', level='Energy performance')), axis=0)
        flow_area_constructed_ep = pd.Series(
            [2.5 * stock_area_new_existing_seg.sum(), 2 * stock_area_new_existing_seg.sum()], index=['BBC', 'BEPOS'])
        flow_area_constructed_ep.index.names = ['Energy performance']
        flow_area_constructed_he_ep = add_level(flow_area_constructed_ep,
                                                pd.Index(self.levels_values['Heating energy'], name='Heating energy'))
        return flow_area_constructed_he_ep


class Housing:

    def __init__(self, occupancy_status=None, housing_type=None, performance=None, heating_energy=None,
                 income_class=None, income_class_owner=None, location=None, age=None):

        self.occupancy_status = occupancy_status
        self.housing_type = housing_type
        self.performance = performance
        self.heating_energy = heating_energy
        self.income_class = income_class
        self.income_class_owner = income_class_owner
        self.location = location
        self.age = age

    def __repr__(self):
        txt = ''
        for attribute, val in self.__dict__.items():
            if val is not None:
                txt += '{}: {} \n'.format(attribute, val)
        return txt
