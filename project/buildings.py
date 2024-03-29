# Copyright 2020-2021 Ecole Nationale des Ponts et Chaussées
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Original author Lucas Vivier <vivier@centre-cired.fr>
# Based on a scilab program mainly by written by L.G Giraudet and others, but fully rewritten.

import math

import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

from copy import deepcopy

from utils import reindex_mi, val2share, get_levels_values, remove_rows, add_level


class HousingStock:
    """Multi-attributes, dynamic housing stock.

    HousingStock contains multiple agents. An agent is defined as the combination of a building, an owner and a tenant.
    Specific agent behavior are declared as method.
    """

    def __init__(self, stock, attributes_values,
                 year=2018,
                 attributes2area=None,
                 attributes2horizon=None,
                 attributes2discount=None,
                 attributes2income=None,
                 attributes2consumption=None,
                 price_behavior='myopic',
                 kwh_cumac_transition=None):
        """Initialize HousingStock object.

        Parameters
        ----------
        stock : pd.Series
            MultiIndex levels describing buildings attributes. Values are number of buildings.
        attributes_values : dict
            Possible values for building attributes.
        year: int
        attributes2area : float, pd.Series, pd.DataFrame, dict, optional
            Area by agent attributes.
        attributes2horizon : float, pd.Series, pd.DataFrame, dict, optional
            Investment horizon by agent attributes.
        attributes2discount : float, pd.Series, pd.DataFrame, dict, optional
            Interest rate by agent attributes.
        attributes2income : float, pd.Series, pd.DataFrame, dict, optional
            Income by agent attributes.
        attributes2consumption : float, pd.Series, pd.DataFrame, dict, optional
            Consumption by agent attributes.
        """

        self._year = year
        self._calibration_year = year
        self._price_behavior = price_behavior

        self._stock = stock
        self._stock_dict = {self.year: self._stock}
        self._segments = stock.index

        self._levels = stock.index.names
        self._dimension = len(self._stock.index.names)

        # explains what kind of levels needs to be used
        self.attributes_values = attributes_values
        self.total_attributes_values = attributes_values

        self.attributes2area = attributes2area
        self.attributes2horizon = attributes2horizon
        self.attributes2discount = attributes2discount
        self.attributes2income = attributes2income
        self.attributes2consumption = attributes2consumption
        self.kwh_cumac_transition = kwh_cumac_transition

        self.discount_rate = None
        self.horizon = None
        self.discount_factor = None
        self.area = None
        self.budget_share = None
        self.heating_intensity = None
        self.consumption_conventional = None
        self.consumption_actual = None
        self.energy_cash_flows = None
        self.energy_cash_flows_disc = None

        self.area_all = None
        self.consumption_conventional_all = None
        self.consumption_actual_all = None
        self.energy_cash_flows_all = None
        self.energy_cash_flows_disc_all = None

        transitions = [['Energy performance'], ['Heating energy'], ['Energy performance', 'Heating energy']]
        temp = dict()
        for t in transitions:
            temp[tuple(t)] = dict()

        self.consumption_final = deepcopy(temp)
        self.energy_saving = deepcopy(temp)
        self.emission_saving = deepcopy(temp)

        self.consumption_final_all = deepcopy(temp)
        self.energy_saving_all = deepcopy(temp)
        self.emission_saving_all = deepcopy(temp)

        self.lcc_final = deepcopy(temp)
        self.market_share = deepcopy(temp)
        self.pv = deepcopy(temp)
        self.npv = deepcopy(temp)
        self.capex_total = deepcopy(temp)
        self.capex = deepcopy(temp)
        self.capex_intangible = deepcopy(temp)
        self.subsidies_curtailment = deepcopy(temp)


        self.subsidies_detailed = deepcopy(temp)
        self.subsidies_detailed_euro = deepcopy(temp)
        self.subsidies_total = deepcopy(temp)

        # energy_lcc depends on transition as investment horizon depends on it
        temp = dict()
        for t in transitions:
            temp[tuple(t)] = dict()
            for c in ['conventional', 'actual']:
                temp[tuple(t)][c] = dict()

                self.energy_lcc = deepcopy(temp)
                self.energy_lcc_final = deepcopy(temp)
                self.energy_saving_lc = deepcopy(temp)
                self.emission_saving_lc = deepcopy(temp)

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, val):
        self._year = val

    @property
    def calibration_year(self):
        return self._calibration_year

    @property
    def stock(self):
        return self._stock

    @stock.setter
    def stock(self, new_stock):
        self._stock = new_stock
        self._stock_dict[self.year] = new_stock
        self._segments = new_stock.index

    @property
    def stock_dict(self):
        return self._stock_dict

    @staticmethod
    def data2area(l2area, ds_seg):
        """Multiply unitary measured data by surface data.

        Parameters
        ----------
        l2area : pd.Series
            Attributes indexed data surface.
        ds_seg : pd.Series or pd.DataFrame
            Attributes indexed data to be multiplied by data area.

        Returns
        -------
        pd.Series or pd.DataFrame
        """
        area_seg = reindex_mi(l2area, ds_seg.index, l2area.index.names)
        return ds_seg * area_seg

    def add_flow(self, flow):
        """Add flow to stock attribute object.

        Parameters
        ----------
        flow : pd.Series
            Attributes indexed flow to be added to stock.
        """

        flow = flow.reorder_levels(self.stock.index.names)
        flow = flow.reindex(self.stock.index).fillna(0)

        new_stock = self.stock + flow
        new_stock.fillna(0, inplace=True)
        assert new_stock.min() >= 0, 'Buildings stock cannot be negative'
        self.stock = new_stock.copy()

    @staticmethod
    def _attributes2(segments, attributes2, scenario=None):
        """Returns segmented value based on self._segments and by using attributes2 table.

        Parameters
        ----------
        segments : pd.MultiIndex
            Indexes to use.
        attributes2 : float, pd.Series, pd.DataFrame, dict
            Attributes indexed data.
        scenario : str, optional
            Scenario needs to be specified if attributes2 is a dictionary.

        Returns
        -------
        pd.Series or pd.DataFrame
            Attributes indexed data based on segments pd.MultiIndex.
        """
        if isinstance(attributes2, dict):
            if scenario is None:
                val = attributes2[list(attributes2.keys())[0]]
                val = reindex_mi(val, segments)
            else:
                if isinstance(attributes2[scenario], float) or isinstance(attributes2[scenario], int):
                    val = pd.Series(attributes2[scenario], index=segments)
                else:
                    val = reindex_mi(attributes2[scenario], segments)
        else:
            val = attributes2

        if isinstance(val, float) or isinstance(val, int):
            val = pd.Series(val, index=segments)
        elif isinstance(val, pd.Series) or isinstance(val, pd.DataFrame):
            val = reindex_mi(val, segments)
        return val

    def to_stock_area(self, scenario=None, segments=None):
        """Returns area (surface) of current building stock by agent.

        Parameters
        ----------
        segments : pd.MultiIndex, optional
        scenario : str, optional
            Scenario needs to be specified if attributes2area is a dictionary.

        Returns
        -------
        pd.Series
            Area by agent.
        """
        if self.attributes2area is None:
            raise AttributeError('Need to define a table from attributes2area')
        if segments is None:
            segments = self._segments
        area = HousingStock._attributes2(segments, self.attributes2area, scenario=scenario)
        return area * self._stock

    def to_income(self, scenario=None, segments=None):
        """Returns annual income by agent.

        Parameters
        ----------
        segments : pd.MultiIndex, optional
            Indexed used to reindex data. Default building stock indexes.
        scenario : str, optional
            Scenario needs to be specified if attributes2income is a dictionary.

        Returns
        -------
        pd.Series
            Annual income by agent.
        """

        if self.attributes2income is None:
            raise AttributeError('Need to define a table from attributes2income')
        if segments is None:
            segments = self._segments
        return HousingStock._attributes2(segments, self.attributes2income, scenario=scenario)

    def to_area(self, scenario=None, segments=None):
        """Returns area by agent.

        Parameters
        ----------
        segments : pd.MultiIndex, optional
            Indexed used to reindex data. Default building stock indexes.
        scenario : str, optional
            Scenario needs to be specified if attributes2income is a dictionary.

        Returns
        -------
        pd.Series
            Area by agent.
        """

        if self.attributes2area is None:
            raise AttributeError('Need to define a table from attributes2area')
        if segments is None:
            segments = self._segments
        return HousingStock._attributes2(segments, self.attributes2area, scenario=scenario)

    def to_horizon(self, scenario=None, segments=None):
        """Returns investment horizon by agent.

        Parameters
        ----------
        segments : pd.MultiIndex, optional
            Indexed used to reindex data. Default building stock indexes.
        scenario : str, optional
            Scenario needs to be specified if attributes2income is a dictionary.

        Returns
        -------
        pd.Series
            Investment horizon by agent.
        """

        if self.attributes2horizon is None:
            raise AttributeError('Need to define a table from attributes2horizon')
        if segments is None:
            segments = self._segments
        horizon = HousingStock._attributes2(segments, self.attributes2horizon, scenario=scenario)
        return horizon

    def to_discount_rate(self, scenario=None, segments=None):
        """Returns discount rate by agent.

        Parameters
        ----------
        segments : pd.MultiIndex, optional
            Indexed used to reindex data. Default building stock indexes.
        scenario : str, optional
            Scenario needs to be specified if attributes2income is a dictionary.

        Returns
        -------
        pd.Series
            Discount rate by agent.
        """
        if self.attributes2discount is None:
            raise AttributeError('Need to define a table from attributes2horizon')
        if segments is None:
            segments = self._segments
        discount_rate = HousingStock._attributes2(segments, self.attributes2discount, scenario=scenario)
        return discount_rate

    def to_discount_factor(self, scenario_horizon=None, scenario_discount=None, segments=None):
        """Calculate discount factor by agent.

        Discount factor can be used when agents doesn't anticipate prices evolution.
        Discount factor does not depend on the year it is calculated.

        Returns
        -------
        pd.Series
            Discount factor by agent
        """
        horizon = self.to_horizon(scenario=scenario_horizon, segments=segments)
        discount_rate = self.to_discount_rate(scenario=scenario_discount, segments=segments)
        discount_factor = (1 - (1 + discount_rate) ** -horizon) / discount_rate
        self.discount_factor = discount_factor
        return discount_factor

    @staticmethod
    def rate2time_series(rate, yrs):
        return [(1 + rate) ** -(yr - yrs[0]) for yr in yrs]

    @staticmethod
    def to_discounted(df, rate):
        """Returns discounted DataFrame from DataFrame and discount rate.

        Parameters
        __________
        df: pd.DataFrame
            Time series data by agent. Agents are in row and years in columns.
        rate: float or pd.Series
            Discount rate constant or by agent.

        Returns
        -------
        pd.DataFrame
            Discounted data. Agents are in row and years in columns.
        """

        if isinstance(rate, int) and rate == 0:
            return df

        yrs = df.columns
        if isinstance(rate, float):
            discount = pd.Series(HousingStock.rate2time_series(rate, yrs), index=yrs)
        elif isinstance(rate, pd.Series):
            discounted_df = rate.apply(HousingStock.rate2time_series, args=[yrs])
            discounted_df = pd.DataFrame.from_dict(dict(zip(discounted_df.index, discounted_df.values))).T
            discounted_df.columns = yrs
            discounted_df.index.names = rate.index.names
            discount = reindex_mi(discounted_df, df.index, discounted_df.index.names)

        else:
            raise

        return discount * df

    def to_consumption_conventional(self, scenario=None, segments=None):
        """Returns conventional energy consumption by agent.

        Parameters
        ----------
        segments : pd.MultiIndex, optional
            Indexed used to reindex data. Default building stock indexes.
        scenario : str, optional
            Scenario needs to be specified if attributes2income is a dictionary.

        Returns
        -------
        pd.Series
            Conventional energy consumption by agent.
        """
        if segments is None:
            segments = self._segments

        idx = segments.sort_values()

        if self.consumption_conventional is not None and self.consumption_conventional.index.sort_values().equals(idx):
            return self.consumption_conventional

        else:
            if self.attributes2consumption is None:
                raise AttributeError('Need to define a table from attributes2consumption')
            return HousingStock._attributes2(segments, self.attributes2consumption, scenario=scenario)

    def to_consumption_actual(self, energy_prices, detailed_output=False, segments=None):
        """Returns actual energy consumption by agent and years.

        A growing number of academic studies point to a gap between the conventional energy consumption predicted
        by energy performance certificates and actual energy consumption.
        The most common explanation is a more intense heating of the heating infrastructure after an energy efficiency
        improvement – a phenomenon commonly referred to as the “rebound effect.”

        Heating Intensity = -0.191 * log(Income Share) + 0.1105

        Parameters
        __________
        energy_prices: pd.DataFrame
        detailed_output: boolean, default False
        segments: pd.Index, optional
            Indexed used to reindex data. Default building stock indexes.

        Returns
        _______
        pd.DataFrame
            Actual energy consumption by agent and years.
        """

        if self.consumption_actual is not None and not detailed_output:
            return self.consumption_actual

        area = self.to_area(segments=segments)
        income = self.to_income(segments=segments)
        if segments is None:
            energy_prices = reindex_mi(energy_prices, self._segments)
        else:
            energy_prices = reindex_mi(energy_prices, segments)

        consumption_conventional = self.to_consumption_conventional(segments=segments)
        budget_share = (area * consumption_conventional * (energy_prices / income).T).T
        heating_intensity = -0.191 * budget_share.apply(np.log) + 0.1105
        consumption_actual = (consumption_conventional * heating_intensity.T).T
        self.area = area
        self.budget_share = budget_share
        self.heating_intensity = heating_intensity
        self.consumption_conventional = consumption_conventional
        self.consumption_actual = consumption_actual
        if detailed_output:
            result_dict = {'Area': area,
                           'Income': income,
                           'Energy prices': energy_prices,
                           'Budget share': budget_share,
                           'Heating intensity': heating_intensity,
                           'Consumption-conventional': consumption_conventional,
                           'Consumption-actual': consumption_actual
                           }
            return result_dict
        else:
            return consumption_actual

    def to_consumption(self, consumption, segments=None, energy_prices=None):
        """Returns conventional or actual energy consumption.

        Parameters
        ----------
        consumption: str, {'conventional', 'actual'}
        segments: pd.Index, optional
            If segments is not filled, use self.segments.
        energy_prices: pd.DataFrame, optional

        Returns
        -------
        pd.Series or pd.DataFrame
            Conventional or actual energy consumption.
        """
        if consumption == 'conventional':
            return self.to_consumption_conventional(segments=segments)
        elif consumption == 'actual':
            return self.to_consumption_actual(energy_prices, detailed_output=False, segments=segments)
        else:
            raise AttributeError("Consumption must be in ['conventional', 'actual']")

    @staticmethod
    def mul_consumption(consumption, mul_df, option='initial'):
        """Multiply any pd.DataFrame to consumption.

        Parameters
        ----------
        consumption: pd.DataFrame or pd.Series
        mul_df: pd.DataFrame or pd.Series
        option: str, {'initial', 'final'}, default 'initial'

        Returns
        -------
        pd.DataFrame or pd.Series
        """
        temp = mul_df.copy()
        if option == 'final':
            if len(temp.index.names) == 1:
                temp.index.names = ['Heating energy final']
            else:
                temp.index.rename('Heating energy final', 'Heating energy', inplace=True)

        temp = reindex_mi(temp, consumption.index)

        if isinstance(temp, pd.DataFrame):
            if isinstance(consumption, pd.DataFrame):
                idx = temp.columns.union(consumption.columns)
                temp = temp.reindex(idx, axis=1)
                consumption = consumption.reindex(idx, axis=1)
                return temp * consumption
            elif isinstance(consumption, pd.Series):
                return (consumption * temp.T).T
        elif isinstance(temp, pd.Series):
            if isinstance(consumption, pd.DataFrame):
                return (temp * consumption.T).T
            elif isinstance(consumption, pd.Series):
                return consumption * temp

    def energy_expenditure(self, energy_price):
        """Calculate total energy expenditure (€) by agent for current year.

        € = €/kWh x kWh/m2 x m2/building x buildings

        Parameters
        ----------
        energy_price: pd.Series or pd.DataFrame

        Returns
        -------
        pd.Series
            Total energy expenditure by agent.
        """
        consumption = self.consumption_actual.loc[:, self.year] * self.area * self.stock
        return HousingStock.mul_consumption(consumption, energy_price).loc[:, self.year]

    @staticmethod
    def to_summed(df, yr_ini, horizon):
        """Sum each rows of df from yr_ini to its horizon.

        Parameters
        ----------
        df : pd.DataFrame
            Agents are in row and years in columns.
        yr_ini : int
        horizon : pd.Series
            pd.MultiIndex as index

        Returns
        -------
        pd.Series
        """
        if isinstance(horizon, int):
            yrs = range(yr_ini, yr_ini + horizon, 1)
            df_summed = df.loc[:, yrs].sum(axis=1)
        elif isinstance(horizon, pd.Series):
            horizon_re = reindex_mi(horizon, df.index, horizon.index.names)

            def horizon2years(num, yr):
                """Return list of years based on a number of years and an initial year.

                Parameters
                ----------
                num : int
                yr : int

                Returns
                -------
                list
                """
                return [yr + k for k in range(int(num))]

            yrs = horizon_re.apply(horizon2years, args=[yr_ini])

            def time_series2sum(ds, years, levels):
                """
                Sum n values over axis=1 based on years to consider for each row.

                Parameters
                ----------
                ds : pd.Series
                    Segments as index, years as column
                years : pd.Series
                    List of years to use for each segment
                levels : str
                    Levels used to catch idxyears

                Returns
                -------
                float
                """
                idx_invest = [ds[lvl] for lvl in levels]
                idxyears = years.loc[tuple(idx_invest)]
                return ds.loc[idxyears].sum()

            df_summed = df.reset_index().apply(time_series2sum, args=[yrs, df.index.names], axis=1)
            df_summed.index = df.index
        else:
            raise
        return df_summed

    def to_energy_lcc(self, energy_prices, transition=None, consumption='conventional', segments=None):
        """Return energy-life-cycle-cost discounted by agents, and energy prices.

        Energy LCC is calculated on an agent-specific horizon, and discount rate.
        Time horizon depends on transition (transition defined the investment horizon).

        Parameters
        ----------
        energy_prices: pd.DataFrame
        transition: list, default ['Energy performance']
        consumption: str, {'conventional', 'actual}, default 'conventional'
        segments: pd.MultiIndex, optional

        Returns
        -------
        pd.Series
            Energy-life-cycle-cost discounted by agents.
        """
        if transition is None:
            transition = ['Energy performance']
        try:
            return self.energy_lcc[tuple(transition)][consumption][self.year]
        except (KeyError, TypeError):
            try:
                energy_cash_flows = self.energy_cash_flows[consumption]
            except (KeyError, TypeError):
                try:
                    energy_cash_flows = self.energy_cash_flows_all[consumption]
                except (KeyError, TypeError):
                    consumption_seg = self.to_consumption(consumption, segments=segments)
                    energy_cash_flows = HousingStock.mul_consumption(consumption_seg, energy_prices)
            if self._price_behavior == 'myopic':
                energy_cash_flows = energy_cash_flows.loc[:, self.year]

            # to discount cash-flows and fasten the script special case where nothing depends on time
            if self._price_behavior == 'myopic' and consumption == 'conventional':
                discount_factor = self.to_discount_factor(scenario_horizon=tuple(transition), segments=segments)
                energy_lcc = energy_cash_flows * discount_factor
                energy_lcc.sort_index(inplace=True)

            else:
                energy_cost_discounted_seg = HousingStock.to_discounted(energy_cash_flows, self.attributes2discount)
                energy_lcc = HousingStock.to_summed(energy_cost_discounted_seg, self.year, self.to_horizon(scenario=tuple(transition)))

            self.energy_lcc[tuple(transition)][consumption][self.year] = energy_lcc

            return energy_lcc

    def to_transition(self, ds, transition=None):
        """Returns pd.DataFrame from pd.Series by adding final state as column with same value.

        Create a MultiIndex columns when it occurs simultaneous transitions.

        Parameters
        ----------
        ds: pd.Series, pd.DataFrame
        transition: list, default ['Energy performance']

        Returns
        -------
        pd.DataFrame
        """
        if transition is None:
            transition = ['Energy performance']

        if isinstance(transition, list):
            for t in transition:
                ds = pd.concat([ds] * len(self.attributes_values[t]),
                               keys=self.attributes_values[t], names=['{} final'.format(t)], axis=1)
            return ds
        else:
            raise AttributeError('transition should be a list')

    @staticmethod
    def initial2final(ds, idx_full, transition):
        """Catch final state agent values as the initial state of another agent.

        When a segment final state match another segment initial state,
        it's therefore fasten to directly catch the value.

        Parameters
        ----------
        ds : pd.Series
            Data to pick values
        idx_full : pd.MultiIndex
            Corresponds to final state data index
        transition : list

        Returns
        -------
        pd.Series
        """
        names_final = ds.index.names
        for t in transition:
            # replace Energy performance by Energy performance final
            names_final = [i.replace(t, '{} final'.format(t)) for i in names_final]

        # select index corresponding to final state
        idx_final = get_levels_values(idx_full, names_final)
        # returns values based on these indexes
        if isinstance(ds, pd.Series):
            ds_final = ds.loc[idx_final]
        elif isinstance(ds, pd.DataFrame):
            ds_final = ds.loc[idx_final, :]
        else:
            raise

        ds_final_re = reindex_mi(ds_final, idx_full, ds_final.index.names)

        # ds_re = reindex_mi(ds, idx_full, ds.index.names)
        return ds_final_re

    def to_final(self, ds, transition=None, segments=None):
        """Returns final state value index by columns and by initial state agent.

        For transition attributes, initial state doesn't influence results.

        Parameters
        ----------
        ds : pd.Series
        transition : list, default ['Energy performance', 'Heating energy']
        segments : pd.Index, optional
            Use self.segments if input is not filled.

        Returns
        -------
        pd.DataFrame
        """
        if transition is None:
            transition = ['Energy performance', 'Heating energy']
        if segments is None:
            stock_transition = self.to_transition(pd.Series(dtype='float64', index=self._segments), transition)
        else:
            stock_transition = self.to_transition(pd.Series(dtype='float64', index=segments), transition)

        stock_transition.fillna(0, inplace=True)
        idx_full = stock_transition.stack(stock_transition.columns.names).index
        ds_final = HousingStock.initial2final(ds, idx_full, transition)
        for t in transition:
            ds_final = ds_final.unstack('{} final'.format(t))
        return ds_final

    def to_consumption_final(self, consumption='conventional', transition=None):
        """Returns conventional or actual energy consumption by agent for all possible final state.

        Parameters
        ----------
        consumption: str, {'conventional', 'actual'}
        transition  list, default ['Energy performance', 'Heating energy']
            Transition defined possible final states.

        Returns
        -------
        pd.Series or pd.DataFrame
            Conventional or actual energy consumption by agent and
        """
        if transition is None:
            transition = ['Energy performance', 'Heating energy']

        try:
            return self.consumption_final[tuple(transition)][consumption]
        except (KeyError, TypeError):
            consumption_initial = self.to_consumption(consumption)
            consumption_final = self.to_final(consumption_initial, transition=transition)
            self.consumption_final[tuple(transition)][consumption] = consumption_final
            return consumption_final

    def to_energy_saving(self, transition=None, consumption='conventional'):
        """Calculate energy saving by agent between initial and all possible final state.

        Parameters
        ----------
        transition: list, default ['Energy performance', 'Heating energy']
            Transition defined possible final states.
        consumption: str, {'conventional', 'actual'}

        Returns
        -------
        pd.Series or pd.DataFrame
            Conventional or actual energy saving by agent.
        """
        if transition is None:
            transition = ['Energy performance']

        try:
            return self.energy_saving[tuple(transition)][consumption]
        except (KeyError, TypeError):

            consumption_initial = self.to_consumption(consumption)
            consumption_final = self.to_consumption_final(consumption=consumption, transition=transition)
            consumption_final = consumption_final.stack(consumption_final.columns.names)
            consumption_initial_re = reindex_mi(consumption_initial, consumption_final.index,
                                                consumption_initial.index.names)
            energy_saving = consumption_initial_re - consumption_final
            self.energy_saving[tuple(transition)][consumption] = energy_saving
            return energy_saving

    def to_energy_saving_lc(self, transition=None, consumption='conventional', discount=0.04):
        """Calculate life-cycle energy saving between initial and all possible final state for the entire project duration.

        Parameters
        ----------
        transition: (Energy performance, ), (Heating energy, ), (Energy performance, Heating energy)
            Transition defined possible final states.
        consumption: {'conventional', 'actual'}
        discount: float, default 0.04

        Returns
        -------
        pd.DataFrame
        """

        energy_saving = self.to_energy_saving(transition=transition, consumption=consumption)
        # energy_saving = HousingStockConstructed.data2area(self.attributes2area, energy_saving)

        if consumption == 'conventional':
            energy_saving = pd.concat([energy_saving] * 30, axis=1)
            energy_saving.columns = range(self.year, self.year + 30, 1)
        energy_saving_disc = HousingStock.to_discounted(energy_saving, discount)
        if 'Energy performance' in transition:
            horizon = 30
        else:
            horizon = 16
        energy_saving_lc = HousingStock.to_summed(energy_saving_disc, self.year, horizon)
        self.energy_saving_lc[tuple(transition)][consumption][self.year] = energy_saving_lc
        return energy_saving_lc

    def to_emission_saving(self, co2_content, transition=None, consumption='conventional', energy_prices=None):
        """Calculate emission saving by agent between initial and all possible final state.

        Parameters
        ----------
        co2_content: pd.DataFrame
        transition: (Energy performance, ), (Heating energy, ), (Energy performance, Heating energy)
            Transition defined possible final states.
        consumption: {'conventional', 'actual'}
        energy_prices: pd.DataFrame

        Returns
        -------
        pd.Series or pd.DataFrame
            Conventional or actual energy saving by agent.
        """

        if transition is None:
            transition = ['Energy performance']

        try:
            return self.emission_saving[tuple(transition)][consumption]
        except (KeyError, TypeError):
            consumption_initial = self.to_consumption(consumption, energy_prices=energy_prices)
            consumption_final = self.to_consumption_final(consumption=consumption, transition=transition)
            if isinstance(consumption_initial, pd.DataFrame):
                t = ['{} final'.format(t) for t in transition]
                consumption_final = consumption_final.stack(t)

            else:
                consumption_final = consumption_final.stack(consumption_final.columns.names)
            consumption_initial_re = reindex_mi(consumption_initial, consumption_final.index)

            emission_initial = HousingStock.mul_consumption(consumption_initial_re, co2_content)
            emission_final = HousingStock.mul_consumption(consumption_final, co2_content, option='final')
            emission_saving = emission_initial - emission_final
            self.emission_saving[tuple(transition)][consumption] = emission_saving
            return emission_saving

    def to_emission_saving_lc(self, co2_content, transition=None, consumption='conventional', horizon=30, discount=0.04,
                              energy_prices=None):
        """Calculate life-cycle emission saving between initial and final state for the entire project duration.

        Parameters
        ----------
        co2_content: pd.DataFrame
        transition: (Energy performance, ), (Heating energy, ), (Energy performance, Heating energy)
            Transition defined possible final states.
        consumption: {'conventional', 'actual'}
        horizon: int, default 30
        discount: float, default 0.04
        energy_prices: pd.DataFrame

        Returns
        -------
        pd.Series or pd.DataFrame
            Conventional or actual emission saving by agent.
        """
        emission_saving = self.to_emission_saving(co2_content, transition=transition, consumption=consumption,
                                                  energy_prices=energy_prices)
        # emission_saving = HousingStockConstructed.data2area(self.attributes2area, emission_saving)

        if consumption == 'conventional':
            emission_saving = pd.concat([emission_saving] * 30, axis=1)
            emission_saving.columns = range(self.year, self.year + 30, 1)
        emission_saving_disc = HousingStock.to_discounted(emission_saving, discount)

        emission_saving_lc = HousingStock.to_summed(emission_saving_disc, self.year, horizon)
        self.emission_saving_lc[tuple(transition)][consumption][self.year] = emission_saving_lc
        return emission_saving_lc

    def to_energy_lcc_final(self, energy_prices, transition=None, consumption='conventional', segments=None):
        """Calculate energy life-cycle cost based on transition.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        transition: (Energy performance, ), (Heating energy, ), (Energy performance, Heating energy)
            Transition defined possible final states.
        consumption: {'conventional', 'actual'}
        segments : pd.Index, optional
            Use self.segments if input is not filled.

        Returns
        -------
        pd.Series, or pd.DataFrame
        """
        if transition is None:
            transition = ['Energy performance']

        try:
            return self.energy_lcc_final[tuple(transition)][consumption][self.year]
        except (KeyError, TypeError):
            # energy performance and heating energy are not necessarily in segments
            index = None
            if segments is not None:
                index = segments
                for t in transition:
                    if t not in segments.names:
                        index = add_level(pd.Series(dtype='float64', index=index),
                                          pd.Index(self.attributes_values[t], name=t)).index

            energy_lcc = self.to_energy_lcc(energy_prices, transition=transition, consumption=consumption,
                                            segments=index)

            energy_lcc_final = self.to_final(energy_lcc, transition=transition, segments=segments)
            self.energy_lcc_final[tuple(transition)][consumption][self.year] = energy_lcc_final
            return energy_lcc_final

    def to_lcc_final(self, energy_prices, cost_invest=None, cost_intangible=None,
                     transition=None, consumption='conventional', subsidies=None, segments=None, energy_lcc=None):
        """Calculate life-cycle-cost of home-energy retrofits by agent and for every possible transition.

        Parameters
        ----------
        energy_prices: pd.DataFrame
            index are heating energy and columns are years.
        cost_invest: dict, optional
            keys are transition (cost_invest['Energy performance']) and item are pd.DataFrame
        cost_intangible: dict, optional
            keys are transition (cost_intangible['Energy performance']) and item are pd.DataFrame
        consumption: {'conventional', 'actual'}, default 'conventional
        transition: list, default ['Energy performance']
            define transition. Transition can be defined as attributes transition, energy transition, or attributes-energy transition.
        subsidies: list, optional
            list of subsidies object
        segments: pd.MultiIndex or pd.Index, optional
        energy_lcc: pd.Series, optional
            Allow the user to chose manually an energy_lcc. Otherwise self.to_energy_lcc_final will be launched.

        Returns
        -------
        pd.DataFrame
            Life-cycle-cost DataFrame is structured for every initial state (index) to every final state defined by transition (columns).
        """

        if transition is None:
            transition = ['Energy performance']

        if energy_lcc is None:
            lcc_final = self.to_energy_lcc_final(energy_prices, transition, consumption=consumption, segments=segments)
        else:
            lcc_final = energy_lcc
            lcc_final = reindex_mi(lcc_final, self.stock.index)

        lcc_transition = lcc_final.copy()
        columns = lcc_transition.columns

        capex = None
        capex_intangible = None
        capex_total = None

        for t in transition:
            if cost_invest[t] is not None:
                c = reindex_mi(cost_invest[t], lcc_final.index)
                c = reindex_mi(c, lcc_final.columns, c.columns.names, axis=1)
                if capex_total is None:
                    capex_total = c.copy()
                    capex = c.copy()
                else:
                    capex_total += c
                    capex += c
            if cost_intangible is not None:
                if cost_intangible[t] is not None:
                    c = reindex_mi(cost_intangible[t], lcc_final.index, cost_intangible[t].index.names)
                    c = reindex_mi(c, lcc_final.columns, c.columns.names, axis=1)
                    c.fillna(0, inplace=True)
                    capex_total += c
                    if capex_intangible is None:
                        capex_intangible = c.copy()
                    else:
                        capex_intangible += c

        self.capex[tuple(transition)][self.year] = capex.copy()
        if capex_intangible is not None:
            self.capex_intangible[tuple(transition)][self.year] = capex_intangible.copy()
        self.capex_total[tuple(transition)][self.year] = capex_total.copy()

        self.subsidies_detailed[tuple(transition)][self.year] = dict()
        self.subsidies_detailed_euro[tuple(transition)][self.year] = dict()

        total_subsidies = None
        if subsidies is not None:
            for policy in subsidies:
                if policy.transition == transition and policy.policy != 'subsidies_curtailment':
                    if policy.policy == 'subsidies' or policy.policy == 'subsidy_tax':
                        if policy.unit == '%':
                            s = policy.to_subsidy(self.year, cost=capex)
                            s = s.reindex(lcc_transition.index, axis=0).reindex(columns, axis=1)
                            s.fillna(0, inplace=True)

                        elif policy.unit == 'euro/kWh':
                            # energy saving is kWh/m2
                            energy_saving = reindex_mi(self.kwh_cumac_transition, capex.index).reindex(capex.columns,
                                                                                                       axis=1).fillna(0)

                            s = policy.to_subsidy(self.year, energy_saving=energy_saving)
                            s[s < 0] = 0
                            s.fillna(0, inplace=True)

                        elif policy.unit == 'euro/m2':
                            s = policy.to_subsidy(self.year, cost=capex)
                            s = s.reindex(lcc_transition.index, axis=0).reindex(columns, axis=1)
                            s.fillna(0, inplace=True)

                        else:
                            raise AttributeError('Subsidies unit can be euro/kWh, euro/m2 or %')

                        if policy.priority is True:
                            capex -= s

                    elif policy.policy == 'regulated_loan':
                        capex_euro = (self.to_area() * capex.T).T
                        s = policy.to_opportunity_cost(capex_euro)
                        s = (s.T * (self.to_area() ** -1)).T
                        s = s.reindex(lcc_transition.index, axis=0).reindex(columns, axis=1)
                        s.fillna(0, inplace=True)

                    if total_subsidies is None:
                        total_subsidies = s.copy()
                    else:
                        total_subsidies += s

                    self.subsidies_detailed[tuple(transition)][self.year]['{} (euro/m2)'.format(policy.name)] = s
                    self.subsidies_detailed_euro[tuple(transition)][self.year]['{} (euro)'.format(policy.name)] = (
                                self.area * s.T).T

        if capex_total is not None:
            lcc_transition += capex_total
        if total_subsidies is not None:
            subsidies_curtailment = [s for s in subsidies if s.policy == 'subsidies_curtailment']
            if subsidies_curtailment != []:
                subsidies_curtailment = subsidies_curtailment[0]
                a = reindex_mi(subsidies_curtailment.value, capex.index)

                curtail = total_subsidies - (a * capex.T).T
                curtail = curtail[curtail > 0].fillna(0)
                total_subsidies -= curtail

                subsidies_curtailed = subsidies_curtailment.subsidies_curtailed

                self.subsidies_curtailment[tuple(transition)][self.year] = curtail
                if curtail.sum().sum() > 0:
                    print('Curtailment')
                    self.subsidies_detailed[tuple(transition)][self.year][
                        '{} (euro/m2)'.format(subsidies_curtailed)] -= curtail
                    self.subsidies_detailed_euro[tuple(transition)][self.year]['{} (euro)'.format(subsidies_curtailed)] -= (
                            self.area * curtail.T).T

            self.subsidies_total[tuple(transition)][self.year] = total_subsidies
            lcc_transition -= total_subsidies

        self.lcc_final[tuple(transition)][self.year] = lcc_transition

        return lcc_transition

    @staticmethod
    def lcc2market_share(lcc_df, nu=8.0):
        """Returns market share for each segment based on lcc_df.

        Parameters
        ----------
        lcc_df : pd.DataFrame or pd.Series
        nu: float, optional

        Returns
        -------
        pd.Series, or pd.DataFrame
        """

        lcc_reverse_df = lcc_df.apply(lambda x: x ** -nu)
        if isinstance(lcc_df, pd.DataFrame):
            return (lcc_reverse_df.sum(axis=1) ** -1 * lcc_reverse_df.T).T
        elif isinstance(lcc_df, pd.Series):
            return lcc_reverse_df / lcc_reverse_df.sum()

    def to_market_share(self, energy_prices, transition=None, consumption='conventional', cost_invest=None,
                        cost_intangible=None, subsidies=None, nu=8.0, segments=None, final=None):
        """Returns market share for each segment and each possible final state.

        Parameter nu characterizing the heterogeneity of preferences is set to 8 in the model.
        Intangible costs are calibrated so that the observed market shares are reproduced in the initial year.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: {'conventional', 'actual'}, default 'conventional
        transition: list, {['Energy performance'], ['Heating energy'], ['Energy performance', 'Heating energy']},
            default ['Energy performance']
        subsidies: list, optional
            List of subsidies object
        segments: pd.MultiIndex, optional
        nu: float or int, default 8.0
        final: list, optional

        Returns
        -------
        pd.DataFrame
            market_share, lcc_final
        """
        if transition is None:
            transition = ['Energy performance']

        lcc_final = self.to_lcc_final(energy_prices, cost_invest=cost_invest, cost_intangible=cost_intangible,
                                      transition=transition, consumption=consumption, subsidies=subsidies,
                                      segments=segments)
        if final is not None:
            lcc_final = lcc_final.loc[:, final]

        market_share = HousingStock.lcc2market_share(lcc_final, nu=nu)
        # ms.columns.names = ['{} final'.format(transition)]

        if final is None:
            self.market_share[tuple(transition)][self.year] = market_share
        return market_share, lcc_final

    def _to_market_share_energy(self, energy_prices, consumption='conventional', cost_invest=None,
                                cost_intangible=None, subsidies=None, nu=8.0, segments=None):
        """Only used for nested technology transition.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: {'conventional', 'actual'}, default 'conventional
        subsidies: list, optional
            List of subsidies object
        segments: pd.MultiIndex, optional
        nu: float or int, default 8.0

        Returns
        -------
        """
        lcc_final = self.to_lcc_final(energy_prices, cost_invest=cost_invest, cost_intangible=cost_intangible,
                                      transition=['Heating energy'], consumption=consumption, subsidies=subsidies,
                                      segments=segments)

        lcc = lcc_final.copy()
        temp = list(lcc_final.index.names)
        temp[temp.index('Energy performance')] = 'Energy performance final'
        lcc.index.names = temp
        lcc = lcc.unstack('Energy performance final')
        lcc = add_level(lcc, pd.Index(self.attributes_values['Energy performance'], name='Energy performance'), axis=0)
        lcc = lcc.reorder_levels(lcc_final.index.names)

        possible_transition = cost_invest['Energy performance'].copy()
        possible_transition[possible_transition > 0] = 1
        possible_transition = reindex_mi(possible_transition, lcc.index)
        possible_transition = reindex_mi(possible_transition, lcc.columns, axis=1)

        lcc = lcc * possible_transition

        lcc = remove_rows(lcc, 'Energy performance', 'A')
        # lcc = remove_rows(lcc, 'Energy performance', 'B')

        lcc = lcc.stack('Energy performance final')
        market_share = HousingStock.lcc2market_share(lcc, nu=nu)

        return market_share

    def to_pv(self, energy_prices, transition=None, consumption='conventional', cost_invest=None, cost_intangible=None,
              subsidies=None, nu=8.0):
        """Calculate present value of home-energy retrofits by agent.

        Parameters
        ----------
        energy_prices: pd.DataFrame
            index are heating energy and columns are years.
        cost_invest: dict, optional
            keys are transition (cost_invest['Energy performance']) and item are pd.DataFrame
        cost_intangible: dict, optional
            keys are transition (cost_intangible['Energy performance']) and item are pd.DataFrame
        consumption: {'conventional', 'actual'}, default 'conventional
        transition: list, default ['Energy performance']
            define transition. Transition can be defined as attributes transition, energy transition, or attributes-energy transition.
        subsidies: list, optional
            list of subsidies object
        nu: float or int, default 8.0

        Returns
        -------
        pd.Series
            Present value is structured by agent.
        """

        if transition is None:
            transition = ['Energy performance']

        ms_final, lcc_final = self.to_market_share(energy_prices,
                                                   transition=transition,
                                                   consumption=consumption,
                                                   cost_invest=cost_invest,
                                                   cost_intangible=cost_intangible,
                                                   subsidies=subsidies,
                                                   nu=nu)

        pv = (ms_final * lcc_final).dropna(axis=0, how='all').sum(axis=1)
        self.pv[tuple(transition)][self.year] = pv
        return pv

    def to_npv(self, energy_prices, transition=None, consumption='conventional', cost_invest=None, cost_intangible=None,
               subsidies=None, nu=8.0):
        """Calculate net present value of home-energy retrofits by agent.

        Parameters
        ----------
        energy_prices: pd.DataFrame
            index are heating energy and columns are years.
        cost_invest: dict, optional
            keys are transition (cost_invest['Energy performance']) and item are pd.DataFrame
        cost_intangible: dict, optional
            keys are transition (cost_intangible['Energy performance']) and item are pd.DataFrame
        consumption: {'conventional', 'actual'}, default 'conventional
        transition: list, default ['Energy performance']
            define transition. Transition can be defined as attributes transition, energy transition, or attributes-energy transition.
        subsidies: list, optional
            list of subsidies object
        nu: float or int, default 8.0

        Returns
        -------
        pd.Series
            Net present value is structured by agent.
        """
        if transition is None:
            transition = ['Energy performance']

        energy_lcc_seg = self.to_energy_lcc(energy_prices, transition=transition, consumption=consumption)
        pv = self.to_pv(energy_prices,
                        transition=transition,
                        consumption=consumption,
                        cost_invest=cost_invest,
                        cost_intangible=cost_intangible,
                        subsidies=subsidies, nu=nu)

        pv.sort_index(inplace=True)
        energy_lcc_seg.sort_index(inplace=True)
        # assert energy_lcc_seg.index.equals(pv_seg.index), 'Index should match'

        npv = (energy_lcc_seg - pv).dropna()
        self.npv[tuple(transition)][self.year] = npv
        return npv

    def _calibration_market_share(self, energy_prices, market_share_ini, cost_invest=None,
                                  consumption='conventional', subsidies=None, option=2):
        """Calculate intangible costs by calibrating market_share.

        Solving the equations system:
        Intangible costs are calibrated so that the life-cycle cost model, fed with the investment costs, matches the
        observed market shares in the initial year. However, infinite sets of intangible costs can be found to reproduce
        the relative market shares. In other words the equation system is under-determined and must be solved with an
        additional condition. Accordingly, intangible costs are calibrated by incorporating one equation controlling for
        λ, the ratio of average intangible costs over average life cycle costs. Parameter λ is set as the minimum value
        for which the system can be solved from 0.15 to 0.6 (exogenous limits). The equation system is independent per
        agent and should solve for a given lambda per agent. Another constraint imposes that the agents sharing the same
        initial performance state find the same lambda.

        From aggregated observed market share to agent-based:
        Intangible costs could be calculated for every agent as LCC depends on every attributes. However, the
        observed market share represents the average values over Energy performance initial and final dimensions.
        Function defined various way to solve this representation issue:


        - Option 0: Each agent got its own intangible costs to match the observed market share. Regardless of the technical and
          behavioral characteristics of the agents, their market share will be identical as long as they start and end at
          the same level of energy performance.
        - Option 1: Intangible cost is aggregated to reflect the average initial market share. Each agents group share the same
          intangible costs to match the observed market share. It shows diversity among agents initial market share to
          reflect different technical and behavioral characteristics. Concretely the aggregation of the data is done by a
          weighted average of the distribution of the households.
        - Option 2: Intangible cost are aggregated over the Heating energy dimension thanks a representative agent of all heating
          energy. This solution has been implemented to match previous version of Res-IRF.
        - Not implemented: another way would be to calibrate the intangible cost by calculating life-cycle cost for
          a representative agent.


        Parameters
        ----------
        energy_prices: pd.DataFrame
        market_share_ini: pd.DataFrame
            Observed market share to match during calibration_year.
        cost_invest: dict, optional
        consumption: {'conventional', 'actual'}, default 'conventional'
        subsidies: list, optional
            subsidies to consider in the market share
        option: {0, 1, 2}, default 0
            0: intangible cost is segmented by agent
                Each agent got its own intangible costs to match the observed market share.
            1: intangible cost is aggregated based on initial market share attributes
                Each agents group share intangible costs to match the observed market share.
                This option allows some diversity in the calculated market share between member of each group.
            2: intangible cost is aggregated on heating energy level
                Used in Res-IRF 3.0

        Returns
        -------
        pd.DataFrame
            Intangible cost
        pd.DataFrame
            Market share calculated and observed
        """

        # following code is written to copy/paste Scilab version
        energy_consumption = self.to_area() * self.stock * self.to_consumption_conventional()
        energy_consumption = energy_consumption.groupby(['Occupancy status', 'Housing type', 'Heating energy', 'Energy performance']).sum()
        share_energy = energy_consumption.groupby('Heating energy').sum() / energy_consumption.sum()
        energy_price = energy_prices.loc[:, self.calibration_year]

        consumption_conventional = self.attributes2consumption
        average_energy_cost = (
                    consumption_conventional * reindex_mi(energy_price, consumption_conventional.index) * reindex_mi(
                share_energy, consumption_conventional.index)).groupby('Energy performance').sum()

        discount_factor = self.to_discount_factor(scenario_horizon=('Energy performance', )).droplevel('Heating energy')
        discount_factor = discount_factor[~discount_factor.index.duplicated(keep='first')]
        energy_lcc = discount_factor * reindex_mi(average_energy_cost, discount_factor.index)
        energy_lcc = energy_lcc.unstack('Energy performance')
        energy_lcc.columns.set_names('Energy performance final', inplace=True)

        lcc_final = self.to_lcc_final(energy_prices, consumption=consumption, cost_invest=cost_invest,
                                      transition=['Energy performance'], subsidies=subsidies, energy_lcc=energy_lcc)

        # remove income class as MultiIndex and drop duplicated indexes
        lcc_final.reset_index(level='Income class', drop=True, inplace=True)
        lcc_final = lcc_final[~lcc_final.index.duplicated(keep='first')]

        # remove idx when certificate = 'A' (no transition) and certificate = 'B' (intangible_cost = 0)
        lcc_useful = remove_rows(lcc_final, 'Energy performance', 'A')
        lcc_useful = remove_rows(lcc_useful, 'Energy performance', 'B')

        market_share_temp = HousingStock.lcc2market_share(lcc_useful)
        market_share_objective = reindex_mi(market_share_ini, market_share_temp.index)
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
            root, info_dict, ier, message = fsolve(func, x0, args=(lcc_np, ms_obj, factor), full_output=True, xtol=1e-10)

            if ier == 1:
                return ier, root

            else:
                return ier, None

        lambda_min = 0.15
        lambda_max = 0.6
        step = 0.01

        idx_list, lambda_list, intangible_list = [], [], []
        num_certificate = list(lcc_final.index.names).index('Energy performance')

        temp = lcc_final.droplevel('Energy performance')
        temp = temp.index[~temp.index.duplicated()]

        ep_list = [ep for ep in self.attributes_values['Energy performance'] if ep not in ['A', 'B']]

        for index in temp:
            for lambda_current in range(int(lambda_min * 100), int(lambda_max * 100), int(step * 100)):
                lambda_current = lambda_current / 100

                validation = True
                idx_sublist, lambda_sublist, intangible_sublist = [], [], []
                for ep in ep_list:
                    idx = (index[0], index[1], index[2], ep, index[3])
                    num_ini = self.attributes_values['Energy performance'].index(idx[num_certificate])
                    certificate_final = self.attributes_values['Energy performance'][num_ini + 1:]
                    # intangible cost would be for index = idx, and certificate_final.

                    lcc_row_np = lcc_final.loc[idx, certificate_final].to_numpy()
                    ms_obj_np = ms_obj_approx.loc[idx, certificate_final].to_numpy()
                    ier, root = solve_intangible_cost(lambda_current, lcc_row_np, ms_obj_np)

                    if ier == 1:
                        lambda_sublist += [lambda_current]
                        idx_sublist += [idx]
                        intangible_sublist += [pd.Series(root ** 2, index=certificate_final)]
                    else:
                        validation = False
                        break

                if validation is True:
                    lambda_list += lambda_sublist
                    idx_list += idx_sublist
                    intangible_list += intangible_sublist
                    # func(root, lcc_row_np, ms_obj_np, lambda_current)
                    break

        intangible_cost = pd.concat(intangible_list, axis=1).T
        intangible_cost.index = pd.MultiIndex.from_tuples(idx_list)
        intangible_cost.index.names = lcc_final.index.names
        intangible_cost.columns.names = lcc_final.columns.names

        assert len(lcc_useful.index) == len(idx_list), "Calibration didn't work for all segments"

        # adding Income class that have been removed first
        intangible_cost = add_level(intangible_cost,
                                    pd.Index(self.attributes_values['Income class'], name='Income class'),
                                    axis=0)

        market_share_calibration = None
        if option == 1 or option == 2:
            if option == 1:
                levels = list(market_share_ini.index.names)
            elif option == 2:
                levels = [i for i in self.stock.index.names if i != 'Heating energy']
            else:
                raise ValueError('option can only be 0, 1 or 2')

            weight = val2share(self.stock, levels, option='column')
            ic_temp = intangible_cost.unstack(weight.columns.names)
            weight_re = reindex_mi(weight, ic_temp.columns, axis=1)
            ic_temp = ic_temp.reorder_levels(weight_re.index.names)

            # when adding Income class level some segments has been added that was not in self.stock
            weight_re = weight_re.reindex(weight_re.index.union(ic_temp.index))
            idx = weight_re.isna().all(axis=1)
            weight_re[idx] = 1 / weight_re.shape[1]

            weight_re = weight_re.loc[ic_temp.index, :]
            ic_weighted = (weight_re * ic_temp).fillna(0).groupby('Energy performance final', axis=1).sum()
            intangible_cost = reindex_mi(ic_weighted, intangible_cost.index)

            cost_intangible = dict()
            cost_intangible['Energy performance'] = intangible_cost
            market_share, _ = self.to_market_share(energy_prices,
                                                   transition=['Energy performance'],
                                                   consumption=consumption,
                                                   cost_invest=cost_invest,
                                                   cost_intangible=cost_intangible,
                                                   subsidies=subsidies)

            market_share = market_share.unstack(weight.columns.names)
            market_share_weighted = (market_share * weight_re).fillna(0).groupby('Energy performance final', axis=1).sum()
            market_share_weighted.name = 'Market share calculated after calibration'
            if False:
                market_share_calibration = pd.concat((market_share_ini.stack(), market_share_weighted.stack()), axis=1)

        return intangible_cost, market_share_calibration

    def calibration_market_share(self, energy_prices, market_share_ini, cost_invest=None,
                                 consumption='conventional', subsidies=None):
        """Calculate intangible costs by calibrating market_share.

        Intangible costs are calculated for
        Calculate intangible cost to match observed market_share with initial LCC.
        LCC represents a representative agent (mean agent).
        Normalize the minimum intangible cost to 0.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        market_share_ini: pd.DataFrame
            Observed market share to match during calibration_year.
        cost_invest: dict, optional
        consumption: {'conventional', 'actual'}, default 'conventional'
        subsidies: list, optional
            subsidies to consider in the market share

        Returns
        -------
        pd.DataFrame
            Intangible cost
        pd.DataFrame
            Market share calculated and observed
        """

        lcc_final = self.to_lcc_final(energy_prices, consumption=consumption, cost_invest=cost_invest,
                                      transition=['Energy performance'], subsidies=subsidies)

        levels = market_share_ini.index.names
        weight = val2share(self.stock, levels, option='column')

        lcc_agent_mean = lcc_final.unstack(weight.columns.names)
        weight_re = reindex_mi(weight, lcc_agent_mean.columns, axis=1)
        lcc_agent_mean = (weight_re * lcc_agent_mean).fillna(0).stack('Energy performance final').sum(axis=1)

        # remove idx when certificate = 'A' (no transition) and certificate = 'B' (intangible_cost = 0)
        lcc_agent_mean = remove_rows(lcc_agent_mean, 'Energy performance', 'A')
        lcc_agent_mean = remove_rows(lcc_agent_mean, 'Energy performance', 'B')
        lcc_agent_mean = remove_rows(lcc_agent_mean, 'Energy performance final', 'G')
        lcc_agent_mean = lcc_agent_mean.unstack('Energy performance final')
        lcc_agent_mean.replace(0, float('nan'), inplace=True)
        _lcc_agent_mean = lcc_agent_mean.copy()

        market_share_temp = HousingStock.lcc2market_share(lcc_agent_mean)
        market_share_objective = reindex_mi(market_share_ini, market_share_temp.index)
        market_share_objective = market_share_objective.reindex(market_share_temp.columns, axis=1)

        for _, ds in market_share_objective.iteritems():
            ds[ds == 0] = 0.001
            ds = ds / ds.sum()

        def func(intangible_cost, lcc, ms, insert, nu=7):
            """Functions of intangible_cost that are equal to 0.

            Parameters
            ----------
            intangible_cost: np.array
            lcc: np.array
                Calculated market-share without intangible cost.
            ms: np.array
                Observed market-share.
            nu: int, default 8
                Heterogenity parameter.
            insert: int
                Position to assign 0 (normalization location).
            """
            intangible = np.insert(intangible_cost, insert, 0)
            market_share = (lcc + intangible) ** -nu / np.sum(
                (lcc + intangible) ** -nu)
            result = market_share[1:] - ms[1:]
            return result

        cost_intangible = pd.DataFrame(index=market_share_objective.index, columns=market_share_objective.columns,
                                       dtype='float64')
        for performance in lcc_agent_mean.index:
            lcc = lcc_agent_mean.loc[performance, :].dropna().sort_index(ascending=False)
            ms = market_share_objective.loc[performance, :].dropna().sort_index(ascending=False)
            x0 = (lcc[1:] * 0).to_numpy()
            location_normalize = 0
            root, info_dict, ier, message = fsolve(func, x0, args=(lcc.to_numpy(), ms.to_numpy(),
                                                                   location_normalize, ), full_output=True)
            if ier == 1:
                intangible = np.insert(root, location_normalize, 0)
                cost_intangible.loc[performance, lcc.index] = intangible
                if cost_intangible.loc[performance, :].min() < 0:
                    idx_normalize = cost_intangible.loc[performance, :].idxmin()
                    location_normalize = lcc.index.tolist().index(idx_normalize)
                    if location_normalize > 0:
                        root, info_dict, ier, message = fsolve(func, x0, args=(lcc.to_numpy(), ms.to_numpy(),
                                                                               location_normalize,), full_output=True)
                        intangible = np.insert(root, location_normalize, 0)
                        cost_intangible.loc[performance, lcc.index] = intangible

            else:
                raise ValueError

        # checking
        market_share, lcc_final = self.to_market_share(energy_prices,
                                                       transition=['Energy performance'],
                                                       consumption=consumption,
                                                       cost_invest=cost_invest,
                                                       cost_intangible={'Energy performance': cost_intangible},
                                                       subsidies=subsidies)
        lcc_agent_mean = lcc_final.unstack(weight.columns.names)
        lcc_agent_mean = (weight_re * lcc_agent_mean).fillna(0).stack('Energy performance final').sum(axis=1)
        lcc_agent_mean = remove_rows(lcc_agent_mean, 'Energy performance', 'A')
        lcc_agent_mean = remove_rows(lcc_agent_mean, 'Energy performance', 'B')
        lcc_agent_mean = remove_rows(lcc_agent_mean, 'Energy performance final', 'G')
        lcc_agent_mean = lcc_agent_mean.unstack('Energy performance final')
        lcc_agent_mean.replace(0, float('nan'), inplace=True)
        market_share_agent_mean = HousingStock.lcc2market_share(lcc_agent_mean)

        market_share_ = market_share.unstack(weight.columns.names)
        market_share_mean_agents = (weight_re * market_share_).fillna(0).stack('Energy performance final').sum(axis=1)
        market_share_mean_agents = remove_rows(market_share_mean_agents, 'Energy performance', 'A')
        market_share_mean_agents = remove_rows(market_share_mean_agents, 'Energy performance', 'B')
        market_share_mean_agents = remove_rows(market_share_mean_agents, 'Energy performance final', 'G')
        market_share_mean_agents = market_share_mean_agents.unstack('Energy performance final')

        market_share_dict = dict()
        for type in ['Single-family', 'Multi-family']:
            market_share_type = market_share[market_share.index.get_level_values('Housing type') == type]
            market_share_type = market_share_type.unstack(weight.columns.names)
            stock_single = self.stock[self.stock.index.get_level_values('Housing type') == type]
            weight_single = val2share(stock_single, levels, option='column')
            weight_single_re = reindex_mi(weight_single, market_share_type.columns, axis=1)
            market_share_type = (weight_single_re * market_share_type).fillna(0).stack('Energy performance final').sum(axis=1)
            market_share_type = remove_rows(market_share_type, 'Energy performance', 'A')
            market_share_type = remove_rows(market_share_type, 'Energy performance', 'B')
            market_share_type = remove_rows(market_share_type, 'Energy performance final', 'G')
            market_share_dict[type] = market_share_type.unstack('Energy performance final')

        return cost_intangible, {'Objective': market_share_objective, 'Mean agents': market_share_mean_agents,
                                 'Agent mean': market_share_agent_mean, 'LCC agent mean': _lcc_agent_mean,
                                 'Single-family': market_share_dict['Single-family'],
                                 'Multi-family': market_share_dict['Multi-family']}

    def to_io_share_seg(self):
        """Calculate attributes share by income class owner.

        Returns
        -------
        pd.Series
        """
        levels = [lvl for lvl in self.stock.index.names if lvl not in ['Income class owner', 'Energy performance']]
        temp = val2share(self.stock, levels, option='column')
        io_share_seg = temp.groupby('Income class owner', axis=1).sum()
        return io_share_seg

    @staticmethod
    def lbd(knowledge, learning_rate):
        return knowledge ** (np.log(1 + learning_rate) / np.log(2))

    @staticmethod
    def learning_by_doing(knowledge, cost, learning_rate, cost_lim=None):
        """Decrease capital cost after considering learning-by-doing effect.

        Investment costs decrease exponentially with the cumulative sum of operations so as to capture
        a “learning-by-doing” process.
        The rate of cost reduction is set at 15% in new construction and 10% in renovation for a doubling of production.

        Parameters
        ----------
        knowledge: pd.Series
            knowledge indexes match cost columns to reach final state after transition
        cost: pd.DataFrame
        learning_rate: float
        cost_lim: optional, pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        lbd = HousingStock.lbd(knowledge, learning_rate)
        if cost_lim is not None:
            lbd = lbd.reorder_levels(cost.columns.names)
            # cost_lim = cost_lim.unstack('Energy performance final')
            level = 'Heating energy final'
            if level not in cost_lim.columns.names:
                indexes = lbd.index.get_level_values(level).unique()
                temp = add_level(cost_lim.copy(), indexes, axis=1)
            else:
                temp = cost_lim.copy()
            return lbd * cost + (1 - lbd) * temp
        else:
            idx_union = lbd.index.union(cost.T.index)
            return lbd.reindex(idx_union) * cost.T.reindex(idx_union).T

    @staticmethod
    def information_rate(knowledge, learning_rate, share):
        """Returns information rate that capture peer effects and knowledge diffusion.

        More info_rate is high, more intangible_cost are low.
        Intangible renovation costs decrease according to a logistic curve with the same cumulative
        production so as to capture peer effects and knowledge diffusion.
        intangible_cost[yr] = intangible_cost[calibration_year] * info_rate with info rate [1-info_rate_max ; 1]
        This function calibrate a logistic function, so rate of decrease is set at 25% for a doubling of cumulative
        production.

        Parameters
        ----------
        knowledge : pd.Series
            Knowledge indexes match cost columns to reach final state after transition.
        share : float
            Share of non-fixed costs.
        learning_rate : float
            Rate of decrease of non-fixed share for a doubling of cumulative production.

        Returns
        -------
        pd.Series
        """

        def equations(sh=share, alpha=learning_rate):
            A = np.array([[1, 1], [1, 2]])
            b = np.array([np.log(1/sh-1), np.log(1/(sh*(1-alpha))-1)])
            param = np.linalg.solve(A, -b)
            a = np.exp(- param[0])
            r = param[1]
            return a, r

        a, r = equations(sh=share, alpha=learning_rate)
        return 1 / (1 + a * np.exp(- r * knowledge)) + 1 - share

    @staticmethod
    def information_acceleration(knowledge, cost_intangible, share, learning_rate):
        """Decrease intangible cost to capture peer effects and knowledge diffusion.

        Intangible renovation costs decrease according to a logistic curve with the same cumulative production so as
        to capture peer effects and knowledge diffusion.
        The rate of decrease (learning_rate) is set at 25% of non-fixed share for a doubling of cumulative production.

        Res-IRF splits intangible costs into a fixed share `share` mimicking hidden costs, and a variable share
        1−`share` mimicking adoption externalities. Note that the value of `share` is fraught with uncertainty, due to
        the upstream problem of empirically separating the different types of barriers.

        Parameters
        ----------
        knowledge: pd.Series
            Knowledge indexes should match cost columns.
        cost_intangible: pd.DataFrame
        share: float
            Share of fixed intangible costs.
        learning_rate: float
            Rate of decrease of non-fixed share for a doubling of cumulative production.

        Returns
        -------
        pd.DataFrame
            cost_intangible
        """
        info_rate = HousingStock.information_rate(knowledge, learning_rate, share)

        temp = cost_intangible.T.copy()
        if isinstance(temp.index, pd.MultiIndex):
            info_rate = info_rate.reorder_levels(temp.index.names)
        cost_intangible = info_rate.loc[temp.index] * temp.T
        return cost_intangible

    def ini_energy_cash_flows(self, energy_price):
        """Initialize exogenous variable that doesn't depend on dynamic to fasten the script.

        For instance budget_share only depends on energy_price, and income that are exogenous variables.
        So does, heating_intensity and consumption_actual.

        List of attribute initialized by launching self.to_consumption_actual(energy_price):

        - buildings.area: pd.Series (doesn't depend on time)
        - buildings.budget_share: pd.DataFrame (depends on energy_price and income so depend on time)
        - buildings.heating_intensity: pd.DataFrame (depends on energy_price and income so depend on time)
        - buildings.consumption_conventional: pd.Series (doesn't depend on time)
        - buildings.consumption_actual: pd.DataFrame (depends on energy_price and income so depend on time)

        - buildings.energy_cash_flows['conventional']: pd.DataFrame
        - buildings.energy_cash_flows_disc['conventional']: pd.DataFrame
        - buildings.energy_cash_flows['actual']: pd.DataFrame
        - buildings.energy_cash_flows_disc['actual']: pd.DataFrame

        Parameters
        ----------
        energy_price: pd.DataFrame
        """

        # initialized (area, budget_share, heating_intensity, consumption_conventional, consumption_actual) attributes
        self.to_consumption_actual(energy_price)

        self.energy_cash_flows = dict()
        self.energy_cash_flows_disc = dict()
        self.energy_cash_flows['actual'] = HousingStock.mul_consumption(self.consumption_actual,
                                                                        energy_price)
        self.energy_cash_flows['conventional'] = HousingStock.mul_consumption(
            self.consumption_conventional,
            energy_price)
        self.energy_cash_flows_disc['actual'] = HousingStock.to_discounted(
            self.energy_cash_flows['actual'], self.attributes2discount)
        self.energy_cash_flows_disc['conventional'] = HousingStock.to_discounted(
            self.energy_cash_flows['conventional'], self.attributes2discount)

    @staticmethod
    def add_tenant_income(building_stock, tenant_income):
        """Add Income class level to stock using information of tenant income.

        Homeowners and social-housing tenant and owner income class are the same.

        Parameters
        ----------
        building_stock: pd.Series
        tenant_income: pd.Series

        Returns
        -------
        pd.Series
        """
        # building stock without landlords
        bs = building_stock.copy()
        building_stock_wolandlords = remove_rows(bs, 'Occupancy status', 'Landlords')
        temp = pd.Series(building_stock_wolandlords.index.get_level_values('Income class owner'))
        temp.index = building_stock_wolandlords.index

        building_stock_wolandlords = pd.concat((building_stock_wolandlords, temp), axis=1)
        building_stock_wolandlords = building_stock_wolandlords.set_index('Income class owner', drop=True,
                                                                          append=True).iloc[:, 0]

        # landlords
        building_stock_landlords = building_stock.xs('Landlords', level='Occupancy status', drop_level=False)
        tenant_income_re = reindex_mi(tenant_income, building_stock_landlords.index)
        tenant_income_re.columns.names = ['Income class']
        building_stock_landlords = (building_stock_landlords * tenant_income_re.T).T.stack()

        # concatenate
        building_stock = pd.concat((building_stock_landlords, building_stock_wolandlords), axis=0)
        building_stock = building_stock.reorder_levels(
            ['Housing type', 'Occupancy status', 'Income class', 'Heating energy', 'Energy performance',
             'Income class owner'])

        return building_stock


class HousingStockRenovated(HousingStock):
    """Class that represents an existing buildings stock that can (i) renovate buildings, (ii) demolition buildings.

    Some stocks imply change for other stock: stock_master.
    Stock_master should be property as the setter methods need to change all dependencies stock: stock_slave.
    As they need to be initialize there is a private attribute in the init.
    Example:
         A modification of stock will change stock_mobile_seg. stock_mobile_seg cannot change directly.

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
        _stock_mobile & _stock_mobile_dict
    """

    def __init__(self, stock, attributes_values, year=2018,
                 residual_rate=0.0, demolition_rate=0.0,
                 rate_renovation_ini=None, learning_year=None,
                 npv_min=None, rate_max=None, rate_min=None,
                 attributes2area=None, attributes2horizon=None, attributes2discount=None, attributes2income=None,
                 attributes2consumption=None, kwh_cumac_transition=None, ms_switch_fuel_ini=None):

        super().__init__(stock, attributes_values, year,
                         attributes2area=attributes2area,
                         attributes2horizon=attributes2horizon,
                         attributes2discount=attributes2discount,
                         attributes2income=attributes2income,
                         attributes2consumption=attributes2consumption,
                         kwh_cumac_transition=kwh_cumac_transition)

        self.residual_rate = residual_rate
        self._demolition_rate = demolition_rate

        # slave stock of stock property
        self._stock_residual = stock * residual_rate
        self._stock_mobile = stock - self._stock_residual
        self._stock_mobile_dict = {year: self._stock_mobile}
        self._stock_area = self.to_stock_area()

        # initializing knowledge
        flow_area_renovated_seg = self.flow_area_renovated_seg_ini(rate_renovation_ini, learning_year)
        self._flow_knowledge_ep = self.to_flow_knowledge(flow_area_renovated_seg)
        self._stock_knowledge_ep = self._flow_knowledge_ep
        self._stock_knowledge_ep_dict = {year: self._stock_knowledge_ep}
        self._knowledge = self._stock_knowledge_ep / self._stock_knowledge_ep
        self._knowledge_dict = {year: self._knowledge}

        # share of decision-maker in the total stock
        self._dm_share_tot = stock.groupby(['Occupancy status', 'Housing type']).sum() / stock.sum()

        # calibration
        self.rate_renovation_ini = rate_renovation_ini
        self.rho = None
        self.npv_intangible = pd.Series()
        self._npv_min = npv_min
        self._rate_max = rate_max
        self._rate_min = rate_min

        # attribute to keep information bu un-necessary for the endogeneous
        self.flow_demolition_dict = {}
        self.flow_remained_dict = {}
        self.flow_renovation_label_energy_dict = {}
        self.flow_renovation_label_dict = {}
        self.renovation_rate_dm = {}

        self.flow_renovation_obligation = {}
        self.renovation_rate_dict = {}
        self.ms_switch_fuel_ini = ms_switch_fuel_ini

    @property
    def stock(self):
        return self._stock

    @stock.setter
    def stock(self, new_stock):
        """
        Master stock that implement modification for stock slave.
        """
        self._segments = new_stock.index

        self._stock = new_stock
        self._stock_dict[self.year] = new_stock
        self._stock_mobile = new_stock - self._stock_residual
        self._stock_mobile[self._stock_mobile < 0] = 0
        self._stock_mobile_dict[self.year] = self._stock_mobile
        self._stock_area = self.to_stock_area()

        self._wo_ep_share_tot = new_stock.groupby(
            [i for i in new_stock.index.names if i != 'Energy performance']).sum() / new_stock.sum()

    @property
    def stock_mobile(self):
        return self._stock_mobile

    @property
    def flow_knowledge_ep(self):
        return self._flow_knowledge_ep

    @flow_knowledge_ep.setter
    def flow_knowledge_ep(self, new_flow_knowledge_ep):
        self._flow_knowledge_ep = new_flow_knowledge_ep
        self._stock_knowledge_ep = self._stock_knowledge_ep + new_flow_knowledge_ep
        self._stock_knowledge_ep_dict[self.year] = self._stock_knowledge_ep
        self._knowledge = self._stock_knowledge_ep / self._stock_knowledge_ep_dict[self._calibration_year]
        self._knowledge_dict[self.year] = self._knowledge

    @property
    def stock_knowledge_ep_dict(self):
        return self._stock_knowledge_ep_dict

    @property
    def knowledge_dict(self):
        return self._knowledge_dict

    @property
    def knowledge(self):
        return self._knowledge

    def flow_area_renovated_seg_ini(self, rate_renovation_ini, learning_year):
        """
        Initialize flow area renovated.

        Flow area renovation is defined as:
         renovation rate (2.7%/yr) x number of learning years (10 yrs) x renovated area (m2).
        """
        renovation_rate = reindex_mi(rate_renovation_ini, self._stock_area.index,
                                     rate_renovation_ini.index.names)

        return renovation_rate * self._stock_area * learning_year

    @property
    def stock_area(self):
        return self._stock_area

    @staticmethod
    def _renovate_rate_func(npv, rho, npv_min, rate_max, rate_min):
        """Calculate renovation rate for indexed pd.Series rho and indexed pd.Series npv.
        Version 3.0.

        Parameters
        ----------
        npv: pd.Series
            Indexes are the first elements of NPV and last value is the actual NPV value.
        rho: pd.Series
            Rho MultiIndex pd.Series containing rho values and indexed by stock attributes.
        npv_min: float
        rate_max: float
        rate_min: float

        Returns
        -------
        float
        """
        if isinstance(rho, pd.Series):
            rho_f = rho.loc[tuple(npv.iloc[:-1].tolist())]
        else:
            rho_f = rho

        if np.isnan(rho_f):
            return float('nan')
        else:
            return HousingStockRenovated._renovation_rate(npv.loc[0], rho_f, npv_min, rate_max, rate_min)

    @staticmethod
    def renovation_rate(npv, rho, version='version_4', constant=890.818723, rate_max=0.4,
                        rate_min=0.2, npv_min=-1000):
        """Renovation rate function.

        Parameters
        ----------
        npv: pd.Series
        rho: float
        constant: Alternative-specific constant

        Returns
        -------
        pd.Series
        """

        if version == 'version_4':
            return 1 / (1 + np.exp(- rho * (npv - constant)))
        elif version == 'version_3':
            return rate_max / (1 + (rate_max / rate_min - 1) * np.exp(- rho * (npv - npv_min)))
        else:
            raise ValueError

    def to_renovation_rate(self, energy_prices, transition=None, consumption='conventional', cost_invest=None,
                           cost_intangible=None, subsidies=None, rho=None, version='version_4'):

        """Routine calculating renovation rate from segments for a particular yr.

        Cost (energy, investment) & rho parameter are also required.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        transition: list
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        subsidies: list, optional
        rho: pd.Series, optional
        version: str, default 'version_4"
        """
        if transition is None:
            transition = ['Energy performance']

        if rho is None:
            rho = self.rho

        npv = self.to_npv(energy_prices,
                          transition=transition,
                          consumption=consumption,
                          cost_invest=cost_invest,
                          cost_intangible=cost_intangible,
                          subsidies=subsidies)

        renovation_rate = HousingStockRenovated.renovation_rate(npv, rho, version=version,
                                                                rate_max=self._rate_max, rate_min=self._rate_min,
                                                                npv_min=self._npv_min)

        self.renovation_rate_dict[self.year] = renovation_rate
        return renovation_rate

    def flow_renovation_ep(self, energy_prices, consumption='conventional', cost_invest=None, cost_intangible=None,
                           subsidies=None, renovation_obligation=None, mutation=0.0, rotation=0.0, version='version_4'):
        """Calculate flow renovation by energy performance final.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        subsidies: dict, optional
        renovation_obligation: RenovationObligation, optional
        mutation: pd.Series or float, default 0.0
        rotation: pd.Series or float, default 0.0

        Returns
        -------
        pd.DataFrame
        """

        transition = ['Energy performance']

        renovation_rate = self.to_renovation_rate(energy_prices,
                                                  transition=transition,
                                                  consumption=consumption,
                                                  cost_invest=cost_invest,
                                                  cost_intangible=cost_intangible,
                                                  subsidies=subsidies,
                                                  version=version)

        stock = self.stock_mobile.copy()
        flow_renovation = renovation_rate * stock
        stock_obligation = stock - flow_renovation

        flow_renovation_obligation = 0
        if renovation_obligation is not None:
            flow_renovation_obligation = self.to_flow_obligation(renovation_obligation, stock=stock_obligation,
                                                                 mutation=mutation, rotation=rotation)
            # stock = stock - flow_renovation_obligation

            market_share_obligation_ep = self.to_market_share(energy_prices,
                                                              transition=transition,
                                                              consumption=consumption,
                                                              cost_invest=cost_invest,
                                                              cost_intangible=cost_intangible,
                                                              subsidies=subsidies,
                                                              final=renovation_obligation.final)[0]

            flow_renovation_obligation_ep = (flow_renovation_obligation * market_share_obligation_ep.T).T

        # indicators
        _flow_renovation = flow_renovation + flow_renovation_obligation
        renovation_rate_aggr = _flow_renovation.sum() / stock.sum()
        renovation_rate_dm = _flow_renovation.groupby(['Occupancy status', 'Housing type']).sum() / stock[
            stock.index.get_level_values('Energy performance') != 'A'].groupby(
            ['Occupancy status', 'Housing type']).sum()
        self.renovation_rate_dm[self.year] = renovation_rate_dm

        if self.year in self.market_share[tuple(transition)]:
            market_share_ep = self.market_share[tuple(transition)][self.year]
        else:
            market_share_ep = self.to_market_share(energy_prices,
                                                   transition=transition,
                                                   consumption=consumption,
                                                   cost_invest=cost_invest,
                                                   cost_intangible=cost_intangible,
                                                   subsidies=subsidies)[0]

        flow_renovation_ep = (flow_renovation * market_share_ep.T).T

        if renovation_obligation is not None:
            flow_renovation_ep += flow_renovation_obligation_ep.reindex(flow_renovation_ep.columns, axis=1).fillna(0)


        self.flow_renovation_label_dict[self.year] = flow_renovation_ep

        return flow_renovation_ep

    def to_flow_renovation_ep(self, energy_prices, consumption='conventional', cost_invest=None, cost_intangible=None,
                              subsidies=None, renovation_obligation=None, mutation=0.0, rotation=0.0, error=1,
                              version='version_4'):
        """
        Functions only useful if a subsidy_tax is declared. Run a dichotomy to find the subsidy rate that recycle energy
        tax revenue.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        subsidies: dict, optional
        renovation_obligation: RenovationObligation, optional
        mutation: pd.Series or float, default 0.0
        rotation: pd.Series or float, default 0.0
        error: int, default 1

        Returns
        -------
        pd.DataFrame
        """

        flow_renovation_ep = self.flow_renovation_ep(energy_prices,
                                                     consumption=consumption,
                                                     cost_invest=cost_invest,
                                                     cost_intangible=cost_intangible,
                                                     subsidies=subsidies,
                                                     renovation_obligation=renovation_obligation,
                                                     mutation=mutation, rotation=rotation, version=version)

        for policy in subsidies:
            if policy.policy == 'subsidy_tax':

                policy.value = policy.value_max
                value_max = policy.value
                value_min = 0.0

                tax_revenue = policy.tax_revenue[self.year - 1]
                subsidy_expense = (flow_renovation_ep * self.subsidies_detailed_euro[('Energy performance',)][self.year][
                    '{} (euro)'.format(policy.name)]).sum().sum()

                while abs(subsidy_expense - tax_revenue) > error:
                    # function grows with policy.value

                    policy.value = (value_max + value_min) / 2

                    flow_renovation_ep = self.flow_renovation_ep(energy_prices,
                                                                 consumption=consumption,
                                                                 cost_invest=cost_invest,
                                                                 cost_intangible=cost_intangible,
                                                                 subsidies=subsidies,
                                                                 renovation_obligation=renovation_obligation,
                                                                 mutation=mutation, rotation=rotation)

                    subsidy_expense = (
                                flow_renovation_ep * self.subsidies_detailed_euro[('Energy performance',)][self.year][
                            '{} (euro)'.format(policy.name)]).sum().sum()

                    if subsidy_expense > tax_revenue:
                        value_max = policy.value
                    else:
                        value_min = policy.value

                policy.subsidy_value[self.year] = policy.value
                policy.subsidy_expense[self.year] = subsidy_expense

                print('{:,.0f}'.format(tax_revenue))
                print('{:,.0f}'.format(subsidy_expense))

        return flow_renovation_ep

    def to_flow_renovation_ep_energy(self, energy_prices, consumption='conventional', cost_invest=None,
                                     cost_intangible=None, subsidies=None, renovation_obligation=None, mutation=0.0,
                                     rotation=0.0, version='version_4'):
        """De-aggregate stock_renovation_attributes by final heating energy.

        1. Flow renovation returns number of renovation by final energy performance.
        2. Heating energy technology market-share is then calculated based on flow renovation.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        subsidies: list, optional
        renovation_obligation: RenovationObligation, optional
        mutation: pd.Series or float, default 0.0
        rotation: pd.Series or float, default 0.0

        Returns
        -------
        pd.DataFrame
        """

        """
        market_share_energy = self.to_market_share_energy(energy_prices,
                                                          cost_invest=cost_invest,
                                                          consumption=consumption,
                                                          subsidies=subsidies)
        """

        flow_renovation = self.to_flow_renovation_ep(energy_prices,
                                                     consumption=consumption,
                                                     cost_invest=cost_invest,
                                                     cost_intangible=cost_intangible,
                                                     subsidies=subsidies,
                                                     renovation_obligation=renovation_obligation,
                                                     mutation=mutation, rotation=rotation,
                                                     version=version)

        flow_renovation_temp = flow_renovation.stack()
        market_share_energy_temp = reindex_mi(self.ms_switch_fuel_ini, flow_renovation_temp.index)

        flow_renovation_label_energy = (market_share_energy_temp.T * flow_renovation_temp).T
        flow_renovation_label_energy = flow_renovation_label_energy.unstack('Energy performance final')

        self.flow_renovation_label_energy_dict[self.year] = flow_renovation_label_energy

        np.testing.assert_almost_equal(flow_renovation.sum().sum(), flow_renovation_label_energy.sum().sum(),
                                       err_msg='Market share should not erased renovation')

        return flow_renovation_label_energy

    def to_flow_remained(self, energy_prices, consumption='conventional', cost_invest=None, cost_intangible=None,
                         subsidies=None, renovation_obligation=None, mutation=0.0, rotation=0.0, version='version_4'):
        """Calculate flow_remained for each segment.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        subsidies: list, optional
        renovation_obligation: RenovationObligation, optional
        mutation: pd.Series or float, default 0.0
        rotation: pd.Series or float, default 0.0

        Returns
        -------
        flow_remained_seg, pd.Series
            positive (+) flow for buildings segment reached by the renovation (final state),
            negative (-) flow for buildings segment (initial state) that have been renovated.

        flow_area_renovation_seg, pd.Series
        """

        flow_renovation_label_energy_seg = self.to_flow_renovation_ep_energy(energy_prices,
                                                                             consumption=consumption,
                                                                             cost_invest=cost_invest,
                                                                             cost_intangible=cost_intangible,
                                                                             subsidies=subsidies,
                                                                             renovation_obligation=renovation_obligation,
                                                                             mutation=mutation, rotation=rotation,
                                                                             version=version
                                                                             )

        area_seg = reindex_mi(self.attributes2area, flow_renovation_label_energy_seg.index, self.attributes2area.index.names)
        flow_area_renovation_seg = (area_seg * flow_renovation_label_energy_seg.T).T

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

    def update_stock(self, flow_remained_seg, flow_area_renovation_seg=None):
        """Update HousingStock.

        Parameters
        ----------
        flow_remained_seg: pd.Series
        flow_area_renovation_seg: pd.Series, pd.DataFrame
        """
        # update segmented stock  considering renovation
        self.add_flow(flow_remained_seg)

        if flow_area_renovation_seg is not None:
            flow_knowledge_renovation = self.to_flow_knowledge(flow_area_renovation_seg)
            self.flow_knowledge_ep = flow_knowledge_renovation

    def _to_flow_demolition_dm(self):
        flow_demolition = self._stock.sum() * self._demolition_rate
        flow_demolition_dm = self._dm_share_tot * flow_demolition
        # flow_area_demolition_seg_dm = flow_demolition_seg_dm * self.attributes2area
        return flow_demolition_dm

    def _to_flow_demolition_seg(self):
        """ Returns stock_demolition -  segmented housing number demolition.
        Res-IRF 3.0.

        Buildings to destroy are chosen in stock_mobile.
        1. type_housing_demolition is respected to match decision-maker proportion; - type_housing_demolition_reindex
        2. income_class, income_class_owner, heating_energy match stock_remaining proportion; - type_housing_demolition_wo_performance
        3. worst energy_performance_attributes for each segment are targeted. - stock_demolition

        Returns
        -------
        pd.Series segmented
        """

        flow_demolition_dm = self.to_flow_demolition_dm()

        # stock_mobile = self._stock_mobile_dict[self.year - 1]
        stock_mobile = self.stock_mobile.copy()
        stock_mobile_ini = self._stock_mobile_dict[self._calibration_year].copy()
        segments_mobile = stock_mobile.index
        segments_mobile = segments_mobile.droplevel('Energy performance')
        segments_mobile = segments_mobile.drop_duplicates()

        def worst_certificate(seg_mobile, st_mobile):
            """Returns worst certificate for each segment with stock > 1.

            Parameters
            __________
            segments_mobile, pd.MultiIndex
            MultiIndex without Energy performance level.

            stock_mobile, pd.Series
            with Energy performance level

            Returns
            _______
            worst_cert_idx, list
            Index with the worst Energy Performance certificate value

            worst_cert_idx, dict
            Worst certificate for each segment
            """
            worst_lbl_idx = []
            worst_lbl_dict = dict()
            for seg in seg_mobile:
                for lbl in self.total_attributes_values['Energy performance']:
                    indx = (seg[0], seg[1], seg[2], seg[3], lbl, seg[4])
                    if st_mobile.loc[indx] > 1:
                        worst_lbl_idx.append(indx)
                        worst_lbl_dict[seg] = lbl
                        break
            return worst_lbl_idx, worst_lbl_dict

        worst_certificate_idx, worst_certificate_dict = worst_certificate(segments_mobile, stock_mobile)

        # we know type_demolition, then we calculate nb_housing_demolition_ini based on proportion of stock remaining
        levels = ['Occupancy status', 'Housing type']
        levels_wo_performance = [lvl for lvl in self._levels if lvl != 'Energy performance']
        stock_remaining_woperformance = stock_mobile.groupby(levels_wo_performance).sum()
        stock_remaining_woperformance = stock_remaining_woperformance[stock_remaining_woperformance > 0]
        stock_share_dm = val2share(stock_remaining_woperformance, levels, option='column')
        flow_demolition_dm_re = reindex_mi(flow_demolition_dm, stock_share_dm.index, levels)
        flow_demolition_wo_ep = (flow_demolition_dm_re * stock_share_dm.T).T
        flow_demolition_wo_ep = flow_demolition_wo_ep.stack(
            flow_demolition_wo_ep.columns.names)
        np.testing.assert_almost_equal(flow_demolition_dm.sum(), flow_demolition_wo_ep.sum(),
                                       err_msg='Not normal')

        # we don't have the information about which certificates are going to be destroyed first
        prop_stock_worst_certificate = stock_mobile.loc[worst_certificate_idx] / stock_mobile_ini.loc[
            worst_certificate_idx]

        # initialize attributes with worst_attributes
        flow_demolition_ini = reindex_mi(flow_demolition_wo_ep, prop_stock_worst_certificate.index,
                                         levels_wo_performance)
        # initialize flow_demolition_remain with worst certificate based on how much have been destroyed so far
        flow_demolition_remain = prop_stock_worst_certificate * flow_demolition_ini

        # we year with the worst attributes and we stop when nb_housing_demolition_theo == 0
        flow_demolition = pd.Series(0, index=stock_mobile.index, dtype='float64')
        for segment in segments_mobile:
            # dangerous conditions
            if segment in worst_certificate_dict.keys():
                certificate = worst_certificate_dict[segment]
            else:
                continue
            num = self.total_attributes_values['Energy performance'].index(certificate)
            idx_w_ep = (segment[0], segment[1], segment[2], segment[3], certificate, segment[4])

            while flow_demolition_remain.loc[idx_w_ep] != 0:
                # stock_demolition cannot be sup to stock_mobile and to flow_demolition_theo
                flow_demolition.loc[idx_w_ep] = min(stock_mobile.loc[idx_w_ep], flow_demolition_remain.loc[idx_w_ep])

                if certificate != 'A':
                    num += 1
                    certificate = self.total_attributes_values['Energy performance'][num]
                    certificates = [c for c in self.total_attributes_values['Energy performance'] if c > certificate]
                    idx_wo_ep = (segment[0], segment[1], segment[2], segment[3], segment[4])
                    idx_w_ep = (segment[0], segment[1], segment[2], segment[3], certificate, segment[4])
                    list_idx = [(segment[0], segment[1], segment[2], segment[3], c, segment[4]) for c in certificates]

                    # flow_demolition_remain: remaining housing that need to be destroyed for this segment
                    flow_demolition_remain[idx_w_ep] = flow_demolition_wo_ep.loc[idx_wo_ep] - flow_demolition.loc[list_idx].sum()

                else:
                    # stop while loop --> all buildings has not been destroyed (impossible case anyway)
                    flow_demolition_remain[idx_w_ep] = 0

        assert (stock_mobile - flow_demolition).min() >= 0, 'More demolition than mobile stock'

        self.flow_demolition_dict[self.year] = flow_demolition
        return flow_demolition

    def to_flow_demolition_seg(self):
        """Demolition function from (Nauleau, 2015) PhD thesis.
        0.1% each year for EPC E, 0.2% the EPC F and 0.34% for ECP G

        Returns
        -------
        Flow demolition
        """
        stock_mobile = self.stock_mobile.copy()

        share_demolition = pd.Series({'E': 0.333, 'F': 0.333, 'G': 0.333})
        flow_demolition_ep = self._stock.sum() * self._demolition_rate * share_demolition

        flow_demolition = pd.Series()
        for ep in flow_demolition_ep.index:
            stock_mobile_select = stock_mobile[
                stock_mobile.index.get_level_values('Energy performance') == ep]
            share = stock_mobile_select / stock_mobile_select.sum()
            flow_demolition_select = share * flow_demolition_ep.loc[ep]
            flow_demolition = flow_demolition.append(flow_demolition_select)

        flow_demolition.index = pd.MultiIndex.from_tuples(flow_demolition.index)
        flow_demolition.index.names = stock_mobile.index.names

        flow_demolition[(stock_mobile - flow_demolition) < 0] = stock_mobile[(stock_mobile - flow_demolition) < 0]

        return flow_demolition

    def to_flow_knowledge(self, flow_area_renovated_seg):
        """Returns knowledge renovation.

        Parameters
        ----------
        flow_area_renovated_seg: pd.DataFrame
            Data array containing renovation are for each segment and each transition.

        Returns
        -------
        """

        # initialization is based on stock initial so a pd.Series
        if isinstance(flow_area_renovated_seg, pd.Series):
            flow_area_renovated_ep = flow_area_renovated_seg.groupby(['Energy performance']).sum()
        elif isinstance(flow_area_renovated_seg, pd.DataFrame):
            flow_area_renovated_ep = flow_area_renovated_seg.groupby('Energy performance final', axis=1).sum().sum()
        else:
            raise ValueError('Flow area renovated segmented should be a DataFrame (Series for calibration year')

        # knowledge_renovation_ini depends on energy performance final
        flow_knowledge_renovation = pd.Series(dtype='float64',
                                              index=[ep for ep in self.total_attributes_values['Energy performance'] if
                                                     ep != 'G'])
        flow_area_renovated_ep.sort_index(ascending=False, inplace=True)

        flow_knowledge_renovation['F'] = flow_area_renovated_ep.iloc[0] + flow_area_renovated_ep.iloc[1]
        flow_knowledge_renovation['E'] = flow_area_renovated_ep.iloc[0] + flow_area_renovated_ep.iloc[1]
        flow_knowledge_renovation['D'] = flow_area_renovated_ep.iloc[2] + flow_area_renovated_ep.iloc[3]
        flow_knowledge_renovation['C'] = flow_area_renovated_ep.iloc[2] + flow_area_renovated_ep.iloc[3]
        flow_knowledge_renovation['B'] = flow_area_renovated_ep.iloc[4] + flow_area_renovated_ep.iloc[5]
        flow_knowledge_renovation['A'] = flow_area_renovated_ep.iloc[4] + flow_area_renovated_ep.iloc[5]
        flow_knowledge_renovation.index.set_names('Energy performance final', inplace=True)
        return flow_knowledge_renovation

    def calibration_renovation_rate(self, energy_prices, renovation_rate_ini, consumption='conventional',
                                    cost_invest=None, cost_intangible=None, subsidies=None, version='version_4'):
        """Calibration of the ρ parameter of the renovation rate function (logistic function of the NPV).

        Renovation rate of dwellings attributes led is calculated as a logistic function of the NPV.
        The logistic form captures heterogeneity in heating preference and habits,
        assuming they are normally distributed.
        Parameter ρ is calibrated, for each type of attributes. It is then aggregated or not, depending of weighted
        parameter.

        For instance, ρ was calibrated in 2012, for each type of  decision-maker and each initial certificates
        (i.e., 6x6=36 values), so that the NPVs calculated with the subsidies in effect in 2012 (see main article)
        reproduce the observed renovation rates.
        Renovation rate observed depends on (Occupancy status, Housing type)
        Option 3 from Res-IRF 3.0.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        renovation_rate_ini: pd.Series
        consumption: {'conventional', 'actual'}, default 'conventional'
        cost_invest: dict
        cost_intangible: dict
        subsidies: list
        option: int, default 0
            0: rho for each agent_mean
            4: rho for each agent_mean (based on a NPV mean)
            1: unique rho and so unique calibration function for all agents (rho, npv_min, rate_min, rate_max)
            2: rho for each individual agent
            3: calculate rho mean for group of agents (Res-IRF 3.0)


        Returns
        -------
        pd.Series
            ρ parameters by segment
        """

        # weight to calculate weighted average of variable
        levels = renovation_rate_ini.index.names
        weight = val2share(self.stock, levels, option='column')

        npv = self.to_npv(energy_prices,
                          transition=['Energy performance'],
                          consumption=consumption,
                          cost_invest=cost_invest,
                          cost_intangible=cost_intangible,
                          subsidies=subsidies)
        rho = None
        if version == 'version_3':
            # calculate agent_mean npv
            npv_mean = npv.unstack(weight.columns.names)
            npv_mean = (weight * npv_mean).fillna(0).sum(axis=1)

            rho_agent_mean = (np.log(self._rate_max / self._rate_min - 1) - np.log(
                self._rate_max / renovation_rate_ini - 1)) / (npv_mean - self._npv_min)

            # if na find assign the value to the  closest group
            rho_agent_mean = rho_agent_mean.sort_index()
            idx = rho_agent_mean.index[rho_agent_mean.isna()]
            for i in idx:
                try:
                    rho_agent_mean.loc[i] = rho_agent_mean.iloc[list(rho_agent_mean.index).index(i) + 1]
                except IndexError:
                    rho_agent_mean.loc[i] = rho_agent_mean.iloc[list(rho_agent_mean.index).index(i) - 1]

            rho = reindex_mi(rho_agent_mean, npv.index)

        if version == 'version_4':

            coefficient = pd.Series([1.35, 1.39, 1.03, 0.87, 0.8, 0.86], index=['G', 'F', 'E', 'D', 'C', 'B'],
                                    name='Energy performance')

            levels = list(renovation_rate_ini.index.names) + [coefficient.name]
            weight = val2share(self.stock, levels, option='column')
            npv_agent_mean = (weight * npv.unstack(weight.columns.names)).fillna(0).sum(axis=1)

            rho, idx = [], []
            for n, g in npv_agent_mean.groupby(renovation_rate_ini.index.names):
                idx += [n]
                renovation_rate = renovation_rate_ini.loc[n] * coefficient

                df = pd.concat((g.droplevel(renovation_rate_ini.index.names), renovation_rate), axis=1).dropna()
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(HousingStockRenovated.renovation_rate, df.iloc[:, 0], df.iloc[:, 1],
                                    p0=[0.001], bounds=(0, np.inf)
                                    )
                rho += [popt[0]]
            rho = pd.Series(rho, index=pd.MultiIndex.from_tuples(idx))
            rho.index.names = renovation_rate_ini.index.names
            rho = reindex_mi(rho, npv.index)

        else:
            ValueError('option should be in [0, 1, 2, 3, 4]')

        return rho

    @staticmethod
    def _calibration_rho(npv, renovation_rate_target, rate_max, rate_min, npv_min):
        """
        Function that returns rho in order to match the targeted renovation rate.

        Parameters
        ----------
        npv: pd.Series
            Net present value calculated.
        renovation_rate_target: pd.Series
            Targeted renovation rate.
        rate_max: float
        rate_min: float
        npv_min: float

        Returns
        -------
        pd.Series
        """
        return (np.log(rate_max / rate_min - 1) - np.log(
            rate_max / renovation_rate_target - 1)) / (npv - npv_min)

    def _calibration_renovation_rate(self, energy_prices, renovation_rate_ini, consumption='conventional',
                                     cost_invest=None, cost_intangible=None, subsidies=None):
        """Calibration of the ρ and npv_intangible parameters of the renovation rate function (logistic function of the NPV).

        Parameters
        ----------
        energy_prices: pd.DataFrame
        renovation_rate_ini: pd.Series
        consumption: {'conventional', 'actual'}, default 'conventional'
        cost_invest: dict
        cost_intangible: dict
        subsidies: list

        Returns
        -------
        float
            ρ parameter
        pd.Series
            npv_intangible by decision_maker to march observed renovation rate
        pd.DataFrame
            Concatenation of npv, renovation_rate_ini, renovation_rate_calculated.
        """

        # weight to calculate weighted average of variable
        levels = renovation_rate_ini.index.names
        weight = val2share(self.stock, levels, option='column')

        npv = self.to_npv(energy_prices,
                          transition=['Energy performance'],
                          consumption=consumption,
                          cost_invest=cost_invest,
                          cost_intangible=cost_intangible,
                          subsidies=subsidies)

        # calculate agent_mean npv
        npv_agent_mean = npv.unstack(weight.columns.names)
        npv_agent_mean = (weight * npv_agent_mean).fillna(0).sum(axis=1)

        idx = pd.IndexSlice
        # calculate rho
        ref = ('Homeowners', 'Single-family')
        rho_ini = - np.log(1/renovation_rate_ini.loc[ref] - 1) / npv_agent_mean[ref]
        npv_ref = npv.loc[idx[ref]]
        weight_ref = weight.loc[ref, :].dropna().reindex(npv_ref.index)

        def func(x, npv_np, weight_np, rate):
            return rate - ((1 / (1 + np.exp(-x * npv_np))) * weight_np).sum()

        root, info_dict, ier, message = fsolve(func, np.array(rho_ini),
                                               args=(npv_ref.to_numpy(), weight_ref.to_numpy(),
                                                     renovation_rate_ini.loc[ref]), full_output=True)
        rho = float(root)

        # checking
        renovation_rate_calculated = HousingStockRenovated.renovation_rate(npv, rho)
        renovation_rate_mean_agents = renovation_rate_calculated.unstack(weight.columns.names)
        renovation_rate_mean_agents = (weight * renovation_rate_mean_agents).fillna(0).sum(axis=1)
        # renovation_rate_ini.loc[ref] == renovation_rate_ref

        def func(x, npv_np, weight_np, rate, rho):
            return rate - ((1 / (1 + np.exp(-rho * (npv_np + x)))) * weight_np).sum()

        npv_intangible = []
        for dm in renovation_rate_ini.index:
            npv_dm = npv.loc[dm]
            weight_dm = weight.loc[dm, :].dropna().reindex(npv_dm.index)

            root, info_dict, ier, message = fsolve(func, np.array(0),
                                                   args=(npv_dm.to_numpy(), weight_dm.to_numpy(),
                                                         renovation_rate_ini.loc[dm], rho), full_output=True)
            npv_intangible += [float(root)]

        npv_intangible = pd.Series(npv_intangible, index=renovation_rate_ini.index)

        renovation_rate_calculated = HousingStockRenovated.renovation_rate(npv, rho, intangible_npv=npv_intangible)
        renovation_rate_mean_agents = renovation_rate_calculated.unstack(weight.columns.names)
        renovation_rate_mean_agents = (weight * renovation_rate_mean_agents).fillna(0).sum(axis=1)

        renovation_rate_calibration = pd.concat((npv_agent_mean, npv_intangible,
                                                 renovation_rate_ini, renovation_rate_mean_agents), axis=1).dropna()
        renovation_rate_calibration.columns = ['NPV mean calculated', 'NPV intangible', 'Renovation rate observed',
                                               'Renovation rate calculated']

        option = None

        if option == 1:
            # calculate agent_mean npv
            npv_mean = npv.unstack(weight.columns.names)
            npv_mean = (weight * npv_mean).fillna(0).sum(axis=1)
            # npv_mean = (weight * npv_mean).dropna(axis=1).dropna().sum(axis=1)

            # solution 0: calculate rho for each agent_mean
            rho_agent_mean = HousingStockRenovated.calibration_rho(npv_mean, renovation_rate_ini, self._rate_max, self._rate_min,
                                                                   self._npv_min)
            # if na find assign the value to the  closest group
            rho_agent_mean = rho_agent_mean.sort_index()
            idx = rho_agent_mean.index[rho_agent_mean.isna()]
            for i in idx:
                try:
                    rho_agent_mean.loc[i] = rho_agent_mean.iloc[list(rho_agent_mean.index).index(i) + 1]
                except IndexError:
                    rho_agent_mean.loc[i] = rho_agent_mean.iloc[list(rho_agent_mean.index).index(i) - 1]

            rho = reindex_mi(rho_agent_mean, npv.index)

            # solution 1.: calculate unique renovation function
            from scipy.optimize import curve_fit

            df = pd.concat((npv_mean, renovation_rate_ini), axis=1).dropna()
            df.columns = ['NPV mean calculated', 'Renovation rate observed']
            popt, _ = curve_fit(HousingStockRenovated.renovation_rate, df.iloc[:, 0], df.iloc[:, 1],
                                p0=[rho_agent_mean.mean(), self._npv_min, self._rate_max, self._rate_min],
                                bounds=((0, -1000, 0.1, 10**-5), (1, npv_mean.min(), 0.5, renovation_rate_ini.min())))

            rho = pd.Series(popt[0], index=npv.index)

            self._npv_min = popt[1]
            self._rate_max = popt[2]
            self._rate_min = popt[3]
            print('NPV Min: {}'.format(self._npv_min))
            print('Rate Max: {}'.format(self._rate_max))
            print('Rate Min: {}'.format(self._rate_min))

            renovation_rate_agent_mean = npv_mean.reset_index().apply(HousingStockRenovated.renovate_rate_func,
                                                                      args=[pd.Series(popt[0], index=npv_mean.index),
                                                                            self._npv_min, self._rate_max,
                                                                            self._rate_min], axis=1)
            renovation_rate_agent_mean.index = npv_mean.index
            renovation_rate_agent_mean.name = 'Renovation rate agent mean'
            df = pd.concat((df, renovation_rate_agent_mean), axis=1)

            # verification
            renovation_rate = self.to_renovation_rate(energy_prices, transition=['Energy performance'],
                                                      consumption='conventional', cost_invest=cost_invest,
                                                      cost_intangible=cost_intangible, subsidies=subsidies, rho=rho)
            renovation_rate = renovation_rate.unstack(weight.columns.names)
            renovation_rate_mean_agents = (renovation_rate * weight).fillna(0).sum(axis=1)
            renovation_rate_mean_agents.name = 'Renovation rate mean agents'
            renovation_rate_calibration = pd.concat((df, renovation_rate_mean_agents), axis=1)
            renovation_rate_calibration.to_csv('renovation_rate.csv')

        if option == 2:
            # solution 2: calculate rho for each individual agent
            renovation_rate_obj = reindex_mi(renovation_rate_ini, npv.index)
            rho = HousingStockRenovated.calibration_rho(npv, renovation_rate_obj, self._rate_max, self._rate_min,
                                                        self._npv_min)

        if option == 3:
            # solution 2: calculate rho for each individual agent
            renovation_rate_obj = reindex_mi(renovation_rate_ini, npv.index)
            rho = HousingStockRenovated.calibration_rho(npv, renovation_rate_obj, self._rate_max, self._rate_min,
                                                        self._npv_min)

            # solution 3: calculate rho_mean for group of agents
            rho_temp = rho.unstack(weight.columns.names)
            rho_weighted = (weight * rho_temp).fillna(0).sum(axis=1)
            rho = reindex_mi(rho_weighted, npv.index)
            rho.replace(to_replace=0, value=0.005, inplace=True)

        return rho, npv_intangible, renovation_rate_calibration

    def to_flow_obligation(self, renovation_obligation, stock=None, mutation=0.0, rotation=0.0):

        if stock is None:
            stock = self.stock_mobile.copy()

        if isinstance(mutation, pd.Series):
            mutation = reindex_mi(mutation, stock.index, mutation.index.names)
        if isinstance(rotation, pd.Series):
            rotation = reindex_mi(rotation, stock.index, rotation.index.names)

        mutation_stock = stock * mutation
        rotation_stock = stock * rotation

        target_stock = mutation_stock + rotation_stock
        target = renovation_obligation.targets.loc[:, self.year]
        target = reindex_mi(target, target_stock.index)
        flow_renovation_obligation = target * target_stock * renovation_obligation.participation_rate
        self.flow_renovation_obligation[self.year] = flow_renovation_obligation
        return flow_renovation_obligation.fillna(0)


class HousingStockConstructed(HousingStock):
    def __init__(self, stock, attributes_values, year, stock_needed_ts,
                 share_multi_family=None,
                 market_share_energy=None,
                 share_decision_maker=None,
                 attributes2area=None,
                 attributes2consumption=None,
                 attributes2income=None):

        super().__init__(stock, attributes_values,
                         year=year,
                         attributes2area=attributes2area,
                         attributes2consumption=attributes2consumption,
                         attributes2income=attributes2income)

        self._flow_constructed = 0
        self._flow_constructed_dict = {self.year: self._flow_constructed}

        self._flow_constructed_seg = None
        self._flow_constructed_seg_dict = {self.year: self._flow_constructed_seg}
        self._stock_constructed_seg_dict = {self.year: self._flow_constructed_seg}

        self._stock_needed_ts = stock_needed_ts
        self._stock_needed = stock_needed_ts.loc[self._calibration_year]
        # used to estimate share of housing type

        self._share_multi_family_tot_dict = share_multi_family
        self._share_multi_family_tot = self._share_multi_family_tot_dict[self._calibration_year]

        self._market_share_energy = market_share_energy
        self._share_decision_maker = share_decision_maker

        self._area_construction_dict = {self.year: self.attributes2area}

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

        self.add_flow(val)

    def to_share_housing_type(self):
        """Returns share of Housing type ('Multi-family', 'Single-family') in the new constructed housings.

        Share of multi-family in the total stock to reflect urbanization effects.
        Demolition dynamic is made endogenously and therefore construction should reflect the evolution..

        Share_multifamily[year] = Stock_multifamily[year] / Stock[year]
        Share_multifamily[year] = (Stock_multifamily[year - 1] + Flow_multifamily_construction) / Stock[year]
        Share_multifamily[year] = (Share_multifamily[year - 1] * Stock[year - 1] + Flow_multifamily_construction) / Stock[year]
        Flow_multifamily_construction = Share_multifamily[year] * Stock[year] - Share_multifamily[year - 1] * Stock[year - 1]


        Returns
        -------
        pd.Series
            index: housing type, value: share of housing type in the stock of constructed buildings.
        """
        # self._share_multi_family_tot must be updated first
        stock_need_prev = self._stock_needed_ts[self.year - 1]
        share_multi_family_prev = self._share_multi_family_tot_dict[self.year - 1]
        share_multi_family_construction = (self._stock_needed * self._share_multi_family_tot_dict[
            self.year] - stock_need_prev * share_multi_family_prev) / self.flow_constructed

        ht_share_tot_construction = pd.Series([share_multi_family_construction, 1 - share_multi_family_construction],
                                              index=['Multi-family', 'Single-family'])
        ht_share_tot_construction.index.set_names('Housing type', inplace=True)
        return ht_share_tot_construction

    def update_flow_constructed_seg(self):
        """Update HousingConstructed object flow_constructed_seg attribute.

        Parameters
        ----------
        energy_price: pd.DataFrame
                cost_intangible: pd.DataFrame, optional
        cost_intangible: dict, optional
        cost_invest:  dict, optional
        consumption: {'conventional', 'actual'}, default 'conventional'
        nu: float, default 8.0
        subsidies: list, optional
        """
        flow_constructed_housing = self.to_share_housing_type() * self.flow_constructed
        flow_constructed = (flow_constructed_housing * self._share_decision_maker.T).T
        flow_constructed = flow_constructed.stack(flow_constructed.columns.names).dropna()

        market_share_energy = reindex_mi(self._market_share_energy, flow_constructed.index)
        flow_constructed = (flow_constructed * market_share_energy.T).T.stack()
        flow_constructed = flow_constructed[flow_constructed > 0]

        flow_constructed = add_level(flow_constructed,
                                     pd.Index(self.attributes_values['Energy performance'], name='Energy performance'))
        flow_constructed = flow_constructed.reorder_levels(self.attributes_values.keys())

        self.flow_constructed_seg = flow_constructed

        return flow_constructed

    @staticmethod
    def evolution_area_construction(area_construction_prev, area_construction_ini, area_construction_max,
                                    elasticity_area, available_income_ratio):
        """Evolution of new buildings area based on total available income. Function represents growth.

        Parameters
        ----------
        area_construction_prev: pd.Series
        area_construction_ini: pd.Series
        area_construction_max: pd.Series
        elasticity_area: pd.Series
        available_income_ratio: float

        Returns
        -------
        pd.Series
        """
        area_max_construction = area_construction_max.reorder_levels(area_construction_ini.index.names)

        eps_area_new = (area_max_construction - area_construction_prev) / (
                area_max_construction - area_construction_ini)
        eps_area_new = eps_area_new.apply(lambda x: max(0, min(1, x)))
        elasticity_area_new = eps_area_new.multiply(elasticity_area)

        factor_area_new = elasticity_area_new * max(0.0, available_income_ratio - 1.0)

        area_construction = pd.concat([area_max_construction, area_construction_prev * (1 + factor_area_new)],
                                      axis=1).min(axis=1)
        return area_construction

    def update_area_construction(self, elasticity_area_new_ini, available_income_real_pop_ds, area_max_construction):
        """Every year, average area of new buildings increase with available income.

        Trend is based on elasticity area / income.
        eps_area_new decrease over time and reduce elasticity while average area converge towards area_max.
        exogenous_dict['population_total_ds']

        Parameters
        ----------
        elasticity_area_new_ini: pd.Series
        available_income_real_pop_ds: pd.Series
        area_max_construction: pd.Series
        """

        area_construction_ini = self._area_construction_dict[self._calibration_year]
        area_construction_prev = self._area_construction_dict[self.year - 1]

        available_income_real_pop_ini = available_income_real_pop_ds.loc[self._calibration_year]
        available_income_real_pop = available_income_real_pop_ds.loc[self.year]

        area_construction = HousingStockConstructed.evolution_area_construction(area_construction_prev,
                                                                                area_construction_ini,
                                                                                area_max_construction,
                                                                                elasticity_area_new_ini,
                                                                                available_income_real_pop / available_income_real_pop_ini)

        self.attributes2area = area_construction
        self._area_construction_dict[self.year] = self.attributes2area

    def _to_flow_area_constructed_ini(self, stock_area_existing):
        """Initialize construction knowledge.
        """

        if self.calibration_year >= 2012:
            stock_area_new_existing_seg = pd.concat(
                (stock_area_existing.xs('A', level='Energy performance'),
                 stock_area_existing.xs('B', level='Energy performance')), axis=0)
            flow_area_constructed_ep = pd.Series(
                [2.5 * stock_area_new_existing_seg.sum(), 2 * stock_area_new_existing_seg.sum()], index=['BBC', 'BEPOS'])

        else:

            area_ep = stock_area_existing.groupby('Energy performance').sum()

            flow_area_constructed_ep = pd.Series(0, index=self.attributes_values['Energy performance'])
            flow_area_constructed_ep['G'] = 2.5 * area_ep.loc[['G', 'F', 'E', 'D', 'C', 'B', 'A']].sum()
            flow_area_constructed_ep['F'] = 2.5 * area_ep.loc[['F', 'E', 'D', 'C', 'B', 'A']].sum()
            flow_area_constructed_ep['E'] = 2.5 * area_ep.loc[['E', 'D', 'C', 'B', 'A']].sum()
            flow_area_constructed_ep['D'] = 2.5 * area_ep.loc[['D', 'C', 'B', 'A']].sum()
            flow_area_constructed_ep['C'] = 2.5 * area_ep.loc[['D', 'C']].sum()
            flow_area_constructed_ep['B'] = 1
            flow_area_constructed_ep['A'] = 1

        flow_area_constructed_ep.index.names = ['Energy performance']
        flow_area_constructed_he_ep = add_level(flow_area_constructed_ep,
                                                pd.Index(self.attributes_values['Heating energy'], name='Heating energy'))
        return flow_area_constructed_he_ep

    @property
    def knowledge(self):
        return self._knowledge

    def _to_flow_constructed_dm(self):
        """Returns flow of constructed buildings segmented by decision-maker (dm) (Housing type, Occupancy status).

        1. Increase in the share of multi-family housing in the total stock.
        2. The share of owner-occupied and rented dwellings is held constant.

        Returns
        -------
        pd.Series
            MultiIndex: (Housing type, Occupancy status), value: buildings constructed
        """
        ht_share_tot_construction = self.to_share_housing_type()
        dm_share_tot_construction = (self._os_share_ht.T * ht_share_tot_construction).T.stack()
        return self.flow_constructed * dm_share_tot_construction

    def _to_flow_constructed_dm_he_ep(self, energy_price, cost_intangible=None, cost_invest=None,
                                      consumption='conventional', nu=8.0, subsidies=None):
        """Returns flow of constructed buildings segmented.

        Version Res-IRF 3.0.
        1. Calculate construction flow segmented by decision-maker:
        2. Calculate the market-share of Heating energy and Energy performance type by decision-maker: market_share_dm;
        3. Calculate construction flow segmented by decision-maker and heating energy, energy performance;

        Parameters
        ----------
        energy_price: pd.DataFrame
        cost_intangible: pd.DataFrame, optional
        cost_invest:  pd.DataFrame, optional
        consumption: {'conventional', 'actual'}, default 'conventional'
        nu: float, default 8.0
        subsidies: list, optional

        Returns
        -------
        pd.Series
            flow of constructed housing segmented by (Housing type, Occupancy status, Energy performance,
            Heating energy)
        """

        flow_constructed_dm = self.to_flow_constructed_dm()
        segments = get_levels_values(self._segments, ['Occupancy status', 'Housing type']).drop_duplicates()

        market_share_dm = self.to_market_share(energy_price,
                                               cost_invest=cost_invest,
                                               cost_intangible=cost_intangible,
                                               transition=['Energy performance', 'Heating energy'],
                                               consumption=consumption, nu=nu, subsidies=subsidies, segments=segments)[0]

        flow_constructed_dm = flow_constructed_dm.reorder_levels(market_share_dm.index.names)
        flow_constructed_seg = (flow_constructed_dm * market_share_dm.T).T
        flow_constructed_seg = flow_constructed_seg.stack(flow_constructed_seg.columns.names)

        for t in ['Energy performance', 'Heating energy']:
            flow_constructed_seg.index.rename('{}'.format(t),
                                              level=list(flow_constructed_seg.index.names).index('{} final'.format(t)),
                                              inplace=True)

        # at this point flow_constructed is not segmented by income class tenant and owner

        return flow_constructed_seg

    def _to_flow_constructed_seg(self, energy_price, cost_intangible=None, cost_invest=None,
                                 consumption=None, nu=8.0, subsidies=None):
        """Add Income class and Income class owner levels to flow_constructed.

        Version Res-IRF 3.0.
        io_share_seg: pd.DataFrame
            for each segment (rows) distribution of income class owner decile (columns)
        ic_share_seg: pd.DataFrame
            for each segment (rows) distribution of income class owner decile (columns)

        Parameters
        ----------
        energy_price: pd.DataFrame
                cost_intangible: pd.DataFrame, optional
        cost_intangible: pd.DataFrame, optional
        cost_invest:  pd.DataFrame, optional
        consumption: {'conventional', 'actual'}, default 'conventional'
        nu: float, default 8.0
        subsidies: list, optional

        Returns
        -------
        pd.Series
            flow of constructed housing segmented by (Housing type, Occupancy status, Energy performance,
            Heating energy, Income class, Income class owner)
        """

        flow_constructed_seg = self._to_flow_constructed_dm_he_ep(energy_price,
                                                                 cost_intangible=cost_intangible,
                                                                 cost_invest=cost_invest,
                                                                 consumption=consumption, nu=nu, subsidies=subsidies)
        # same repartition of income class
        owner_income = pd.DataFrame(1 / len(self.attributes_values['Income class owner']),
                                    columns=self.attributes_values['Income class owner'],
                                    index=flow_constructed_seg.index)
        owner_income.columns.set_names('Income class owner', inplace=True)
        flow_constructed_seg = (flow_constructed_seg * owner_income.T).T.stack()

        # keep the same proportion for income class owner than in the initial parc
        flow_constructed_seg = HousingStock.add_tenant_income(flow_constructed_seg, self._tenants_income)
        flow_constructed_seg = flow_constructed_seg[flow_constructed_seg > 0]

        return flow_constructed_seg
