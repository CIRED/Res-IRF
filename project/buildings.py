import pandas as pd
import os
import numpy as np
from scipy.optimize import fsolve
from copy import deepcopy
from itertools import product

from utils import reindex_mi, val2share, logistic, get_levels_values, remove_rows, add_level, de_aggregate_series

# TODO: calibration_renovation_rate // rho should depends on Income class owner and Heating energy initial
# TODO: clean HousingStockConstructed
# TODO: put exogenous parameters outside the script (heating intensity + lambda)
# TODO: use 4% for calibration cf. 2012 paper


class HousingStock:
    """Class represents housing stocks. Housing is a building and household/owner archetype.

    AttributesAttributes
    ----------
    stock_seg : pd.Series
        MultiIndex pd.Series that contains the number of archetypes by attribute
    attributes2area : pd.Series
        pd.Series that returns area (m2) of the building based on archetypes attribute
    attributes2horizon : pd.Series
        pd.Series that returns investment horizon (yrs) of the building owner based on archetypes attribute
    attributes2discount : pd.Series
        pd.Series that returns discount rate (%/yr) of the building owner based on archetypes attribute
    attributes2income : pd.Series
        pd.Series that returns income (€/yr) of the building owner based on archetypes attribute
    attributes2consumption : pd.Series
        pd.Series that returns conventional consumption (kWh/m2.yr) of the building based on archetypes attribute

    Methods
    -------
    to_consumption_actual(scenario=None)
        returns actual consumption considering household specific behavior
    """

    def __init__(self, stock_seg, attributes_values,
                 year=2018,
                 attributes2area=None,
                 attributes2horizon=None,
                 attributes2discount=None,
                 attributes2income=None,
                 attributes2consumption=None,
                 price_behavior='myopic'):
        """Initialize Housing Stock object.

        Parameters
        ----------
        stock_seg : pd.Series
            MultiIndex levels describing buildings attributes. Values are number of buildings.
        attributes_values: dict
            possible values for levels
        year: int
        attributes2area: float, pd.Series, pd.DataFrame, dict, optional
            area by segments
        attributes2horizon: float, pd.Series, pd.DataFrame, dict, optional
            investment horizon by segments
        attributes2discount: float, pd.Series, pd.DataFrame, dict, optional
            interest rate by segments
        attributes2income: float, pd.Series, pd.DataFrame, dict, optional
            income by segments
        attributes2consumption: float, pd.Series, pd.DataFrame, dict, optional
            consumption by segments
        """

        self._year = year
        self._calibration_year = year
        self._price_behavior = price_behavior

        self._stock_seg = stock_seg
        self._stock_seg_dict = {self.year: self._stock_seg}
        self._segments = stock_seg.index

        self._levels = stock_seg.index.names
        self._dimension = len(self._stock_seg.index.names)

        # explains what kind of levels needs to be used
        self.attributes_values = attributes_values

        self.attributes2area = attributes2area
        self.attributes2horizon = attributes2horizon
        self.attributes2discount = attributes2discount
        self.attributes2income = attributes2income
        self.attributes2consumption = attributes2consumption

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

        self.policies_detailed = deepcopy(temp)
        self.policies_total = deepcopy(temp)

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
    def stock_seg(self):
        return self._stock_seg

    @stock_seg.setter
    def stock_seg(self, new_stock):
        self._stock_seg = new_stock
        self._stock_seg_dict[self.year] = new_stock
        self._segments = new_stock.index

    @property
    def stock_seg_dict(self):
        return self._stock_seg_dict

    @staticmethod
    def data2area(l2area, ds_seg):
        area_seg = reindex_mi(l2area, ds_seg.index, l2area.index.names)
        return ds_seg * area_seg

    def add_flow(self, flow):
        flow = flow.reorder_levels(self.stock_seg.index.names)
        new_stock = self.stock_seg + flow
        new_stock.fillna(0, inplace=True)
        assert new_stock.min() >= 0, 'Buildings stock cannot be negative'
        self.stock_seg = new_stock.copy()

    @staticmethod
    def _attributes2(segments, attributes2, scenario=None):
        """Returns segmented value based on self._segments and by using attributes2 table.

        Parameters
        ----------
        attributes2: float, pd.Series, pd.DataFrame, dict
        scenario: str, optional

        Returns
        -------
        pd.Series or pd.DataFrame
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

    def to_stock_area_seg(self, scenario=None, segments=None):
        """Suppose that area_seg levels are included in self.levels.
        """
        if self.attributes2area is None:
            raise AttributeError('Need to define a table from attributes2area')
        if segments is None:
            segments = self._segments
        area = HousingStock._attributes2(segments, self.attributes2area, scenario=scenario)
        return area * self._stock_seg

    def to_income(self, scenario=None, segments=None):
        if self.attributes2income is None:
            raise AttributeError('Need to define a table from attributes2income')
        if segments is None:
            segments = self._segments
        return HousingStock._attributes2(segments, self.attributes2income, scenario=scenario)

    def to_area(self, scenario=None, segments=None):
        if self.attributes2area is None:
            raise AttributeError('Need to define a table from attributes2area')
        if segments is None:
            segments = self._segments
        return HousingStock._attributes2(segments, self.attributes2area, scenario=scenario)

    def to_horizon(self, scenario=None, segments=None):
        if self.attributes2horizon is None:
            raise AttributeError('Need to define a table from attributes2horizon')
        if segments is None:
            segments = self._segments
        horizon = HousingStock._attributes2(segments, self.attributes2horizon, scenario=scenario)
        return horizon

    def to_discount_rate(self, scenario=None, segments=None):
        if self.attributes2discount is None:
            raise AttributeError('Need to define a table from attributes2horizon')
        if segments is None:
            segments = self._segments
        discount_rate = HousingStock._attributes2(segments, self.attributes2discount, scenario=scenario)
        return discount_rate

    def to_discount_factor(self, scenario_horizon=None, scenario_discount=None, segments=None):
        """Calculate discount factor for all segments.

        Discount factor can be used when agents doesn't anticipate prices evolution.
        Discount factor does not depend on the year it is calculated.
        """
        if self.discount_factor is not None:
            return self.discount_factor
        else:
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
        """Returns discounted DataFrame from DataFrame.

        Parameters
        __________
        df: pd.DataFrame
        rate: float or pd.Series

        Returns
        -------
        pd.DataFrame
        """

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
        if self.consumption_conventional is not None:
            return self.consumption_conventional
        else:
            if self.attributes2consumption is None:
                raise AttributeError('Need to define a table from attributes2consumption')
            if segments is None:
                segments = self._segments
            return HousingStock._attributes2(segments, self.attributes2consumption, scenario=scenario)

    def to_consumption_actual(self, energy_prices, detailed_output=False, segments=None):
        """Return energy consumption for every segment and all years.

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
            If segments is not filled, use self.segments.

        Returns
        _______
        pd.DataFrame
            consumption_actual (rows: segments, columns: years)
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

    def to_consumption(self, consumption, segments=None):
        """Returns consumption conventional or actual based on consumption parameter.

        Parameters
        ----------
        consumption: str, {'conventional', 'actual'}
        segments: pd.Index, optional
            If segments is not filled, use self.segments.

        Returns
        -------

        """
        if consumption == 'conventional':
            return self.to_consumption_conventional(segments=segments)
        elif consumption == 'actual':
            # TODO: if consumption_actual is not filled will not worked
            return self.consumption_actual(segments=segments)
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

        temp = reindex_mi(temp, consumption.index, temp.index.names)

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

    @staticmethod
    def to_summed(df, yr_ini, horizon):
        """Summed df based on its horizon.

        Parameters
        ----------
        df: pd.DataFrame
            pd.MultiIndex as index, and years as columns
        yr_ini
            int
        horizon: pd.Series
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
                num: int
                yr: int

                Returns
                -------
                list

                Example:
                >>> horizon2years(2018, 3)
                [2018, 2019, 2020]
                >>> horizon2years(2020, 10)
                [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029]

                """
                return [yr + k for k in range(int(num))]

            yrs = horizon_re.apply(horizon2years, args=[yr_ini])

            def time_series2sum(ds, years, levels):
                """Sum n values over axis=1 based on years to consider for each row.

                Parameters
                ----------
                ds: pd.Series
                    Segments as index, years as column
                years: pd.Series
                    List of years to use for each segment
                levels: str
                    Levels used to catch idxyears

                Returns
                -------
                float

                Examples
                --------
                >>> data = pd.Series(np.ones((3, 5)), columns=[2018, 2019, 2020, 2021, 2022])
                >>> years = pd.Series([[2018, 2019, 2020], [2018, 2019, 2020, 2021], [2018]])
                >>> time_series2sum(data, years, levels)

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
        """Return segmented energy-life-cycle-cost discounted from segments, and energy prices year=yr.

        Energy LCC is calculated on an segment-specific horizon, and using a segment-specific discount rate.
        Because, time horizon depends of type of renovation (attributes, or heating energy), lcc needs to know which transition.
        NB: transition defined the investment horizon.
        Energy LCC depends on (Occupancy status, Housing type, Income class owner, Energy performance, Heating energy,
        and transition).

        Parameters
        ----------
        energy_prices: pd.DataFrame
        transition: list, default ['Energy performance']
        consumption: str, {'conventional', 'actual}, default 'conventional'
        segments: pd.Index, optional

        Returns
        -------
        pd.Series
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
                # TODO energy_cash_flows if consumption='conventional' from ds to df
                energy_cost_discounted_seg = HousingStock.to_discounted(energy_cash_flows, self.attributes2discount)
                energy_lcc = HousingStock.to_summed(energy_cost_discounted_seg, self.year, self.to_horizon(scenario=tuple(transition)))

            self.energy_lcc[tuple(transition)][consumption][self.year] = energy_lcc
            return energy_lcc

    def to_transition(self, ds, transition=None):
        """Returns pd.DataFrame from pd.Series by adding final state as column.

        Create a MultiIndex columns when it occurs simultaneous transitions.

        Parameters
        ----------
        ds: pd.Series
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
        """Catch final state segment values as the initial state of another segment.

        When a segment final state match another segment initial state,
        it's therefore fasten to directly catch the value.

        Parameters
        ----------
        ds: pd.Series
            Data to pick values
        idx_full: pd.MultiIndex
            Corresponds to final state data index
        transition: list

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
        """Returns pd.DataFrame

        Parameters
        ----------
        ds: pd.Series
        transition: list, default ['Energy performance', 'Heating energy']
        segments: pd.Index, optional
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
        energy_saving = self.to_energy_saving(transition=transition, consumption=consumption)
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

    def to_emission_saving(self, co2_content, transition=None, consumption='conventional'):
        """Calculate emission saving between initial and final state.

        Parameters
        ----------
        co2_content: pd.DataFrame
        transition: {(Energy performance, ), (Heating energy, ), (Energy performance, Heating energy)
        consumption: {'conventional', 'actual'}

        Returns
        -------
        dict
            {transition: {consumption: pd.Series or pd.DataFrame}}
        """

        if transition is None:
            transition = ['Energy performance']

        try:
            return self.emission_saving[tuple(transition)][consumption]
        except (KeyError, TypeError):
            consumption_initial = self.to_consumption(consumption)
            consumption_final = self.to_consumption_final(consumption=consumption, transition=transition)
            consumption_final = consumption_final.stack(consumption_final.columns.names)
            consumption_initial_re = reindex_mi(consumption_initial, consumption_final.index,
                                                consumption_initial.index.names)

            emission_initial = HousingStock.mul_consumption(consumption_initial_re, co2_content)
            emission_final = HousingStock.mul_consumption(consumption_final, co2_content, option='final')
            emission_saving = emission_initial - emission_final
            self.emission_saving[tuple(transition)][consumption] = emission_saving
            return emission_saving

    def to_emission_saving_lc(self, co2_content, transition=None, consumption='conventional', discount=0.04):
        """Calculate life-cycle emission saving between initial and final state for the entire project duration)

        Parameters
        ----------
        co2_content: pd.DataFrame
        transition: {(Energy performance, ), (Heating energy, ), (Energy performance, Heating energy)
        consumption: {'conventional', 'actual'}
        discount: float, default 0.04

        Returns
        -------
        dict
            {transition: {consumption: pd.Series}}
        """
        emission_saving = self.to_emission_saving(co2_content, transition=transition, consumption=consumption)
        if consumption == 'conventional':
            emission_saving = pd.concat([emission_saving] * 30, axis=1)
            emission_saving.columns = range(self.year, self.year + 30, 1)
        emission_saving_disc = HousingStock.to_discounted(emission_saving, discount)
        if 'Energy performance' in transition:
            horizon = 30
        else:
            horizon = 16
        emission_saving_lc = HousingStock.to_summed(emission_saving_disc, self.year, horizon)
        self.emission_saving_lc[tuple(transition)][consumption][self.year] = emission_saving_lc
        return emission_saving_lc

    def to_energy_lcc_final(self, energy_prices, transition=None, consumption='conventional', segments=None):
        """Calculate energy life-cycle cost based on transition.
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
                        index = add_level(pd.Series(dtype='float64', index=index), pd.Index(self.attributes_values[t], name=t)).index

            energy_lcc = self.to_energy_lcc(energy_prices, transition=transition, consumption=consumption,
                                            segments=index)

            energy_lcc_final = self.to_final(energy_lcc, transition=transition, segments=segments)
            self.energy_lcc_final[tuple(transition)][consumption][self.year] = energy_lcc_final
            return energy_lcc_final

    def to_lcc_final(self, energy_prices, cost_invest=None, cost_intangible=None,
                     transition=None, consumption='conventional', policies=None, segments=None):
        """Calculate life-cycle-cost of home-energy retrofits for every segment and every possible transition.

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
        policies: list, optional
            list of Policies object
        segments: pd.MultiIndex, optional

        Returns
        -------
        pd.DataFrame
            life-cycle-cost DataFrame is structured for every initial state (index) to every final state defined by transition (columns).
        """

        if transition is None:
            transition = ['Energy performance']

        lcc_final_seg = self.to_energy_lcc_final(energy_prices, transition, consumption=consumption, segments=segments)

        lcc_transition_seg = lcc_final_seg.copy()
        columns = lcc_transition_seg.columns

        capex_total = None
        capex = None
        capex_intangible = None
        for t in transition:
            if cost_invest[t] is not None:
                c = reindex_mi(cost_invest[t], lcc_final_seg.index)
                c = reindex_mi(c, lcc_final_seg.columns, c.columns.names, axis=1)
                if capex_total is None:
                    capex_total = c.copy()
                    capex = c.copy()
                else:
                    capex_total += c
                    capex += c
            if cost_intangible is not None:
                if cost_intangible[t] is not None:
                    c = reindex_mi(cost_intangible[t], lcc_final_seg.index, cost_intangible[t].index.names)
                    c = reindex_mi(c, lcc_final_seg.columns, c.columns.names, axis=1)
                    c.fillna(0, inplace=True)
                    capex_total += c
                    if capex_intangible is None:
                        capex_intangible = c.copy()
                    else:
                        capex_intangible += c

        self.capex[tuple(transition)][self.year] = capex
        self.capex_intangible[tuple(transition)][self.year] = capex_intangible
        self.capex_total[tuple(transition)][self.year] = capex_total

        self.policies_detailed[tuple(transition)][self.year] = dict()
        total_policies = None
        if policies is not None:
            for policy in policies:
                if policy.transition == transition:
                    if policy.policy == 'subsidies':
                        if policy.kind == '%':
                            s = policy.to_subsidy(cost=capex)
                            s.fillna(0, inplace=True)
                        elif policy.kind == '€/kWh':
                            # energy saving is kWh/m2
                            energy_saving = self.to_energy_saving_lc(transition=transition, consumption=consumption)
                            for t in transition:
                                energy_saving = energy_saving.unstack('{} final'.format(t))
                            s = policy.to_subsidy(energy_saving=energy_saving)
                            s[s < 0] = 0
                            s.fillna(0, inplace=True)
                            # € -> €/m2 :

                    elif policy.policy == 'regulated_loan':
                        capex_euro = (self.to_area() * capex.T).T
                        s = policy.to_opportunity_cost(capex_euro)
                        s = (s.T * (self.to_area() ** -1)).T
                        s = s.reindex(columns, axis=1)
                        s.fillna(0, inplace=True)

                    if total_policies is None:
                        total_policies = s
                    else:
                        total_policies += s

                    self.policies_detailed[tuple(transition)][self.year]['{} (€/m2)'.format(policy.name)] = s
                    self.policies_total[tuple(transition)][self.year] = total_policies

        if capex is not None:
            lcc_transition_seg += capex
        if total_policies is not None:
            lcc_transition_seg -= total_policies

        self.lcc_final[tuple(transition)][self.year] = lcc_transition_seg
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
            return (lcc_reverse_df.sum(axis=1) ** -1 * lcc_reverse_df.T).T
        elif isinstance(lcc_df, pd.Series):
            return lcc_reverse_df / lcc_reverse_df.sum()

    def to_market_share(self, energy_prices, transition=None, consumption='conventional', cost_invest=None,
                        cost_intangible=None, policies=None, nu=8.0, segments=None):
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
        policies: list, optional
            List of Policies object
        segments: pd.MultiIndex, optional
        nu: float or int, default 8.0
        Returns
        -------
        pd.DataFrame
            market_share, lcc_final
        """
        if transition is None:
            transition = ['Energy performance']

        lcc_final = self.to_lcc_final(energy_prices, cost_invest=cost_invest, cost_intangible=cost_intangible,
                                      transition=transition, consumption=consumption, policies=policies,
                                      segments=segments)

        market_share = HousingStock.lcc2market_share(lcc_final, nu=nu)
        # ms.columns.names = ['{} final'.format(transition)]
        self.market_share[tuple(transition)][self.year] = market_share
        return market_share, lcc_final

    def to_pv(self, energy_prices, transition=None, consumption='conventional', cost_invest=None, cost_intangible=None,
              policies=None, nu=8.0):
        # TODO: if NA
        if transition is None:
            transition = ['Energy performance']

        ms_final_seg, lcc_final_seg = self.to_market_share(energy_prices,
                                                           transition=transition,
                                                           consumption=consumption,
                                                           cost_invest=cost_invest,
                                                           cost_intangible=cost_intangible,
                                                           policies=policies,
                                                           nu=nu)

        pv = (ms_final_seg * lcc_final_seg).dropna(axis=0, how='all').sum(axis=1)
        self.pv[tuple(transition)][self.year] = pv
        return pv

    def to_npv(self, energy_prices, transition=None, consumption='conventional', cost_invest=None, cost_intangible=None,
               policies=None, nu=8.0):

        if transition is None:
            transition = ['Energy performance']

        energy_lcc_seg = self.to_energy_lcc(energy_prices, transition=transition, consumption=consumption)
        pv_seg = self.to_pv(energy_prices,
                            transition=transition,
                            consumption=consumption,
                            cost_invest=cost_invest,
                            cost_intangible=cost_intangible,
                            policies=policies, nu=nu)

        pv_seg.sort_index(inplace=True)
        energy_lcc_seg.sort_index(inplace=True)
        # assert energy_lcc_seg.index.equals(pv_seg.index), 'Index should match'

        npv = (energy_lcc_seg - pv_seg).dropna()
        self.npv[tuple(transition)][self.year] = npv
        return npv

    def to_calibration_market_share(self, energy_prices, market_share_objective, cost_invest=None,
                                    consumption='conventional', policies=None):
        """Returns intangible costs by calibrating market_share.

        Intangible costs are calibrated so that the observed market shares are reproduced in the initial year.
        Intangible costs are calibrated so that the life-cycle cost model, fed with the investment costs,
        matches the observed market shares.

        For each segment:
        LCC final is calculated (depends on Occupancy status, Housing type, Income class owner,
        Energy performance initial, Heating energy initial, transition), then Market Share.
        transition here is Energy performance
        NB: MS doesn't depend on Income class (tenant), so to fasten the function it can be first removed.
        Solver finds Intangible cost for each segment to match the observed market share.
        --> Intangible cost(Occupancy status, Housing type, Income class owner, Energy performance initial,
        Heating energy initial, Energy performance final)

        NB: Observed market share only depends on Energy performance initial and Energy performance final.
        Each segment with the same performance transition need to match the same observed market shares.


        Parameters:
        -----------
        energy_prices: pd.DataFrame
        market_share_objective: pd.DataFrame
            observed market share to match
        folder_output: str, optional
            if not None, will dump cost_intangible in .pkl file
        cost_invest: pd.DataFrame, optional
        consumption: {'conventional', 'actual'}, default 'conventional'
        policies: list, optional
            policies to consider in the market share

        Returns:
        --------
        pd.DataFrame
            Intangible cost,
        """

        lcc_final = self.to_lcc_final(energy_prices, consumption=consumption, cost_invest=cost_invest,
                                      transition=['Energy performance'], policies=policies)

        # remove income class as MultiIndex and drop duplicated indexes
        lcc_final.reset_index(level='Income class', drop=True, inplace=True)
        lcc_final = lcc_final[~lcc_final.index.duplicated(keep='first')]

        # remove idx when certificate = 'A' (no transition) and certificate = 'B' (intangible_cost = 0)
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
        num_certificate = list(lcc_final.index.names).index('Energy performance')
        for idx in lcc_useful.index:
            num_ini = self.attributes_values['Energy performance'].index(idx[num_certificate])
            certificate_final = self.attributes_values['Energy performance'][num_ini + 1:]
            # intangible cost would be for index = idx, and certificate_final.
            for lambda_current in range(int(lambda_min * 100), int(lambda_max * 100), int(step * 100)):
                lambda_current = lambda_current / 100
                lcc_row_np = lcc_final.loc[idx, certificate_final].to_numpy()
                ms_obj_np = ms_obj_approx.loc[idx, certificate_final].to_numpy()
                ier, root = solve_intangible_cost(lambda_current, lcc_row_np, ms_obj_np)
                if ier == 1:
                    lambda_list += [lambda_current]
                    idx_list += [idx]
                    intangible_list += [pd.Series(root ** 2, index=certificate_final)]
                    # func(root, lcc_row_np, ms_obj_np, lambda_current)
                    break

        intangible_cost = pd.concat(intangible_list, axis=1).T
        intangible_cost.index = pd.MultiIndex.from_tuples(idx_list)
        intangible_cost.index.names = lcc_final.index.names
        intangible_cost.columns.names = lcc_final.columns.names

        assert len(lcc_useful.index) == len(idx_list), "Calibration didn't work for all segments"

        # adding Income class that have been removed first
        intangible_cost = add_level(intangible_cost, pd.Index(self.attributes_values['Income class'], name='Income class'),
                                    axis=0)

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
            cost_lim = cost_lim.unstack('Energy performance final')
            level = 'Heating energy final'
            indexes = lbd.index.get_level_values(level).unique()
            temp = add_level(cost_lim.copy(), indexes, axis=1)
            return lbd * cost + (1 - lbd) * temp
        else:
            idx_union = lbd.index.union(cost.T.index)
            return lbd.reindex(idx_union) * cost.T.reindex(idx_union).T

    @staticmethod
    def information_rate(knowledge, learning_rate, info_max):
        """Returns information rate.

        More info_rate is high, more intangible_cost are low.
        Intangible renovation costs decrease according to a logistic curve with the same cumulative
        production so as to capture peer effects and knowledge diffusion.
        intangible_cost[yr] = intangible_cost[calibrationyear] * info_rate with info rate [1-info_rate_max ; 1]
        This function calibrate a logistic function, so rate of decrease is set at 25% for a doubling of cumulative
        production.

        Parameters
        ----------
        knowledge: pd.Series
            knowledge indexes match cost columns to reach final state after transition
        info_max: float
        learning_rate:

        Returns
        -------
        pd.Series
        """

        def equations(p, sh=info_max, alpha=learning_rate):
            a, r = p
            return (1 + a * np.exp(-r)) ** -1 - sh, (1 + a * np.exp(-2 * r)) ** -1 - sh - (1 - alpha) * sh + 1

        a, r = fsolve(equations, (1, -1))

        return logistic(knowledge, a=a, r=r) + 1 - info_max

    @staticmethod
    def acceleration_information(knowledge, cost_intangible, info_max, learning_rate):
        """Decrease intangible cost to capture peer effects and knowledge diffusion.

        Intangible renovation costs decrease according to a logistic curve with the same cumulative production so as
        to capture peer effects and knowledge diffusion.
        The rate of decrease (learning_rate) is set at 25% for a doubling of cumulative production.

        Parameters
        ----------
        knowledge: pd.Series
            knowledge indexes match cost columns to reach final state after transition
        cost_intangible: pd.DataFrame
        info_max: float
        learning_rate: float

        Returns
        -------
        pd.DataFrame
            cost_intangible
        """
        info_rate = HousingStock.information_rate(knowledge, learning_rate, info_max)

        temp = cost_intangible.T.copy()
        if isinstance(temp.index, pd.MultiIndex):
            info_rate = info_rate.reorder_levels(temp.index.names)
        cost_intangible = info_rate.loc[temp.index] * temp.T
        return cost_intangible

    def ini_energy_cash_flows(self, energy_price):
        """Initialize exogenous variable that doesn't depend on dynamic to fasten the script.

        For instance budget_share only depends on energy_price, and income that are exogenous variables.
        So does, heating_intensity and consumption_actual.

        List of attribute initialized:
        Initialized by launching self.to_consumption_actual(energy_price)
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
    def to_share_multi_family_tot(stock_needed, param):
        """Calculate share of multi-family buildings in the total stock.

        In Res-IRF 2.0, the share of single- and multi-family dwellings was held constant in both existing and new
        housing stocks, but at different levels; it therefore evolved in the total stock by a simple composition
        effect. These dynamics are now more precisely parameterized in Res-IRF 3.0 thanks to recent empirical
        work linking the increase in the share of multi-family housing in the total stock to the rate of growth of
        the total stock housing growth (Fisch et al., 2015).
        This relationship in particular reflects urbanization effects.

        Parameters
        ----------
        stock_needed: pd.Series
        param: float

        Returns
        -------
        dict
            Dictionary with year as keys and share of multi-family in the total stock as value.
            {2012: 0.393, 2013: 0.394, 2014: 0.395}
        """

        def func(stock, stock_ini, p):
            """Share of multi-family dwellings as a function of the growth rate of the dwelling stock.

            Parameters
            ----------
            stock: float
            stock_ini: float
            p: float

            Returns
            -------
            float
            """
            trend_housing = (stock - stock_ini) / stock * 100
            share = 0.1032 * np.log(10.22 * trend_housing / 10 + 79.43) * p
            return share

        share_multi_family_tot = {}
        stock_needed_ini = stock_needed.iloc[0]
        for year in stock_needed.index:
            share_multi_family_tot[year] = func(stock_needed.loc[year], stock_needed_ini, param)

        return share_multi_family_tot

    # TODO: not used
    def ini_all_indexes(self, energy_price, levels='all'):
        """Initialize values for all segments and transition.

        Parameters
        __________
        energy_price: pd.DataFrame
        levels: list, optional
            stock built will only consider levels as input
        """

        if levels == 'all':
            tpl = (self.attributes_values.values())
            tpl_names = self.attributes_values.keys()
        else:
            tpl = (self.attributes_values[lvl] for lvl in levels)
            tpl_names = levels

        idx_all = pd.MultiIndex.from_tuples(list(product(*tuple(tpl))))
        idx_all.names = tpl_names
        building_all = HousingStock(pd.Series(0, index=idx_all), self.attributes_values,
                                    year=self.year,
                                    attributes2area=self.attributes2area,
                                    attributes2horizon=self.attributes2horizon,
                                    attributes2discount=self.attributes2discount,
                                    attributes2income=self.attributes2income,
                                    attributes2consumption=self.attributes2consumption,
                                    price_behavior=self._price_behavior)

        self.area_all = building_all.to_area()
        consumption_conventional = building_all.to_consumption_conventional()
        self.consumption_conventional_all = consumption_conventional
        self.energy_cash_flows_all = dict()
        self.energy_cash_flows_disc_all = dict()
        energy_cash_flows = HousingStock.mul_consumption(consumption_conventional, energy_price)
        self.energy_cash_flows_all['conventional'] = energy_cash_flows
        self.energy_cash_flows_disc_all['conventional'] = HousingStock.to_discounted(energy_cash_flows,
                                                                                     building_all.attributes2discount)


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

    def __init__(self, stock_seg, attributes_values, year=2018,
                 residual_rate=0.0, destruction_rate=0.0,
                 rate_renovation_ini=None, learning_year=None,
                 npv_min=None, rate_max=None, rate_min=None,
                 attributes2area=None, attributes2horizon=None, attributes2discount=None, attributes2income=None,
                 attributes2consumption=None):

        super().__init__(stock_seg, attributes_values, year,
                         attributes2area=attributes2area,
                         attributes2horizon=attributes2horizon,
                         attributes2discount=attributes2discount,
                         attributes2income=attributes2income,
                         attributes2consumption=attributes2consumption)

        self.residual_rate = residual_rate
        self._destruction_rate = destruction_rate

        # slave stock of stock_seg property
        self._stock_seg_mobile = stock_seg * (1 - residual_rate)
        self._stock_seg_mobile_dict = {year: stock_seg * (1 - residual_rate)}
        self._stock_seg_residual = stock_seg * residual_rate
        self._stock_area_seg = self.to_stock_area_seg()

        # initializing knowledge
        flow_area_renovated_seg = self.flow_area_renovated_seg_ini(rate_renovation_ini, learning_year)
        self._flow_knowledge_ep = self.to_flow_knowledge(flow_area_renovated_seg)
        self._stock_knowledge_ep = self._flow_knowledge_ep
        self._stock_knowledge_ep_dict = {year: self._stock_knowledge_ep}
        self._knowledge = self._stock_knowledge_ep / self._stock_knowledge_ep
        self._knowledge_dict = {year: self._knowledge}

        # share of decision-maker in the total stock
        self._dm_share_tot = stock_seg.groupby(['Occupancy status', 'Housing type']).sum() / stock_seg.sum()

        # calibration
        self.rate_renovation_ini = rate_renovation_ini
        self.rho = pd.Series()
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
        self._stock_seg_mobile = new_stock_seg * (1 - self.residual_rate)
        self._stock_seg_mobile_dict[self.year] = self._stock_seg_mobile
        self._stock_seg_residual = new_stock_seg * self.residual_rate
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
    def renovate_rate_func(npv, rho, npv_min, rate_max, rate_min):
        if isinstance(rho, pd.Series):
            rho_f = rho.loc[tuple(npv.iloc[:-1].tolist())]
        else:
            rho_f = rho

        if np.isnan(rho_f):
            return float('nan')
        else:
            return logistic(npv.loc[0] - npv_min,
                            a=rate_max / rate_min - 1,
                            r=rho_f,
                            k=rate_max)

    def to_renovation_rate(self, energy_prices, transition=None, consumption='conventional', cost_invest=None,
                           cost_intangible=None, policies=None):

        """Routine calculating renovation rate from segments for a particular yr.

        Cost (energy, investment) & rho parameter are also required.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        transition: list
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        policies: list, optional
        """
        if transition is None:
            transition = ['Energy performance']

        npv_seg = self.to_npv(energy_prices,
                              transition=transition,
                              consumption=consumption,
                              cost_invest=cost_invest,
                              cost_intangible=cost_intangible,
                              policies=policies)
        renovation_rate_seg = npv_seg.reset_index().apply(HousingStockRenovated.renovate_rate_func,
                                                          args=[self.rho, self._npv_min, self._rate_max,
                                                                self._rate_min], axis=1)
        renovation_rate_seg.index = npv_seg.index
        self.renovation_rate_dict[tuple(transition)][self.year] = renovation_rate_seg
        return renovation_rate_seg

    def to_flow_renovation_ep(self, energy_prices, consumption='conventional', cost_invest=None, cost_intangible=None,
                              policies=None, renovation_obligation=None, mutation=0.0, rotation=0.0):
        """Calculate flow renovation by energy performance final.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        policies: dict, optional
        renovation_obligation: RenovationObligation, optional
        mutation: pd.Series or float, default 0.0
        rotation: pd.Series or float, default 0.0

        Returns
        -------
        pd.DataFrame
        """

        # TODO: add nu as a parameter
        transition = ['Energy performance']
        renovation_rate_seg = self.to_renovation_rate(energy_prices,
                                                      transition=transition,
                                                      consumption=consumption,
                                                      cost_invest=cost_invest,
                                                      cost_intangible=cost_intangible,
                                                      policies=policies)
        stock_seg = self.stock_seg
        flow_renovation_obligation = 0
        if renovation_obligation is not None:
            flow_renovation_obligation = self.to_flow_obligation(renovation_obligation,
                                                                 mutation=mutation, rotation=rotation)
            stock_seg = stock_seg - flow_renovation_obligation

        flow_renovation_seg = renovation_rate_seg * stock_seg
        flow_renovation_seg += flow_renovation_obligation

        # indicators
        renovation_rate_aggr = flow_renovation_seg.sum() / stock_seg.sum()
        renovation_rate_dm = flow_renovation_seg.groupby(['Occupancy status', 'Housing type']).sum() / stock_seg[stock_seg.index.get_level_values('Energy performance') != 'A'].groupby(['Occupancy status', 'Housing type']).sum()
        self.renovation_rate_dm[self.year] = renovation_rate_dm

        if self.year in self.market_share[tuple(transition)]:
            market_share_seg_ep = self.market_share[tuple(transition)][self.year]
        else:
            market_share_seg_ep = self.to_market_share(energy_prices,
                                                       transition=transition,
                                                       consumption=consumption,
                                                       cost_invest=cost_invest,
                                                       cost_intangible=cost_intangible,
                                                       policies=policies)[0]

        flow_renovation_seg_ep = (flow_renovation_seg * market_share_seg_ep.T).T
        self.flow_renovation_label_dict[self.year] = flow_renovation_seg_ep

        return flow_renovation_seg_ep

    def to_flow_renovation_ep_energy(self, energy_prices, consumption='conventional', cost_invest=None,
                                     cost_intangible=None, policies=None, renovation_obligation=None, mutation=0.0,
                                     rotation=0.0):
        """De-aggregate stock_renovation_attributes by final heating energy.

        stock_renovation columns segmented by final attributes and final heating energy.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        policies: list, optional
        renovation_obligation: RenovationObligation, optional
        mutation: pd.Series or float, default 0.0
        rotation: pd.Series or float, default 0.0

        Returns
        -------
        pd.DataFrame
        """

        market_share_seg_he = self.to_market_share(energy_prices,
                                                   transition=['Heating energy'],
                                                   cost_invest=cost_invest,
                                                   consumption=consumption,
                                                   policies=policies)[0]

        ms_temp = pd.concat([market_share_seg_he.T] * len(self.attributes_values['Energy performance']),
                            keys=self.attributes_values['Energy performance'], names=['Energy performance final'])

        flow_renovation_seg_attributes = self.to_flow_renovation_ep(energy_prices,
                                                                    consumption=consumption,
                                                                    cost_invest=cost_invest,
                                                                    cost_intangible=cost_intangible,
                                                                    policies=policies,
                                                                    renovation_obligation=renovation_obligation,
                                                                    mutation=mutation, rotation=rotation)

        sr_temp = pd.concat([flow_renovation_seg_attributes.T] * len(self.attributes_values['Heating energy']),
                            keys=self.attributes_values['Heating energy'], names=['Heating energy final'])
        flow_renovation_label_energy = (sr_temp * ms_temp).T
        self.flow_renovation_label_energy_dict[self.year] = flow_renovation_label_energy

        return flow_renovation_label_energy

    def to_flow_remained(self, energy_prices, consumption='conventional', cost_invest=None, cost_intangible=None,
                         policies=None, renovation_obligation=None, mutation=0.0, rotation=0.0):
        """Calculate flow_remained for each segment.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        cost_invest: dict, optional
        cost_intangible: dict, optional
        consumption: str, default 'conventional'
        policies: list, optional
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
                                                                                policies=policies,
                                                                                renovation_obligation=renovation_obligation,
                                                                                mutation=mutation, rotation=rotation
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

    def to_flow_demolition_dm(self):
        flow_demolition = self._stock_seg.sum() * self._destruction_rate
        flow_demolition_dm = self._dm_share_tot * flow_demolition
        # flow_area_demolition_seg_dm = flow_demolition_seg_dm * self.attributes2area
        return flow_demolition_dm

    def to_flow_demolition_seg(self):
        """ Returns stock_demolition -  segmented housing number demolition.

        Buildings to destroy are chosen in stock_mobile.
        1. type_housing_demolition is respected to match decision-maker proportion; - type_housing_demolition_reindex
        2. income_class, income_class_owner, heating_energy match stock_remaining proportion; - type_housing_demolition_wo_performance
        3. worst energy_performance_attributes for each segment are targeted. - stock_demolition

        Returns
        -------
        pd.Series segmented
        """

        flow_demolition_dm = self.to_flow_demolition_dm()

        stock_mobile = self._stock_seg_mobile_dict[self.year - 1]
        stock_mobile_ini = self._stock_seg_mobile_dict[self._calibration_year]
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
                for lbl in self.attributes_values['Energy performance']:
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
        # initialize flow_demolition_remain with worst certificate based on how much have been demolition so far
        flow_demolition_remain = prop_stock_worst_certificate * flow_demolition_ini

        # we year with the worst attributes and we stop when nb_housing_demolition_theo == 0
        flow_demolition = pd.Series(0, index=stock_mobile.index, dtype='float64')
        for segment in segments_mobile:
            # dangerous conditions
            if segment in worst_certificate_dict.keys():
                certificate = worst_certificate_dict[segment]
            else:
                continue
            num = self.attributes_values['Energy performance'].index(certificate)
            idx_w_ep = (segment[0], segment[1], segment[2], segment[3], certificate, segment[4])

            while flow_demolition_remain.loc[idx_w_ep] != 0:
                # stock_demolition cannot be sup to stock_mobile and to flow_demolition_theo
                flow_demolition.loc[idx_w_ep] = min(stock_mobile.loc[idx_w_ep], flow_demolition_remain.loc[idx_w_ep])

                if certificate != 'A':
                    num += 1
                    certificate = self.attributes_values['Energy performance'][num]
                    certificates = [c for c in self.attributes_values['Energy performance'] if c > certificate]
                    idx_wo_ep = (segment[0], segment[1], segment[2], segment[3], segment[4])
                    idx_w_ep = (segment[0], segment[1], segment[2], segment[3], certificate, segment[4])
                    list_idx = [(segment[0], segment[1], segment[2], segment[3], c, segment[4]) for c in certificates]

                    # flow_demolition_remain: remaining housing that need to be destroyed for this segment
                    flow_demolition_remain[idx_w_ep] = flow_demolition_wo_ep.loc[idx_wo_ep] - flow_demolition.loc[list_idx].sum()

                else:
                    # stop while loop --> all buildings has not been destroyed (impossible case anyway)
                    flow_demolition_remain[idx_w_ep] = 0

        assert (stock_mobile - flow_demolition).min() >= 0, 'Demolition more than mobile stock'

        self.flow_demolition_dict[self.year] = flow_demolition
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

        def flow_area2knowledge(flow, flow_area_ep, certificate1, certificate2):
            """Very specific function aggregating two energy performance certificate to calculate knowledge.

            Parameters
            ----------
            flow: pd.Series
                Data array containing knowledge.
            flow_area_ep: pd.Series or pd.DataFrame
            certificate1: str, {'A', 'B', 'C', 'D', 'E', 'F', 'G'}
            certificate2: str, {'A', 'B', 'C', 'D', 'E', 'F', 'G'}

            Returns
            -------
            spd.Series
            """
            if certificate1 in flow_area_ep.index and certificate2 in flow_area_ep.index:
                flow.loc[certificate1] = flow_area_ep.loc[certificate1] + flow_area_ep.loc[
                    certificate2]
                flow.loc[certificate2] = flow_area_ep.loc[certificate1] + flow_area_ep.loc[
                    certificate2]
            elif certificate2 not in flow_area_ep.index:
                flow.loc[certificate1] = flow_area_ep.loc[certificate1]
                flow.loc[certificate2] = flow_area_ep.loc[certificate1]
            else:
                flow.loc[certificate1] = flow_area_ep.loc[certificate2]
                flow.loc[certificate2] = flow_area_ep.loc[certificate2]
            return flow

        flow_knowledge_renovation = pd.Series(dtype='float64',
                                              index=[ep for ep in self.attributes_values['Energy performance'] if
                                                     ep != 'G'])

        flow_knowledge_renovation = flow_area2knowledge(flow_knowledge_renovation, flow_area_renovated_ep, 'A', 'B')
        flow_knowledge_renovation = flow_area2knowledge(flow_knowledge_renovation, flow_area_renovated_ep, 'C', 'D')
        flow_knowledge_renovation = flow_area2knowledge(flow_knowledge_renovation, flow_area_renovated_ep, 'E', 'F')

        flow_knowledge_renovation.index.set_names('Energy performance final', inplace=True)
        return flow_knowledge_renovation

    def update_stock(self, flow_remained_seg, flow_area_renovation_seg=None):

        # update segmented stock  considering renovation
        self.add_flow(flow_remained_seg)

        if flow_area_renovation_seg is not None:
            flow_knowledge_renovation = self.to_flow_knowledge(flow_area_renovation_seg)
            self.flow_knowledge_ep = flow_knowledge_renovation

    def calibration_renovation_rate(self, energy_prices, renovation_rate_obj, consumption='conventional',
                                    cost_invest=None, cost_intangible=None, policies=None):
        """Calibration of ρ parameter of the renovation rate function (logistic function of the NPV).

        Renovation rate of dwellings attributesled is calculated as a logistic function of the NPV.
        The logistic form captures heterogeneity in heating preference and habits,
        assuming they are normally distributed.

        Parameter ρ is calibrated, for each type of decision-maker and each initial certificates (i.e., 6x6=36 values),
        so that the NPVs calculated with the subsidies in effect in 2012 (see main article)
        reproduce the observed renovation rates.

        Renovation rate observed depends on (Occupancy status, Housing type)
        NPV that depends on MS and so LCC: (Energy performance initial, Energy performance final,
                                            Occupancy status, Housing type, Income class owner)
        So Parameter ρ should depend on  (Occupancy status, Housing type, Energy performance initial,
                                            Energy performance final, Heating energy intial)
        But, function removed Heating energy initial and Income class owner dependency.

        Parameters
        ----------
        energy_prices: pd.DataFrame
        renovation_rate_obj: pd.Series
        consumption: {'conventional', 'actual'}, default 'conventional'
        cost_invest: dict
        cost_intangible: dict
        policies: list

        Returns
        -------
        pd.Series
            ρ parameters by segment
        """

        npv = self.to_npv(energy_prices,
                          transition=['Energy performance'],
                          consumption=consumption,
                          cost_invest=cost_invest,
                          cost_intangible=cost_intangible,
                          policies=policies)

        renovation_rate_obj = reindex_mi(renovation_rate_obj, npv.index, renovation_rate_obj.index.names)
        rho = (np.log(self._rate_max / self._rate_min - 1) - np.log(
            self._rate_max / renovation_rate_obj - 1)) / (npv - self._npv_min)

        levels = ['Occupancy status', 'Housing type', 'Energy performance']
        weight = val2share(self.stock_seg, levels, option='column')
        rho_temp = rho.unstack(['Income class', 'Heating energy', 'Income class owner'])
        rho_weighted = (weight * rho_temp).fillna(0).sum(axis=1)

        """self.rho = rho
        renovation_rate_bis = self.to_renovation_rate(energy_prices,
                                                        transition=['Energy performance'],
                                                        consumption='conventional',
                                                        cost_invest=cost_invest,
                                                        cost_intangible=cost_intangible, policies=policies)

        renovation_rate_tiers = self.to_renovation_rate(energy_prices,
                                                        transition=['Energy performance'],
                                                        consumption='conventional',
                                                        cost_invest=cost_invest,
                                                        cost_intangible=cost_intangible)
        """

        return rho

    def to_flow_obligation(self, renovation_obligation, mutation=0.0, rotation=0.0):
        if isinstance(mutation, pd.Series):
            mutation = reindex_mi(mutation, self.stock_seg.index, mutation.index.names)
        if isinstance(rotation, pd.Series):
            rotation = reindex_mi(rotation, self.stock_seg.index, rotation.index.names)

        mutation_stock = self.stock_seg * mutation
        rotation_stock = self.stock_seg * rotation

        target_stock = mutation_stock + rotation_stock
        target = renovation_obligation.targets.loc[:, self.year]
        target = reindex_mi(target, target_stock.index, target.index.names)
        flow_renovation_obligation = target * target_stock * renovation_obligation.participation_rate
        self.flow_renovation_obligation[self.year] = flow_renovation_obligation
        return flow_renovation_obligation


class HousingStockConstructed(HousingStock):
    def __init__(self, stock, attributes_values, year, stock_needed_ts,
                 param_share_multi_family=None,
                 os_share_ht=None,
                 io_share_seg=None,
                 stock_area_existing_seg=None,
                 attributes2area=None,
                 attributes2horizon=None,
                 attributes2discount=None,
                 attributes2income=None,
                 attributes2consumption=None):

        super().__init__(stock, attributes_values,
                         year=year,
                         attributes2area=attributes2area,
                         attributes2discount=attributes2discount,
                         attributes2income=attributes2income,
                         attributes2consumption=attributes2consumption,
                         attributes2horizon=attributes2horizon)

        self._flow_constructed = 0
        self._flow_constructed_dict = {self.year: self._flow_constructed}

        self._flow_constructed_seg = None
        self._flow_constructed_seg_dict = {self.year: self._flow_constructed_seg}
        self._stock_constructed_seg_dict = {self.year: self._flow_constructed_seg}

        self._stock_needed_ts = stock_needed_ts
        self._stock_needed = stock_needed_ts.loc[self._calibration_year]
        # used to estimate share of housing type
        # TODO: to_share_multi_family_tot is exogenous and must be done elsewhere
        self._share_multi_family_tot_dict = HousingStock.to_share_multi_family_tot(stock_needed_ts,
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
        """if self._stock_constructed_seg_dict[self.year - 1] is not None:
            self._stock_constructed_seg_dict[self.year] = self._stock_constructed_seg_dict[self.year - 1] + val
        else:
            self._stock_constructed_seg_dict[self.year] = val
        """
        self.add_flow(val)

        flow_area_constructed_seg = HousingStockConstructed.data2area(self.attributes2area, val)
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

    def to_flow_constructed_dm(self):
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

    def to_flow_constructed_dm_he_ep(self, energy_price, cost_intangible=None, cost_invest=None,
                                     consumption='conventional', nu=8.0, policies=None):
        """Returns flow of constructed buildings segmented.

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
        policies: list, optional

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
                                               consumption=consumption, nu=nu, policies=policies, segments=segments)[0]
        flow_constructed_dm = flow_constructed_dm.reorder_levels(market_share_dm.index.names)
        flow_constructed_seg = (flow_constructed_dm * market_share_dm.T).T
        flow_constructed_seg = flow_constructed_seg.stack(flow_constructed_seg.columns.names)

        for t in ['Energy performance', 'Heating energy']:
            flow_constructed_seg.index.rename('{}'.format(t),
                                              level=list(flow_constructed_seg.index.names).index('{} final'.format(t)),
                                              inplace=True)

        # at this point flow_constructed is not segmented by income class tenant and owner
        return flow_constructed_seg

    def to_flow_constructed_seg(self, energy_price, cost_intangible=None, cost_invest=None,
                                consumption=None, nu=8.0, policies=None):
        """Add Income class and Income class owner levels to flow_constructed.

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
        policies: list, optional

        Returns
        -------
        pd.Series
            flow of constructed housing segmented by (Housing type, Occupancy status, Energy performance,
            Heating energy, Income class, Income class owner)
        """

        flow_constructed_seg = self.to_flow_constructed_dm_he_ep(energy_price,
                                                                 cost_intangible=cost_intangible,
                                                                 cost_invest=cost_invest,
                                                                 consumption=consumption, nu=nu, policies=policies)
        # same repartition of income class
        seg_index = flow_constructed_seg.index
        seg_names = flow_constructed_seg.index.names
        val = 1 / len(self.attributes_values["Income class"])
        temp = pd.Series(val, index=self.attributes_values["Income class"])
        ic_share_seg = pd.concat([temp] * len(seg_index), axis=1).T
        ic_share_seg.index = seg_index

        flow_constructed_seg = de_aggregate_series(flow_constructed_seg, ic_share_seg)
        flow_constructed_seg.index.names = seg_names + ['Income class']

        # keep the same proportion for income class owner than in the initial parc
        flow_constructed_seg = de_aggregate_series(flow_constructed_seg, self._io_share_seg)
        flow_constructed_seg = flow_constructed_seg[flow_constructed_seg > 0]
        return flow_constructed_seg

    def update_flow_constructed_seg(self, energy_price, cost_intangible=None, cost_invest=None,
                                    consumption='conventional', nu=8.0, policies=None):
        """Update HousingConstructed object flow_constructed_seg attribute.

        Parameters
        ----------
        energy_price: pd.DataFrame
                cost_intangible: pd.DataFrame, optional
        cost_intangible: dict, optional
        cost_invest:  dict, optional
        consumption: {'conventional', 'actual'}, default 'conventional'
        nu: float, default 8.0
        policies: list, optional
        """
        flow_constructed_seg = self.to_flow_constructed_seg(energy_price,
                                                            cost_intangible=cost_intangible,
                                                            cost_invest=cost_invest,
                                                            consumption=consumption, nu=nu, policies=policies)

        self.flow_constructed_seg = flow_constructed_seg
        return flow_constructed_seg

    """
    @staticmethod
    def to_market_share_objective(os_share_ht_construction, he_share_ht_construction, ht_share_tot_construction,
                                  ep_share_tot_construction):
        os_he_share_ht_construction = de_aggregate_columns(os_share_ht_construction, he_share_ht_construction)
        os_he_ht_share_tot_construction = (ht_share_tot_construction * os_he_share_ht_construction.T).T.stack().stack()

        seg_share_construction = de_aggregating_series(os_he_ht_share_tot_construction,
                                                       ep_share_tot_construction,
                                                       level='Energy performance')
        market_share_objective = val2share(seg_share_construction, ['Occupancy status', 'Housing type'],
                                           option='column')
        # market_share_objective = market_share_objective.droplevel(None, axis=1)
        return market_share_objective
    """

    def to_calibration_market_share(self, energy_price, market_share_objective, cost_invest=None,
                                    consumption='conventional', policies=None):
        """Returns intangible costs construction by calibrating market_share.
        """

        lcc_final = self.to_lcc_final(energy_price, cost_invest=cost_invest, policies=policies,
                                      transition=['Energy performance', 'Heating energy'], consumption=consumption,
                                      segments=market_share_objective.index)

        market_share_objective.sort_index(inplace=True)
        lcc_final = lcc_final.reorder_levels(market_share_objective.index.names)
        lcc_final.sort_index(inplace=True)

        def approximate_ms_objective(ms_obj):
            """Treatment of market share objective to facilitate resolution.
            """
            ms_obj[ms_obj == 0] = 0.001
            return (ms_obj.sum(axis=1) ** -1 * ms_obj.T).T

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

        attributes_final = lcc_final.columns
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
                    intangible_list += [pd.Series(root ** 2, index=attributes_final)]
                    break

        intangible_cost = pd.concat(intangible_list, axis=1).T
        intangible_cost.index = pd.MultiIndex.from_tuples(idx_list)
        intangible_cost.index.names = lcc_final.index.names
        return intangible_cost

    @staticmethod
    def evolution_area_construction(area_construction_prev, area_construction_ini, area_max_construction,
                                    elasticity_area, available_income_ratio):
        """Evolution of new buildings area based on total available income. Function represents growth.

        Parameters
        ----------
        area_construction_prev: pd.Series
        area_construction_ini: pd.Series
        area_max_construction: pd.Series
        elasticity_area: float
        available_income_ratio: float

        Returns
        -------
        pd.Series
        """
        area_max_construction = area_max_construction.reorder_levels(area_construction_ini.index.names)

        eps_area_new = (area_max_construction - area_construction_prev) / (
                area_max_construction - area_construction_ini)
        eps_area_new = eps_area_new.apply(lambda x: max(0, min(1, x)))
        elasticity_area_new = eps_area_new.multiply(elasticity_area)

        factor_area_new = elasticity_area_new * max(0, available_income_ratio - 1)

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
        elasticity_area_new_ini
        available_income_real_pop_ds: pd.Series
        area_max_construction
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
        """
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
        """
        self.attributes2area = area_construction
        self._area_construction_dict[self.year] = self.attributes2area

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
                                                pd.Index(self.attributes_values['Heating energy'], name='Heating energy'))
        return flow_area_constructed_he_ep
