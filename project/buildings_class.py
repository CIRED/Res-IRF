import pandas as pd
import os
import numpy as np
from scipy.optimize import fsolve
import logging

from function_pandas import *
from project.func import logistic
from project.input import calibration_dict, technical_progress_dict, parameters_dict


class HousingStock:
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

    def __init__(self, stock_seg, levels_values, year,
                 label2area=None,
                 label2horizon_envelope=None,
                 label2horizon_heater=None,
                 label2discount=None,
                 label2income=None,
                 label2consumption=None):
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

        self._stock_seg = stock_seg
        self._stock_seg_dict = {self._year: self._stock_seg}
        self._segments = stock_seg.index

        self._levels = stock_seg.index.names
        self._dimension = len(self._stock_seg.index.names)

        # explains what kind of levels needs to be used
        self.levels_values = levels_values

        self.label2area = label2area

        label2horizon = dict()
        label2horizon['envelope'] = label2horizon_envelope
        label2horizon['heater'] = label2horizon_heater
        self.label2horizon = label2horizon

        self.label2discount = label2discount
        self.label2income = label2income
        self.label2consumption = label2consumption

    @property
    def stock_seg(self):
        return self._stock_seg

    @stock_seg.setter
    def stock_seg(self, new_stock):
        self._stock_seg = new_stock
        self._stock_seg_dict = {self._year: self._stock_seg}
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

        if isinstance(val, float):
            val = pd.Series(val, index=self._segments)
        elif isinstance(val, pd.Series) or isinstance(val, pd.DataFrame):
            val = reindex_mi(val, self._segments, val.index.names)

        return val

    def numbers2area(self, scenario=None):
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

    def discount_rate_series(self, index_yr):
        """Return pd.DataFrame - partial segments in index and years in column - with value corresponding to discount rate.
        """

        def interest_rate2series(interest_rate, idx_yr):
            return [(1 + interest_rate) ** -(yr - idx_yr[0]) for yr in idx_yr]

        if isinstance(self.label2discount, pd.Series):
            discounted_df = self.label2discount.apply(interest_rate2series, args=[index_yr])
            discounted_df = pd.DataFrame.from_dict(dict(zip(discounted_df.index, discounted_df.values))).T
            discounted_df.columns = index_yr
            discounted_df.index.names = self.label2discount.index.names
        elif isinstance(self.label2discount, float):
            discounted_df = pd.Series(interest_rate2series(self.label2discount, index_yr), index=index_yr)
        else:
            return ValueError
        return discounted_df

    def to_income(self, scenario=None):
        if self.label2income is None:
            raise AttributeError('Need to define a table from label2income')
        return self.__label2(self.label2income, scenario=scenario)

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
        return self.__label2(self.label2consumption, scenario=scenario)

    def to_consumption_actual(self, energy_prices):
        """
        energy_prices is a DataFrame with segments as rows, and years as columns.
        """
        area_seg = self.numbers2area()
        income_seg = self.to_income()
        energy_prices = reindex_mi(energy_prices, self._segments, energy_prices.index.names)
        consumption_conventional = self.to_consumption_conventional()
        budget_share_seg = ds_mul_df(area_seg * consumption_conventional, energy_prices / income_seg)
        use_intensity_seg = -0.191 * budget_share_seg.apply(np.log) + 0.1105
        consumption_actual = ds_mul_df(consumption_conventional, use_intensity_seg)
        result_dict = {'Budget share': budget_share_seg,
                       'Use intensity': use_intensity_seg,
                       'Consumption-conventional': consumption_conventional,
                       'Consumption-actual': consumption_actual,
                       }
        return result_dict

    @staticmethod
    def energy_consumption2cost(consumption, energy_prices):
        """Returns energy cost segmented and for every year based on energy consumption and energy prices.
        """
        energy_prices = reindex_mi(energy_prices, consumption.index, ['Heating energy'])
        if isinstance(consumption, pd.DataFrame):
            return energy_prices * consumption
        elif isinstance(consumption, pd.Series):
            return ds_mul_df(consumption, energy_prices)

    def to_energy_lcc(self, energy_prices, transition='Energy performance', consumption='conventional'):
        """Return segmented energy-life-cycle-cost discounted from segments, and energy prices year=yr.

        Energy LCC is calculated on an segment-specific horizon, and using a segment-specific discount rate.
        Because, time horizon depends of type of renovation (label, or heating energy), lcc needs to know which transition.
        Energy LCC can also be calculated for new constructed buildings: kind='new'.
        Transition defined the investment horizon.
        """

        if consumption == 'conventional':
            consumption_seg = self.to_consumption_conventional()
        elif consumption == 'actual':
            consumption_seg = self.to_consumption_actual(energy_prices)['Consumption-' + consumption]
        else:
            raise AttributeError("Consumption must be in ['conventional', 'actual']")

        energy_cost_seg = HousingStock.energy_consumption2cost(consumption_seg, energy_prices)
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
            return [year_yr + k for k in range(num)]

        if transition == 'Energy performance':
            invest_horizon_seg = self.to_horizon(scenario='envelope')
        elif transition == 'Heating energy':
            invest_horizon_seg = self.to_horizon(scenario='heater')
        else:
            raise AttributeError("Transition can only be 'Energy performance' or 'Heating energy' for now")

        invest_years = invest_horizon_seg.apply(horizon2years, args=[self._year])

        def time_series2sum(ds, years, levels):
            """Return sum of ds for each segment based on list of years in invest years.

            Parameters:
            -----------
            ds: pd.Series
            segments as index, time series as column

            years: pd.Series
            list of years to use for each segment

            levels: str
            levels used to catch idx_years

            Returns:
            --------
            float
            """
            idx_invest = [ds[lvl] for lvl in levels]
            idx_years = years.loc[tuple(idx_invest)]
            return ds.loc[idx_years].sum()

        energy_lcc = energy_cost_discounted_seg.reset_index().apply(time_series2sum, args=[invest_years, self._levels], axis=1)
        energy_lcc.index = energy_cost_discounted_seg.index
        return energy_lcc

    def to_transition(self, ds, transition):
        """Create pd.DataFrame from pd.Series by duplicating column.

        Create a MultiIndex columns when transition is a list.
        """

        if isinstance(transition, str):
            return pd.concat([ds] * len(self.levels_values[transition]),
                             keys=self.levels_values[transition], names=['{} final'.format(transition)], axis=1)

        elif isinstance(transition, list):
            for t in transition:
                ds = pd.concat([ds] * len(self.levels_values[t]),
                               keys=self.levels_values[t], names=['{} final'.format(t)], axis=1)
            return ds

    def _energy_lcc_final(self, energy_prices, transition, consumption='conventional'):
        """Define a final state based on transition.
        """
        lcc_transition_seg = self.to_transition(pd.Series(dtype='float64', index=self._segments), transition)

        temp = lcc_transition_seg.copy()
        temp.index.rename('{} initial'.format(transition), level=list(temp.index.names).index('{}'.format(transition)),
                          inplace=True)
        temp = temp.stack(dropna=False)
        temp.index.rename('{}'.format(transition), level=list(temp.index.names).index('{} final'.format(transition)),
                          inplace=True)
        stock_seg_final = pd.Series(dtype='float64', index=temp.index)
        energy_final_lcc = HousingStock(stock_seg_final, self.levels_values, self._year,
                                        label2area=self.label2area,
                                        label2horizon_envelope=self.label2horizon['envelope'],
                                        label2horizon_heater=self.label2horizon['heater'],
                                        label2discount=self.label2discount,
                                        label2income=self.label2income,
                                        label2consumption=self.label2consumption).to_energy_lcc(energy_prices,
                                                                                                transition=transition,
                                                                                                consumption=consumption)
        energy_final_lcc = energy_final_lcc.unstack('{}'.format(transition))
        energy_final_lcc.columns.rename('{} final'.format(transition), inplace=True)
        energy_final_lcc.index.rename('{}'.format(transition),
                                      level=list(energy_final_lcc.index.names).index('{} initial'.format(transition)),
                                      inplace=True)

        energy_final_lcc = energy_final_lcc.reorder_levels(lcc_transition_seg.index.names)
        energy_final_lcc.sort_index(inplace=True)
        lcc_transition_seg.sort_index(inplace=True)
        assert energy_final_lcc.index.equals(lcc_transition_seg.index), 'Index should match'

        return energy_final_lcc

    def cost2lcc_final(self, energy_prices, cost_invest=None, cost_switch_fuel=None, cost_intangible=None,
                       transition='Energy performance', consumption='conventional'):
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

        transition: str, optional
        ['label', 'energy', 'label-energy']
        Define transition. Transition can be defined as label transition, energy transition, or label-energy transition.

        Returns
        -------
        pd.Series
        param: option - 'label-energy', 'label', 'energy'
        """

        lcc_final_seg = self._energy_lcc_final(energy_prices, transition, consumption=consumption)

        if cost_invest is not None:
            cost_invest_seg = reindex_mi(cost_invest, lcc_final_seg.index, cost_invest.index.names)
            cost_invest_seg = cost_invest_seg.reindex(lcc_final_seg.columns.get_level_values('{} final'.format(transition)),
                                                      axis=1)

        if cost_intangible is not None:
            levels = [lvl for lvl in lcc_final_seg.index.names if lvl in cost_intangible.index.names]
            cost_intangible = cost_intangible.reorder_levels(lcc_final_seg.index.names)
            cost_intangible_seg = reindex_mi(cost_intangible, lcc_final_seg.index, levels)
            cost_intangible_seg = cost_intangible_seg.reindex(
                lcc_final_seg.columns.get_level_values('{} final'.format(transition)),
                axis=1)
            cost_intangible_seg.fillna(0, inplace=True)

        if cost_switch_fuel is not None:
            cost_switch_fuel_seg = reindex_mi(cost_switch_fuel, lcc_final_seg.index, cost_switch_fuel.index.names)
            cost_switch_fuel_seg = cost_switch_fuel_seg.reindex(
                lcc_final_seg.columns.get_level_values('{} final'.format(transition)),
                axis=1)

        if transition == 'Energy performance':
            lcc_transition_seg = lcc_final_seg + cost_invest_seg
            if cost_intangible is not None:
                lcc_transition_seg = lcc_transition_seg + cost_intangible_seg
        elif transition == 'Heating energy':
            lcc_transition_seg = lcc_final_seg + cost_switch_fuel
        elif transition == ['Energy performance', 'Heating energy']:
            lcc_transition_seg = lcc_final_seg + cost_switch_fuel_seg
            if cost_intangible is not None:
                lcc_transition_seg = lcc_transition_seg + cost_intangible_seg

        # lcc_transition_seg = lcc_transition_seg.reorder_levels(self.segments.names)
        return lcc_transition_seg

    @staticmethod
    def lcc2market_share(lcc_df, nu=8):
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
            return pd.DataFrame((lcc_reverse_df.values.T / lcc_reverse_df.sum(axis=1).values).T,
                                index=lcc_reverse_df.index, columns=lcc_reverse_df.columns)
        elif isinstance(lcc_df, pd.Series):
            return lcc_reverse_df / lcc_reverse_df.sum()

    def to_market_share(self,
                        energy_prices,
                        transition='Energy performance',
                        consumption='conventional',
                        cost_invest=None,
                        cost_switch_fuel=None,
                        cost_intangible=None):

        lcc_final_seg = self.cost2lcc_final(energy_prices, cost_invest=cost_invest, cost_switch_fuel=cost_switch_fuel,
                                            cost_intangible=cost_intangible,
                                            transition=transition, consumption=consumption)

        ms = HousingStock.lcc2market_share(lcc_final_seg)
        ms.columns.names = ['{} final'.format(transition)]

        return ms, lcc_final_seg

    def to_pv(self,
              energy_prices,
              transition='Energy performance',
              consumption='conventional',
              cost_invest=None,
              cost_switch_fuel=None,
              cost_intangible=None):

        ms_final_seg, lcc_final_seg = self.to_market_share(energy_prices,
                                                           transition=transition,
                                                           consumption=consumption,
                                                           cost_invest=cost_invest,
                                                           cost_switch_fuel=cost_switch_fuel,
                                                           cost_intangible=cost_intangible)

        return (ms_final_seg * lcc_final_seg).sum(axis=1)

    def to_npv(self,
               energy_prices,
               transition='Energy performance',
               consumption='conventional',
               cost_invest=None,
               cost_switch_fuel=None,
               cost_intangible=None):

        energy_lcc_seg = self.to_energy_lcc(energy_prices, transition=transition, consumption=consumption)
        pv_seg = self.to_pv(energy_prices,
                            transition=transition,
                            consumption=consumption,
                            cost_invest=cost_invest,
                            cost_switch_fuel=cost_switch_fuel,
                            cost_intangible=cost_intangible)

        # energy_lcc_seg.sort_index(inplace=True)
        # pv_seg.sort_index(inplace=True)

        assert energy_lcc_seg.index.equals(pv_seg.index), 'Index should match'

        return energy_lcc_seg - pv_seg

    def calibration_market_share(self, energy_prices, market_share_objective, folder_output=None, cost_invest=None,
                                 consumption='conventional'):
        """Returns intangible costs by calibrating market_share.

        TODO: Calibration of intangible cost could be based on absolute value instead of market share.
        """

        lcc_final = self.cost2lcc_final(energy_prices, cost_invest=cost_invest,
                                     transition='Energy performance', consumption=consumption)

        # remove idx when label = 'A' (no transition) and label = 'B' (intangible_cost = 0)
        lcc_useful = remove_rows(lcc_final, 'Energy performance', 'A')
        lcc_useful = remove_rows(lcc_useful, 'Energy performance', 'B')

        market_share_temp = HousingStock.lcc2market_share(lcc_useful)
        market_share_objective = reindex_mi(market_share_objective, market_share_temp.index, market_share_objective.index.names)
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

            def func(intangible_cost_np, lcc, ms, factor):
                """Functions of intangible_cost that are equal to 0.

                Returns a vector that should converge toward 0 as intangible cost converge toward optimal.
                """
                result = np.empty(lcc.shape[0])
                market_share_np = (lcc + intangible_cost_np ** 2) ** -parameters_dict['nu_intangible_cost'] / np.sum(
                    (lcc + intangible_cost_np ** 2) ** -parameters_dict['nu_intangible_cost'])
                result[:-1] = market_share_np[:-1] - ms[:-1]
                result[-1] = np.sum(intangible_cost_np ** 2) / np.sum(lcc + intangible_cost_np ** 2) - factor
                return result

            x0 = lcc_np * ini
            root, info_dict, ier, message = fsolve(func, x0, args=(lcc_np, ms_obj, factor), full_output=True)

            if ier == 1:
                return ier, root

            else:
                # logging.debug(message)
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

        assert len(lcc_useful.index) == len(idx_list), "Calibration didn't work for all segments"

        if folder_output is not None:
            intangible_cost.to_pickle(os.path.join(folder_output, 'intangible_cost.pkl'))
        # logging.debug('Average lambda factor: {:.0f}%'.format(sum(lambda_list) / len(lambda_list) * 100))
        intangible_cost_mean = intangible_cost.groupby('Energy performance', axis=0).mean()
        intangible_cost_mean = intangible_cost_mean.loc[intangible_cost_mean.index[::-1], :]
        # logging.debug('Intangible cost (€/m2): \n {}'.format(intangible_cost_mean))
        return intangible_cost

    def to_segments_construction(self, lvl2drop, new_lvl_dict):
        """Returns segments_new from segments.

        Segments_new doesn't get Income class owner at first and got other Energy performance value.
        """
        levels_wo = [lvl for lvl in self._levels if lvl not in lvl2drop]
        segments_new = get_levels_values(self._segments, levels_wo)

        for level in new_lvl_dict:
            segments_new = segments_new.droplevel(level)
            segments_new = segments_new.drop_duplicates()
            segments_new = pd.concat(
                [pd.Series(index=segments_new, dtype='float64')] * len(new_lvl_dict[level]),
                keys=new_lvl_dict[level], names=list(new_lvl_dict.keys()))
            segments_new = segments_new.reorder_levels(levels_wo).index
        return segments_new


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
    def __init__(self, stock_seg, levels_values, year, residual_rate,
                 label2area=None,
                 label2horizon_envelope=None,
                 label2horizon_heater=None,
                 label2discount=None,
                 label2income=None,
                 label2consumption=None):

        super().__init__(stock_seg, levels_values, year,
                         label2area=label2area,
                         label2horizon_envelope=label2horizon_envelope,
                         label2horizon_heater=label2horizon_heater,
                         label2discount=label2discount,
                         label2income=label2income,
                         label2consumption=label2consumption)

        self._calibration_year = year
        self.residual_rate = residual_rate

        # slave stock of stock_seg property
        self._stock_seg_mobile = stock_seg * (1 - residual_rate)
        self._stock_seg_mobile_dict = {year: stock_seg * (1 - residual_rate)}
        self._stock_seg_residual = stock_seg * residual_rate
        self._stock_area_seg = self.numbers2area()

        # initializing knowledge
        self.flow_area_renovated_seg = self.flow_area_renovated_seg_ini()
        self._flow_knowledge_ep = self.to_flow_knowledge()
        self._stock_knowledge_ep = self.to_flow_knowledge()
        self._stock_knowledge_ep_dict = {year: self._stock_knowledge_ep}
        self._knowledge = self._stock_knowledge_ep / self._stock_knowledge_ep

        # share of decision-maker in the total stock
        self._dm_share_tot = stock_seg.groupby(['Occupancy status', 'Housing type']).sum() / stock_seg.sum()

        # calibration
        self._rho_seg = pd.Series()

    @property
    def stock_seg(self):
        return self._stock_seg

    @stock_seg.setter
    def stock_seg(self, new_stock_seg):
        """Master stock that implement modification for stock slave.
        """
        self._segments = new_stock_seg.index

        self._stock_seg = new_stock_seg
        self._stock_seg_dict[self._year] = new_stock_seg
        self._stock_seg_mobile = self.stock_seg * (1 - self.residual_rate)
        self._stock_seg_mobile_dict[self._year] = self._stock_seg_mobile
        self._stock_seg_residual = self.stock_seg * self.residual_rate
        self._stock_area_seg = self.numbers2area()

    @property
    def flow_knowledge_ep(self):
        return self._flow_knowledge_ep

    @flow_knowledge_ep.setter
    def flow_knowledge_ep(self, new_flow_knowledge_ep):
        self._flow_knowledge_ep = new_flow_knowledge_ep
        self._stock_knowledge_ep = self._stock_knowledge_ep + new_flow_knowledge_ep
        self._stock_knowledge_ep_dict[self._year] = self._stock_knowledge_ep
        self._knowledge = self._stock_knowledge_ep / self._stock_knowledge_ep_dict[self._calibration_year]

    @property
    def knowledge(self):
        return self._knowledge

    def flow_area_renovated_seg_ini(self):
        """Initialize flow area renovated.

        Flow area renovation is defined as:
         renovation rate (2.7%/yr) x number of learning years (10 yrs) x renovated area (m2).
        """
        renovation_rate_dm = reindex_mi(calibration_dict['renovation_rate_decision_maker'], self._stock_area_seg.index,
                                        calibration_dict['renovation_rate_decision_maker'].index.names)
        return renovation_rate_dm * self._stock_area_seg * technical_progress_dict['learning_year']

    @staticmethod
    def renovate_rate_func(lcc, rho, parameters_dict):
        if isinstance(rho, pd.Series):
            rho_f = rho.loc[tuple(lcc.iloc[:-1].tolist())]
        else:
            rho_f = rho

        if np.isnan(rho_f):
            return float('nan')
        else:
            return logistic(lcc.loc[0] - parameters_dict['npv_min'],
                            a=parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1,
                            r=rho_f,
                            k=parameters_dict['rate_max'])

    def to_renovation_rate(self, energy_prices,
                           transition='Energy performance',
                           consumption='conventional',
                           cost_invest=None,
                           cost_switch_fuel=None,
                           cost_intangible=None):

        """Routine calculating renovation rate from segments for a particular yr.

        Cost (energy, investment) & rho parameter are also required.
        """
        npv_seg = self.to_npv(energy_prices,
                              transition=transition,
                              consumption=consumption,
                              cost_invest=cost_invest,
                              cost_switch_fuel=cost_switch_fuel,
                              cost_intangible=cost_intangible)
        renovation_rate_seg = npv_seg.reset_index().apply(HousingStockRenovated.renovate_rate_func,
                                                          args=[self._rho_seg, parameters_dict], axis=1)
        renovation_rate_seg.index = npv_seg.index
        return renovation_rate_seg

    def to_flow_renovation_label(self, energy_prices,
                                 consumption='conventional',
                                 cost_invest=None,
                                 cost_intangible=None):

        renovation_rate_seg = self.to_renovation_rate(energy_prices,
                                                      transition='Energy performance',
                                                      consumption=consumption,
                                                      cost_invest=cost_invest,
                                                      cost_intangible=cost_intangible)

        flow_renovation_seg = renovation_rate_seg * self.stock_seg

        market_share_seg_ep = self.to_market_share(energy_prices,
                                                   transition='Energy performance',
                                                   consumption=consumption,
                                                   cost_invest=cost_invest,
                                                   cost_intangible=cost_intangible)[0]

        flow_renovation_seg_ep = ds_mul_df(flow_renovation_seg, market_share_seg_ep)
        return flow_renovation_seg_ep

    def to_flow_renovation_label_energy(self, energy_prices, consumption='conventional',
                                        cost_switch_fuel=None,
                                        cost_invest=None,
                                        cost_intangible=None):
        """De-aggregate stock_renovation_label by final heating energy.

        stock_renovation columns segmented by final label and final heating energy.

        Parameters
        ----------
        energy_prices: pd.DataFrame

        cost_switch_fuel: pd.DataFrame

        cost_invest: pd.DataFrame

        cost_intangible: pd.DataFrame

        consumption: str

        Returns
        -------
        pd.DataFrame
        """

        market_share_seg_he = self.to_market_share(energy_prices,
                                                   transition='Heating energy',
                                                   consumption=consumption,
                                                   cost_switch_fuel=cost_switch_fuel)[0]

        ms_temp = pd.concat([market_share_seg_he.T] * len(self.levels_values['Energy performance']),
                            keys=self.levels_values['Energy performance'], names=['Energy performance final'])

        flow_renovation_seg_label = self.to_flow_renovation_label(energy_prices,
                                                                  consumption=consumption,
                                                                  cost_invest=cost_invest,
                                                                  cost_intangible=cost_intangible)

        sr_temp = pd.concat([flow_renovation_seg_label.T] * len(self.levels_values['Heating energy']),
                            keys=self.levels_values['Heating energy'], names=['Heating energy final'])
        flow_renovation_label_energy = (sr_temp * ms_temp).T
        return flow_renovation_label_energy

    def to_flow_remained(self, energy_prices, consumption='conventional', cost_switch_fuel=None, cost_invest=None,
                         cost_intangible=None):
        """Calculate flow_remained for each segment.

        Returns: positive (+) flow for buildings segment reached by the renovation (final state),
                 negative (-) flow for buildings segment (initial state) that have been renovated.
        """

        flow_renovation_label_energy_seg = self.to_flow_renovation_label_energy(energy_prices,
                                                                                consumption=consumption,
                                                                                cost_switch_fuel=cost_switch_fuel,
                                                                                cost_invest=cost_invest,
                                                                                cost_intangible=cost_intangible)

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
        return flow_remained_seg, flow_area_renovation_seg

    def to_flow_demolition(self, destruction_rate=0.0):
        flow_demolition = self._stock_seg * destruction_rate
        flow_demolition_seg_dm = self._dm_share_tot * flow_demolition
        # flow_area_demolition_seg_dm = flow_demolition_seg_dm * self.label2area
        return flow_demolition_seg_dm

    def to_flow_seg_demolition(self, destruction_rate=0.0):
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

        flow_demolition_seg_dm = self.to_flow_demolition(destruction_rate=destruction_rate)

        stock_mobile = self._stock_seg_dict[self._year - 1]
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
                        worst_lbl_dict[segment] = lbl
                        break
            return worst_lbl_idx, worst_lbl_dict

        worst_label_idx, worst_label_dict = worst_label(segments_mobile, stock_mobile)

        # we know type_demolition, then we calculate nb_housing_demolition_ini based on proportion of stock remaining
        levels = ['Occupancy status', 'Housing type']
        levels_wo_performance = [lvl for lvl in self._levels if lvl != 'Energy performance']
        stock_remaining_woperformance = self._stock_seg.groupby(levels_wo_performance).sum()
        prop_housing_remaining_decision = val2share(stock_remaining_woperformance, levels)

        type_housing_demolition_reindex = reindex_mi(flow_demolition_seg_dm,
                                                     prop_housing_remaining_decision.index, levels)
        type_housing_demolition_wo_performance = type_housing_demolition_reindex * prop_housing_remaining_decision
        np.testing.assert_almost_equal(flow_demolition_seg_dm.sum(), type_housing_demolition_wo_performance.sum(),
                                       err_msg='Not normal')

        # we don't have the information about which labels are going to be demolition first
        prop_stock_worst_label = stock_mobile.loc[worst_label_idx] / stock_mobile_ini.loc[
            worst_label_idx]
        nb_housing_demolition_ini = reindex_mi(type_housing_demolition_wo_performance, prop_stock_worst_label.index,
                                               levels_wo_performance)

        # initialize nb_housing_demolition_theo for worst label based on how much have been demolition so far
        nb_housing_demolition_theo = prop_stock_worst_label * nb_housing_demolition_ini

        # we year with the worst label and we stop when nb_housing_demolition_theo == 0
        flow_demolition = pd.Series(0, index=stock_mobile.index, dtype='float64')
        for segment in segments_mobile:
            label = worst_label_dict[segment]
            num = self.levels_values['Energy performance'].index(label)
            idx_tot = (segment[0], segment[1], label, segment[2], segment[3], segment[4])

            while nb_housing_demolition_theo.loc[idx_tot] != 0:
                # stock_demolition cannot be sup to stock_mobile and to nb_housing_demolition_theo
                flow_demolition.loc[idx_tot] = min(stock_mobile.loc[idx_tot], nb_housing_demolition_theo.loc[idx_tot])
                if label != 'A':
                    num += 1
                    label = self.levels_values['Energy performance'][num]
                    labels = self.levels_values['Energy performance'][:num + 1]
                    idx = (segment[0], segment[1], segment[2], segment[3], segment[4])
                    idx_tot = (segment[0], segment[1], label, segment[2], segment[3], segment[4])
                    idxs_tot = [(segment[0], segment[1], label, segment[2], segment[3], segment[4]) for label in labels]

                    # nb_housing_demolition_theo is the remaining number of housing that need to be demolition for this segment
                    nb_housing_demolition_theo[idx_tot] = type_housing_demolition_wo_performance.loc[idx] - \
                                                          flow_demolition.loc[idxs_tot].sum()

                else:
                    nb_housing_demolition_theo[idx_tot] = 0

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

    def _calibration_renovation_rate(self, energy_prices, renovation_rate_obj, consumption='conventional',
                                     cost_invest=None, cost_intangible=None):
        # TODO: rho_seg must be determine for all future indexes
        # TODO: weight_dm doesn't make a lot of sense
        npv_df = self.to_npv(energy_prices,
                             transition='Energy performance',
                             consumption=consumption,
                             cost_invest=cost_invest,
                             cost_intangible=cost_intangible)

        renovation_rate_obj = reindex_mi(renovation_rate_obj, npv_df.index, renovation_rate_obj.index.names)
        rho = (np.log(parameters_dict['rate_max'] / parameters_dict['rate_min'] - 1) - np.log(
            parameters_dict['rate_max'] / renovation_rate_obj - 1)) / (npv_df - parameters_dict['npv_min'])

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

    def make_calibration_renovation(self, energy_prices, renovation_rate_obj, consumption='conventional',
                                    cost_invest=None,
                                    cost_intangible=None):
        """Calculate rho_seg and apply it as attribute.
        """

        rho_seg = self._calibration_renovation_rate(energy_prices, renovation_rate_obj,
                                                    consumption=consumption,
                                                    cost_invest=cost_invest,
                                                    cost_intangible=cost_intangible)
        self._rho_seg = rho_seg

    @staticmethod
    def information_rate_func(knowledge, rate_max, alpha):
        """Returns information rate. More info_rate is high, more intangible_cost are low.
        Intangible renovation costs decrease according to a logistic curve with the same cumulative
        production so as to capture peer effects and knowledge diffusion.
        intangible_cost[yr] = intangible_cost[calibration_year] * info_rate with info rate [1-info_rate_max ; 1]
        This function calibrate a logistic function, so rate of decrease is set at  25% for a doubling of cumulative
        production.
        """

        def equations(p, sh=rate_max, alpha=alpha):
            a, r = p
            return (1 + a * np.exp(-r)) ** -1 - sh, (1 + a * np.exp(-2 * r)) ** -1 - sh - (1 - alpha) * sh + 1

        a, r = fsolve(equations, (1, -1))

        return logistic(knowledge, a=a, r=r) + 1 - rate_max


class HousingStockConstructed(HousingStock):
    def __init__(self, stock, levels_values, year,
                 label2area=None,
                 label2horizon=None,
                 label2discount=None,
                 label2income=None,
                 label2consumption=None):
        super().__init__(stock, levels_values, year,
                         label2area=label2area,
                         label2horizon=label2horizon,
                         label2discount=label2discount,
                         label2income=label2income,
                         label2consumption=label2consumption)


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








