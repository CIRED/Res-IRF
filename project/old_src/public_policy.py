import os
import json

import nump
from utils import *
import numpy_financial as npf
# npf.irr
# npf.npv


class PublicPolicy:
    """Public policy can be subsidies, taxes, or regulation.

    Loan backed by the state are represented as subsidies.
    """

    def __init__(self, name, start, end, kind, values=None):
        """Define public policies.

        Parameters
        ----------
        name : str
        
        start: int
        First year public policy applies.

        end: int
        Last year (included) public policy applies.

        kind: str
        Kind can be ['subsidies', 'taxes', 'regulation', 'subsidies_loan', 'other']

        values: float, pd.Series, pd.DataFrame
        If public policy depends on:
        - nothing: float,
        - years: pd.Series,
        - years and segments (i.e. households): pd.DataFrame (index = segments, columns = years)
        values cannot depend only on segments for now.

        Returns
        -------
        PublicPolicy object
        """
        self.name = name
        self.start = start
        self.end = end

        list_kind = ['subsidies', 'energy_taxes', 'regulation', 'subsidies_loan', 'other']
        if kind in list_kind:
            self.kind = kind
        else:
            raise AttributeError("Kind attribute must be included in {}".format([list_kind]))

        self.values = None
        if values is not None:
            if isinstance(values, float):
                self.values = pd.Series(values, index=range(start, end, 1))
            elif isinstance(values, pd.Series):
                start = max(start, min(values.index))
                end = min(end, max(values.index))
                self.values = values.loc[start:end]
            elif isinstance(values, pd.DataFrame):
                start = max(start, min(values.columns))
                end = min(end, max(values.columns))
                self.values = values.loc[:, start:end]
            else:
                raise AttributeError(
                    "Values attribute must be of type float, Series, or DataFrame. \n {} instead".format(type(values)))

        self.targeted = False
        self.targets = None
        if isinstance(self.values, pd.DataFrame):
            self.targeted = True
            self.targets = self.values.index.names

    def __repr__(self):
        return 'Public policy: {} \n Kind: {} \n Starting: {} \n Ending: {} \n Targeted: {}'.format(self.name,
                                                                                                    self.kind,
                                                                                                    self.start,
                                                                                                    self.end,
                                                                                                    self.targeted)

    def values_by_cost(self, costs, yr=None, costs_max=None):
        """Multiply costs by self.values whatever types of costs and self.values.

        Parameters
        ----------
        self : PublicPolicy
        if self.kind == 'taxes': apply_cost (=costs * taxes.values) represents value of the tax.values.
        elif self.kind == 'subsidies': apply_cost (=costs * subsidies) represents the discount.
        if self.targeted - behavior depends on households - reindex is needed.

        costs: pd.Series, pd.DataFrame
        if costs is pd.Series: it must be a year-time series.
        if costs is pd.DataFrame: index must be segments, and columns year-time index.
        costs year-time series must include PublicPolicy application.

        yr: int, optional

        costs_max: , optional

        Returns
        -------
        pd.Series, or pd.DataFrame
        costs * self.values
        """

        costs[costs > costs_max] = costs_max

        # costs depends on year
        if yr is None:

            if isinstance(costs, pd.Series):
                start = max(self.start, min(costs.index))
                end = min(self.end, max(costs.index))
                costs = costs.loc[start:end]
            elif isinstance(costs, pd.DataFrame):
                start = max(self.start, min(costs.columns))
                end = min(self.end, max(costs.columns))
                costs = costs.loc[:, start:end]

            if not self.targeted:
                if isinstance(costs, pd.Series):
                    return costs * self.values
                elif isinstance(costs, pd.DataFrame):
                    return ds_mul_df(self.values, costs, option='rows')
                else:
                    raise ValueError("Costs must be type Series or DataFrame")
            elif self.targeted:
                levels_shared = [lvl for lvl in self.values.index.names if lvl in costs.index.names]
                if len(levels_shared) != len(self.values.index.names):
                    print('Costs parameter is not enough segmented')
                values_reindex = reindex_mi(self.values, costs.index, levels_shared)
                return costs * values_reindex

        # costs doesn't depend on year >>> values = values.loc[yr]
        else:
            if not self.targeted:
                return costs * self.values.loc[yr]
            elif self.targeted:
                levels_shared = [lvl for lvl in self.values.index.names if lvl in costs.index.names]
                if len(levels_shared) != len(self.values.index.names):
                    print('Costs parameter is not enough segmented')
                values_reindex = reindex_mi(self.values.loc[:, yr], costs.index, levels_shared)
                return costs * values_reindex

    def remove_policies(self, costs, yr=None):
        if yr is None:
            if isinstance(costs, pd.Series):
                costs = costs.loc[self.start:self.end + 1]
            elif isinstance(costs, pd.DataFrame):
                costs = costs.loc[:, self.start:self.end + 1]

            if not self.targeted:
                if isinstance(costs, pd.Series):
                    return costs * (self.values ** -1)
                elif isinstance(costs, pd.DataFrame):
                    return ds_mul_df(self.values ** -1, costs, option='rows')
                else:
                    raise ValueError("Costs must be type Series or DataFrame")
            elif self.targeted:
                levels_shared = [lvl for lvl in self.values.index.names if lvl in costs.index.names]
                if len(levels_shared) != len(self.values.index.names):
                    print('Costs parameter is not enough segmented')
                values_reindex = reindex_mi(self.values ** -1, costs.index, levels_shared)
                return costs * values_reindex

        else:
            if not self.targeted:
                return costs * (self.values ** -1).loc[yr]
            elif self.targeted:
                levels_shared = [lvl for lvl in self.values.index.names if lvl in costs.index.names]
                if len(levels_shared) != len(self.values.index.names):
                    print('Costs parameter is not enough segmented')
                values_reindex = reindex_mi(self.values ** -1, costs.index, levels_shared)
                return costs * values_reindex


class Subsidies(PublicPolicy):
    """
    Subsidies apply for a specific transition.
    """
    def __init__(self, name, start, end, kind, values, cost, unit='percent'):
        super().__init__(name, start, end, kind, values)
        self.cost = cost
        self.unit = unit
        # unit must be 'percent' or 'absolute'


    def apply_subsidies(self, costs, costs_max=None):
        if self.unit == 'percent':
            discount = self.values_by_cost(costs, costs_max=costs_max)
            return discount, costs - discount
        elif self.unit == 'absolute':
            pass


class EnergyTaxes(PublicPolicy):
    def __init__(self, name, start, end, kind, values):
        super().__init__(name, start, end, kind, values)

    def apply_taxes(self, energy_price):
        taxes = self.values_by_cost(energy_price)
        return taxes, energy_price + taxes


class Regulation(PublicPolicy):
    def __init__(self, name, start, end, kind, values):
        super().__init__(name, start, end, kind, values)

    def constraint_renovation(self, segment_targeted):
        pass

    def constraint_construction(self):
        pass


class RegulatedLoan(PublicPolicy):
    """Loan backed by the state.

    Instead of paying interest_rate (%/yr).
    Example: EPTZ
    """

    def __init__(self, name, start, end, kind, ir_regulated=0.0, principal_max=None, horizon_max=None):
        super().__init__(name, start, end, kind)
        self.ir_regulated = ir_regulated
        self.horizon_max = horizon_max
        self.principal_max = principal_max

    @staticmethod
    def interest_cost(interest_rate, n_period, principal):
        period = np.arange(n_period) + 1
        return - npf.ipmt(interest_rate, period, n_period, principal).sum()

    def loan2subsidary(self, ir_market, horizon, principal):
        """Every parameter must be the same shape.
        """
        ir_market = np.full(principal.shape, ir_market)
        horizon = np.full(principal.shape, horizon)

        if isinstance(principal, np.ndarray):
            principal[principal > self.principal_max] = self.principal_max
            vfunc = np.vectorize(RegulatedLoan.interest_cost)
            ic_market = vfunc(ir_market, horizon, principal)
        else:
            principal = min(self.principal_max, principal)
            ic_market = RegulatedLoan.interest_cost(ir_market, horizon, principal)

        if self.ir_regulated == 0:
            ic_regulated = 0
        else:
            vfunc = np.vectorize(RegulatedLoan.interest_cost)
            ic_regulated = vfunc(self.ir_regulated, horizon, principal)
        opportunity_cost = ic_market - ic_regulated

        return Subsidies(self.name, self.start, self.end, self.kind, opportunity_cost)


if __name__ == '__main__':
    horizon = 30
    principal = 2500

    folder_test = os.path.join(os.getcwd(), '../tests', 'input')
    folder_input = os.path.join(os.getcwd(), '../input')

    name_file = os.path.join(folder_input, 'policies_input.json')
    with open(name_file) as f:
        policy_input = json.load(f)

    test_cost_initial_final = pd.read_csv(os.path.join(folder_test, 'test_cost_initial_final.csv'), index_col=[0],
                                          header=[0])
    test_cost_initial_final.index.set_names('Energy performance', inplace=True)
    test_cost_initial_final.columns.set_names('Energy performance final', inplace=True)

    test_cost_segmented = pd.read_csv(os.path.join(folder_test, 'test_cost_segmented.csv'), index_col=[0, 1, 2],
                                      header=[0], squeeze=True)
    test_energy_price = pd.read_csv(os.path.join(folder_test, 'test_cost_years.csv'), index_col=[0], header=[0]).T
    test_parc = pd.read_csv(os.path.join(folder_test, 'test_parc.csv'), index_col=[0, 1, 2, 3, 4, 5], header=[0],
                            squeeze=True)

    p = PublicPolicy('test', 2015, 2030, 'other', values=0.1)

    year = 2015
    if p.start <= year < p.end:
        subsidies = p.values_by_cost(test_cost_segmented.copy(), yr=year, costs_max=1000)
        test_cost_segmented = test_cost_segmented - subsidies

    eptz = RegulatedLoan('eptz', 2014, 2030, 'other', ir_regulated=0.0, principal_max=1000)
    # a = eptz.loan2subsidary(0.04, 20, np.array([[1500, 1200, 500, 800, 1000], [100, 200, 100, 200, 100]]))

    carbon_tax = EnergyTaxes('vta', 2014, 2030, 'energy_taxes', 0.1)
    cost_carbon_tax, cost_wtax = carbon_tax.apply_taxes(test_energy_price)



    scenario_eptz = 'normal'
    if scenario_eptz == 'normal':
        eptz = EnergyTaxes('vta', 2014, 2030, 'subsidies', 0.09)
    elif scenario_eptz == '+':
        eptz = EnergyTaxes('vta', 2014, 2030, 'subsidies', 0.23)

    scenario_cite = 'not-targeted'
    if scenario_cite == 'not-targeted':
        cite = Subsidies('cite', 2012, 2020, 'subsidies', 0.17)
    elif scenario_cite == 'targeted':
        cite = Subsidies('cite', 2012, 2100, 'subsidies', pd.DataFrame(0.17))

    # cee_taxes = Taxes()


    print('pause')
    print('pause')