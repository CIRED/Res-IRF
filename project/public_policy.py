import pandas as pd
import os
import json

from function_pandas import *


class PublicPolicy:
    """Public policy can be subsidies, taxes, or regulation.

    Loan backed by the state are represented as subsidies.
    """

    def __init__(self, name, start, end, kind, values):
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

        list_kind = ['subsidies', 'taxes', 'regulation', 'subsidies_loan', 'other']
        if kind in list_kind:
            self.kind = kind
        else:
            raise AttributeError("Kind attribute must be included in {}".format([list_kind]))

        if isinstance(values, float):
            self.values = pd.Series(values, index=range(start, end + 1, 1))
        elif isinstance(values, pd.Series):
            self.values = values.loc[start:end]
        elif isinstance(values, pd.DataFrame):
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

    def apply_policies(self, costs):
        """Multiply costs by self.values whatever types of costs and self.values.

        # TODO: test with apply costs and remove policies.

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

        Returns
        -------
        pd.Series, or pd.DataFrame
        costs * self.values
        """

        # costs_before = costs.loc[:self.start]
        # costs_after = costs.loc[self.end + 1:]
        if isinstance(costs, pd.Series):
            costs = costs.loc[self.start:self.end + 1]
        elif isinstance(costs, pd.DataFrame):
            costs = costs.loc[:, self.start:self.end + 1]

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

    def remove_policies(self, costs):
        if not self.targeted:
            if isinstance(costs, pd.Series):
                return costs / (1 + self.values)
            elif isinstance(costs, pd.DataFrame):
                return ds_mul_df((1 + self.values) ** -1, costs, option='columns')
            else:
                raise ValueError("Costs must be type Series or DataFrame")
        elif self.targeted:
            levels_shared = [lvl for lvl in self.values.index.names if lvl in costs.index.names]
            if len(levels_shared) != len(self.values.index.names):
                print('Costs parameter is not enough segmented to match PublicPolicy targets')

            values_reindex = reindex_mi(self.values, costs.index, levels_shared)
            return costs * (1 + values_reindex) ** -1


class Subsidies(PublicPolicy):
    def __init__(self, name, start, end, kind, values):
        super().__init__(name, start, end, kind, values)

    def apply_subsidies(self, costs):
        discount = self.apply_policies(costs)
        return discount, costs - discount


class Taxes(PublicPolicy):

    def __init__(self, name, start, end, kind, values):
        super().__init__(name, start, end, kind, values)

    def apply_taxes(self, costs):
        taxes = self.apply_policies(costs)
        return taxes, costs + taxes


class Regulation(PublicPolicy):
    def __init__(self, name, start, end, kind, values):
        super().__init__(name, start, end, kind, values)

    def constraint_renovation(self, segment_targeted):
        pass

    def constraint_construction(self):
        pass


class SubsidiesLoan(Subsidies):
    """Loan backed by the state.

    Instead of paying interest_rate (%/yr)
    """

    def __init__(self, name, start, end, kind, values):
        super().__init__(name, start, end, kind, values)

    def loan2subsidies(self, interest_rate):
        if isinstance(interest_rate, float):
            interest_rate = pd.Series(interest_rate, index=range(self.start, self.end + 1, 1))
        elif isinstance(interest_rate, pd.Series):
            interest_rate = interest_rate.loc[self.start:self.end]

        # TODO: from interest_rate 2 subsidies


if __name__ == '__main__':

    folder_test = os.path.join(os.getcwd(), 'tests', 'input')
    folder_input = os.path.join(os.getcwd(), 'input')

    name_file = os.path.join(folder_input, 'policies_input.json')
    with open(name_file) as f:
        policy_input = json.load(f)

    test_cost_initial_final = pd.read_csv(os.path.join(folder_test, 'test_cost_initial_final.csv'), index_col=[0], header=[0])
    test_cost_segmented = pd.read_csv(os.path.join(folder_test, 'test_cost_segmented.csv'), index_col=[0, 1, 2], header=[0], squeeze=True)
    test_cost_years = pd.read_csv(os.path.join(folder_test, 'test_cost_years.csv'), index_col=[0], header=[0]).T
    test_parc = pd.read_csv(os.path.join(folder_test, 'test_parc.csv'), index_col=[0, 1, 2, 3, 4, 5], header=[0], squeeze=True)

    carbon_tax = Taxes('vta', 2014, 2030, 'taxes', 0.1)
    cost_carbon_tax, cost_wtax = carbon_tax.apply_taxes(test_cost_years)

    scenario_eptz = 'normal'
    if scenario_eptz == 'normal':
        eptz = Taxes('vta', 2014, 2030, 'subsidies', 0.09)
    elif scenario_eptz == '+':
        eptz = Taxes('vta', 2014, 2030, 'subsidies', 0.23)

    scenario_cite = 'not-targeted'
    if scenario_cite == 'not-targeted':
        cite = Subsidies('cite', 2012, 2020, 'subsidies', 0.17)
    elif scenario_cite == 'targeted':
        cite = Subsidies('cite', 2012, 2100, 'subsidies', pd.DataFrame(0.17))

    # cee_taxes = Taxes()


    print('pause')
    print('pause')