from utils import reindex_mi
import numpy as np
import numpy_financial as npf
import pandas as pd


class PublicPolicy:
    def __init__(self, name, start, end, policy):
        self.name = name
        self.start = start
        self.end = end
        self.policy = policy


class EnergyTaxes(PublicPolicy):
    def __init__(self, name, start, end, kind, value):
        """
        value: pd.DataFrame
        indexes are energy, and columns are years
        """
        super().__init__(name, start, end, 'energy_taxes')

        self.list_kind = ['%', '€/kWh', '€/gCO2']
        self.kind = kind
        self.value = value

    def price_to_taxes(self, energy_prices=None, co2_content=None):

        if self.kind == '%':
            return energy_prices * self.value

        elif self.kind == '€/kWh':
            return self.value

        elif self.kind == '€/gCO2':
            # €/tCO2 * gCO2/kWh / 1000000 -> €/kWh
            taxes = self.value * co2_content
            taxes.fillna(0, inplace=True)
            taxes = taxes.reindex(energy_prices.columns, axis=1)
            return energy_prices * taxes

        else:
            raise AttributeError


class Subsidies(PublicPolicy):
    def __init__(self, name, start, end, kind, value,
                 transition=None, targets=None, cost_max=None, subsidy_max=None):
        super().__init__(name, start, end, 'subsidies')

        if transition is None:
            self.transition = ['Energy performance']
        else:
            self.transition = transition

        # self.targets = targets
        self.list_kind = ['%', '€/kWh', '€/tCO2', '€']
        self.kind = kind
        self.cost_max = cost_max
        self.subsidy_max = subsidy_max
        self.value = value

    def to_subsidy(self, cost=None, energy_saving=None, co2_saving=None):

        if self.kind == '€':
            return self.value
        if self.kind == '%':
            # subsidy apply to one target
            if isinstance(self.value, pd.Series):
                val = reindex_mi(self.value, cost.index, self.value.index.names, axis=0)
            else:
                val = self.value
            # subsidy apply to a maximum cost
            if self.cost_max is not None:
                cost[cost > self.cost_max] = cost

            return val * cost
        if self.kind == '€/kWh':
            return self.value * energy_saving


class RegulatedLoan(PublicPolicy):
    """Loan backed by the state to buy technology capex.

    Instead of paying interest rate at ir_market (%/yr), households only pays ir_regulated.

    Attributes
    __________
    name: str
    start: int
    end: int
    ir_regulated: float or pd.Series, optional
    ir_market: float or pd.Series, optional
    principal_max: float or pd.Series, optional
    horizon: int or pd.Series, optional
    horizon_max: int or pd.Series, optional
    horizon_min: int or pd.Series, optional
    targets: pd.DataFrame, optional

    Methods
    _______
    interest_cost: calculate interest_cost

    loan_approximate2subsidy(principal): calculate subsidies as opportunity cost for the household for a principal.
    principal is a capex from a initial state to a final state.

    Examples
    --------
    eptz = RegulatedLoan('test', 2018, 2030,
                        ir_regulated=None, ir_market=0.05,
                        principal_max=1000,
                        horizon=10)
    >>> eptz.to_opportunity_cost(1000)

    >>> eptz.to_opportunity_cost(2000)

    """

    def __init__(self, name, start, end, ir_regulated=None, ir_market=None,
                 principal_max=None, principal_min=None,
                 horizon=None, targets=None, transition=None):
        super().__init__(name, start, end, 'regulated_loan')

        if transition is None:
            self.transition = ['Energy performance']
        else:
            self.transition = transition

        self.ir_regulated = ir_regulated
        self.ir_market = ir_market
        self.horizon = horizon
        self.principal_min = principal_min
        self.principal_max = principal_max
        self.targets = targets

    def reindex_attributes(self, mi_index):

        if isinstance(self.ir_regulated, float) or isinstance(self.ir_regulated, int):
            self.ir_regulated = pd.Series(self.ir_regulated, index=mi_index)
        elif isinstance(self.ir_regulated, pd.Series):
            self.ir_regulated = reindex_mi(self.ir_regulated, mi_index, self.ir_regulated.index.names)

        if isinstance(self.ir_market, float) or isinstance(self.ir_market, int):
            self.ir_market = pd.Series(self.ir_market, index=mi_index)
        elif isinstance(self.ir_market, pd.Series):
            self.ir_market = reindex_mi(self.ir_market, mi_index, self.ir_market.index.names)

        if isinstance(self.horizon, float) or isinstance(self.horizon, int):
            self.horizon = pd.Series(self.horizon, index=mi_index)
        elif isinstance(self.horizon, pd.Series):
            self.horizon = reindex_mi(self.horizon, mi_index, self.horizon.index.names)

        if isinstance(self.principal_min, float) or isinstance(self.principal_min, int):
            self.principal_min = pd.Series(self.principal_min, index=mi_index)
        elif isinstance(self.principal_min, pd.Series):
            self.principal_min = reindex_mi(self.principal_min, mi_index, self.principal_min.index.names)

        if isinstance(self.principal_max, float) or isinstance(self.principal_max, int):
            self.principal_max = pd.Series(self.principal_max, index=mi_index)
        elif isinstance(self.principal_max, pd.Series):
            self.principal_max = reindex_mi(self.principal_max, mi_index, self.principal_max.index.names)

    @staticmethod
    def interest_cost(interest_rate, n_period, principal):
        """Returns total interest cost.

        Parameters
        ----------
        interest_rate: float
            loan interest rate (%/period)
        n_period: float
            number of periods
        principal: float
            loan amount

        Returns
        -------
        float

        Examples
        ________
        >>> interest_cost(0.1, 1, 100)
        10
        >>> interest_cost(0.1, 10, 100000)

        """
        period = np.arange(n_period) + 1
        return - npf.ipmt(interest_rate, period, n_period, principal).sum()

    def apply_targets(self, principal):
        """Returns principal for transition that are targeted by the regulated loan.

        Parameters
        ----------
        self.targets: pd.DataFrame
            1 when targeted, otherwise 0
        principal: pd.DataFrame
            capex of a transition from row to column

        Returns
        -------
        pd.DataFrame
        """
        targets = reindex_mi(self.targets, principal.index, self.targets.index.names)
        targets = reindex_mi(targets, principal.columns, self.targets.columns.names, axis=1)
        targets.fillna(0, inplace=True)
        principal = principal * targets
        principal.replace(0, np.nan, inplace=True)
        principal.dropna(axis=0, how='all', inplace=True)
        principal.dropna(axis=1, how='all', inplace=True)
        return principal

    def to_opp_cost(self, ir_market, horizon, principal):
        """Returns the opportunity cost to get the regulated loan.

        Calls self.apply_targets to restrict transition that are targeted by the policy.

        Parameters
        ----------
        ir_market: float or pd.Series
            market interest rate that would be paid without the regulated loan
        horizon: int or pd.Series
            loan horizon
        principal: float or pd.Series or pd.DataFrame
            capex of a transition from row to column

        Returns
        -------
        float, pd.Series, pd.DataFrame

        """

        if self.targets is not None:
            principal = self.apply_targets(principal)

        def to_interest_cost(ir, hrzn, p, p_min=None, p_max=None):
            if isinstance(p, pd.Series):
                if p_min is not None:
                    p_min = p_min.loc[p.index]
                    p[p < p_min] = 0
                if p_max is not None:
                    p_max = p_max.loc[p.index]
                    p[p > p_max] = p_max
                v_func = np.vectorize(RegulatedLoan.interest_cost)
                return v_func(ir.to_numpy(), hrzn.to_numpy(), p.to_numpy())

        if isinstance(principal, pd.DataFrame):
            ic_market = pd.DataFrame()
            for column in principal.columns:
                ir_market = ir_market.loc[principal.index]
                horizon = horizon.loc[principal.index]
                ic_market[column] = to_interest_cost(ir_market, horizon,
                                                     principal.loc[:, column],
                                                     p_min=self.principal_min, p_max=self.principal_max)
            ic_market.index = principal.index
            ic_market.columns.names = principal.columns.names

        elif isinstance(principal, pd.Series):
            ic_market = to_interest_cost(ir_market.to_numpy(), horizon.to_numpy(), principal.to_numpy(),
                                         p_min=self.principal_min, p_max=self.principal_max)
            ic_market.index = ir_market.index

        elif isinstance(principal, float):
            principal = min(self.principal_max, principal)
            principal = min(self.principal_min, principal)
            ic_market = RegulatedLoan.interest_cost(ir_market, horizon, principal)
        else:
            raise
        # ic_market = pd.Series(ic_market, index=ir_market.index)

        if self.ir_regulated is None:
            ic_regulated = 0
        else:
            v_func = np.vectorize(RegulatedLoan.interest_cost)
            ic_regulated = v_func(self.ir_regulated.to_numpy(), self.horizon.to_numpy(), principal)
            ic_regulated = pd.Series(ic_regulated, index=ir_market.index)
        opportunity_cost = ic_market - ic_regulated
        return opportunity_cost

    def to_opportunity_cost(self, principal):
        """Every parameter must be the same shape.
        """
        opportunity_cost = self.to_opp_cost(self.ir_market, self.horizon, principal)
        return opportunity_cost
