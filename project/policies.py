from utils import reindex_mi
import numpy as np
import numpy_financial as npf
import pandas as pd


class PublicPolicy:
    def __init__(self, name, start, end):
        self.name = name
        self.start = start
        self.end = end


class EnergyTaxes(PublicPolicy):
    def __init__(self, name, start, end, kind, value):
        """
        value: pd.DataFrame
        indexes are energy, and columns are years
        """
        super().__init__(name, start, end)

        self.list_kind = ['%', '€/kWh', '€/gCO2']
        self.kind = kind
        self.value = value

    def price_to_taxes(self, energy_prices, co2_content=None):

        if self.kind == '%':
            return energy_prices * self.value, energy_prices * (1 + self.value)

        elif self.kind == '€/kWh':
            return self.value, energy_prices * self.value

        elif self.kind == '€/gCO2':
            # €/tCO2 * gCO2/kWh / 1000000 -> €/kWh
            taxes = self.value * co2_content
            taxes.fillna(0, inplace=True)
            taxes = taxes.reindex(energy_prices.columns, axis=1)
            return energy_prices * taxes, energy_prices * (1 + taxes)

        else:
            raise AttributeError


class Subsidies(PublicPolicy):
    def __init__(self, name, start, end, kind, value,
                 transition=None, targets=None, cost_max=None, subsidy_max=None):
        super().__init__(name, start, end)

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
                if isinstance(self.targets, pd.DataFrame):
                    val = reindex_mi(val, cost.index, self.value.columns.names, axis=1)
            else:
                val = self.value
            # subsidy apply to a maximum cost
            if self.cost_max is not None:
                cost[cost > self.cost_max] = cost

            return val * cost

        if self.kind == '€/kWh':
            return self.value * energy_saving


class RegulatedLoan(PublicPolicy):
    """Loan backed by the state.

    Instead of paying interest_rate (%/yr).
    Example: EPTZ
    """

    def __init__(self, name, start, end, ir_regulated=0.0, principal_max=None, horizon_max=None, ir_market=None,
                 horizon=None):
        super().__init__(name, start, end)
        self.ir_regulated = ir_regulated
        self.horizon_max = horizon_max
        self.principal_max = principal_max
        self.ir_market = ir_market
        self.horizon = horizon

    @staticmethod
    def interest_cost(interest_rate, n_period, principal):
        # TODO: discount interest payment
        """Calculate total interest cost.

        Parameters
        ----------
        interest_rate: float
        Loan interest rate (%/period)
        n_period: float
        Number of periods
        principal: float
        Loan amount
        """
        period = np.arange(n_period) + 1
        return - npf.ipmt(interest_rate, period, n_period, principal).sum()

    def loan2subsidy(self, ir_market, horizon, principal):
        """Every parameter must be the same shape.
        """
        # ir_market = np.full(principal.shape, ir_market)
        # horizon = np.full(principal.shape, horizon)
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

        return opportunity_cost

    def loan_approximate2subsidy(self, principal):
        """Every parameter must be the same shape.
        """
        # ir_market = np.full(principal.shape, ir_market)
        # horizon = np.full(principal.shape, horizon)
        if isinstance(principal, np.ndarray):
            principal[principal > self.principal_max] = self.principal_max
            vfunc = np.vectorize(RegulatedLoan.interest_cost)
            ic_market = vfunc(self.ir_market, self.horizon, principal)
        else:
            principal = min(self.principal_max, principal)
            ic_market = RegulatedLoan.interest_cost(self.ir_market, self.horizon, principal)

        if self.ir_regulated == 0:
            ic_regulated = 0
        else:
            vfunc = np.vectorize(RegulatedLoan.interest_cost)
            ic_regulated = vfunc(self.ir_regulated, self.horizon, principal)
        opportunity_cost = ic_market - ic_regulated

        return opportunity_cost
