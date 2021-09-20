from utils import reindex_mi
import numpy as np
import numpy_financial as npf
import pandas as pd


class PublicPolicy:
    """Public policy parent class.

    Attributes
    ----------
    name : str
        Name of the policy.
    start : int
        Year policy starts.
    end : int
        Year policy ends.
    policy : {'energy_taxes', 'subsidies'}
    calibration : bool, default: False
        Should policy be used for the calibration step?
    """
    def __init__(self, name, start, end, policy, calibration=False):
        self.name = name
        self.start = start
        self.end = end
        self.policy = policy
        self.calibration = calibration


class EnergyTaxes(PublicPolicy):
    """Represents energy taxes.

    Attributes
    ----------
    name : str
        Name of the policy.
    start : int
        Year policy starts.
    end : int
        Year policy ends.
    calibration : bool, default: False
        Should policy be used for the calibration step?
    unit : {'%', '€/kWh', '€/tCO2'}
        Unit of measure of the value attribute.
    value: float, pd.Series or pd.DataFrame
        Value of energy taxes.
    """
    def __init__(self, name, start, end, unit, value, calibration=False):
        """EnergyTaxes constructor.

        Parameters
        ----------
        name : str
            Name of the policy.
        start : int
            Year policy starts.
        end : int
            Year policy ends.
        calibration : bool, default: False
            Should policy be used for the calibration step?
        unit : {'%', '€/kWh', '€/tCO2'}
            Unit of measure of the value attribute.
        value: float, pd.Series or pd.DataFrame
            Value of energy taxes.
        """
        super().__init__(name, start, end, 'energy_taxes', calibration)

        self.list_unit = ['%', '€/kWh', '€/tCO2']
        self.unit = unit
        self.value = value
        self.calibration = calibration

    def price_to_taxes(self, energy_prices=None, co2_content=None):
        """Calculate energy taxes cost based on self.unit unit of measure.

        Parameters
        ----------
        energy_prices : pd.Series or pd.DataFrame, optional
            Energy prices in €/kWh. Heating energy as index, and years as columns.
        co2_content : pd.Series or pd.DataFrame, optional
            CO2 content in gCO2/kWh. Heating energy as index, and years as columns.

        Returns
        -------
        pd.Series
            Energy tax cost in €/kWh.
        """
        if self.unit == '%':
            val = energy_prices * self.value

        elif self.unit == '€/kWh':
            val = self.value

        elif self.unit == '€/tCO2':
            # €/tCO2 * gCO2/kWh / 1000000 -> €/kWh
            value = self.value
            if isinstance(value, int):
                value = pd.Series(value, co2_content.columns)
            else:
                idx = value.columns.union(co2_content.columns)
                value = value.reindex(idx, axis=1)
                co2_content = co2_content.reindex(idx, axis=1)
            taxes = value * co2_content
            taxes.fillna(0, inplace=True)
            taxes = taxes.reindex(energy_prices.columns, axis=1)
            val = energy_prices * taxes / 10**6

        else:
            raise AttributeError

        if isinstance(val, float):
            return pd.Series(val, index=range(self.start, self.end))
        elif isinstance(val, pd.DataFrame):
            return val.loc[:, self.start:self.end - 1]


class Subsidies(PublicPolicy):
    """Represents energy taxes.

    Attributes
    ----------
    name : str
        Name of the policy.
    start : int
        Year policy starts.
    end : int
        Year policy ends.
    calibration : bool, default: False
        Should policy be used for the calibration step?
    unit : {'%', '€/kWh', '€/tCO2'}
        Unit of measure of the value attribute.
    value : float, pd.Series or pd.DataFrame
        Value of subsidies.
    transition : list, default: ['Energy performance']
        Transition to apply the subsidy.
    cost_max : float, optional
        Maximum capex cost to receive a subsidy.
    subsidy_max : float, optional
        Maximum subsidy.
    """

    def __init__(self, name, start, end, unit, value,
                 transition=None, cost_max=None, subsidy_max=None, calibration=False, time_dependent=False):
        super().__init__(name, start, end, 'subsidies', calibration)

        if transition is None:
            self.transition = ['Energy performance']
        else:
            self.transition = transition

        # self.targets = targets
        self.list_unit = ['%', '€/kWh', '€/tCO2', '€']
        self.unit = unit
        self.cost_max = cost_max
        self.subsidy_max = subsidy_max
        self.value = value
        self.time_dependent = time_dependent

    def to_subsidy(self, year, cost=None, energy_saving=None):
        """
        Calculate subsidy value based on subsidies parameters.

        Parameters
        ----------
        year: int
        cost: pd.Series, pd.DataFrame, optional
            Necessary if self.unit == '%'
        energy_saving:  pd.Series, pd.DataFrame, optional
            Necessary if self.unit == '€/kWh'

        Returns
        -------

        """

        if self.time_dependent:
            value = self.value[year]
        else:
            value = self.value

        if self.unit == '€':
            return value

        if self.unit == '%':
            # subsidy apply to one target
            if isinstance(value, pd.Series):
                val = reindex_mi(value, cost.index, axis=0)
            else:
                val = value
            # subsidy apply to a maximum cost
            if self.cost_max is not None:
                cost[cost > self.cost_max] = cost
            return val * cost

        if self.unit == '€/kWh':
            if isinstance(value, pd.Series):
                val = reindex_mi(value, energy_saving.index, axis=0)
            else:
                val = value
            return (val * energy_saving.T).T


class SubsidiesRecyclingTax(PublicPolicy):
    def __init__(self, name, start, end, tax_unit, tax_value, subsidy_unit, subsidy_value, calibration=False,
                 transition=None):
        """EnergyTaxes constructor.

        Parameters
        ----------
        name : str
            Name of the policy.
        start : int
            Year policy starts.
        end : int
            Year policy ends.
        subsidy_unit : {'%', '€/kWh', '€/tCO2'}
            Unit of measure of the value attribute.
        subsidy_value: float, pd.Series or pd.DataFrame
            Initial value of energy subsidies.
            Should be high enough to create subsidy expenses bigger than tax revenues, then a dichotomy takes place.
        calibration : bool, default: False
            Should policy be used for the calibration step?
        """
        super().__init__(name, start, end, 'subsidy_tax', calibration)
        
        if transition is None:
            self.transition = ['Energy performance']
        else:
            self.transition = transition

        self.unit = subsidy_unit
        self._value = subsidy_value
        self.value_max = subsidy_value

        self._energy_tax = EnergyTaxes('{} tax'.format(name), start, end, tax_unit, tax_value)
        self._subsidy = Subsidies('{} subsidy', start + 1, end, self.unit, self._value, transition=transition)

        self.tax_revenue = dict()
        self.subsidy_expense = dict()
        self.subsidy_value = dict()

    @property
    def subsidy(self):
        return self._subsidy
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        self._value = val
        self._subsidy.value = val

    def price_to_taxes(self, energy_prices=None, co2_content=None):
        return self._energy_tax.price_to_taxes(energy_prices=energy_prices, co2_content=co2_content)

    def to_subsidy(self, cost=None, energy_saving=None):
        return self._subsidy.to_subsidy(cost=cost, energy_saving=energy_saving)


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
    """

    def __init__(self, name, start, end, ir_regulated=None, ir_market=None,
                 principal_max=None, principal_min=None,
                 horizon=None, targets=None, transition=None, calibration=False):
        super().__init__(name, start, end, 'regulated_loan', calibration)

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
        >>> RegulatedLoan.interest_cost(0.1, 1, 100)
        10
        >>> RegulatedLoan.interest_cost(0.1, 10, 100000)
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


class RenovationObligation:
    def __init__(self, name, start_targets, participation_rate=1.0, columns=None, calibration=False):
        self.name = name
        self.policy = 'renovation_obligation'
        self.start_targets = start_targets
        self.targets = self.to_targets(columns)
        self.participation_rate = participation_rate
        self.calibration = calibration

    def to_targets(self, columns=None):
        if columns is None:
            columns = range(self.start_targets.min(), self.start_targets.max() + 1, 1)
        df = pd.DataFrame(index=self.start_targets.index, columns=columns)
        temp = dict()
        for idx, row in df.iterrows():
            start = self.start_targets.loc[row.name]
            row[row.index >= start] = 1
            row[row.index < start] = 0
            temp[row.name] = row
        df = pd.DataFrame(temp).T
        df.index.name = self.start_targets.index.name
        return df


class ThermalRegulation(PublicPolicy):
    """
    ThermalRegulationConstruction represents public policies that require a minimal performance target to build or
    retrofit a building.

    For example, by 2030 all constructions should be Zero Net Building, and by 2040 Positive Energy Building.
    """
    def __init__(self, name, start, end, target, transition):
        super().__init__(name, start, end, 'thermal_regulation', False)

        self.transition = transition
        self.target = self.parse_targets(target)

    @staticmethod
    def parse_targets(target):
        """
        Returns pd.Series containing a list of element to remove each year because of the regulation.

        Parameters
        ----------
        target: pd.Series
        end: int

        Returns
        -------
        pd.Series
        """
        parse_target = pd.DataFrame()
        for idx, yr in target.items():
            parse_target = pd.concat((parse_target, pd.Series(idx, index=range(yr[0], yr[1]))), axis=1)
        return parse_target

    def apply_regulation(self, attributes, year):
        """
        Remove target of possible attributes.

        Parameters
        ----------
        attributes: dict
            Key are attribute name (Energy performance) and items are list of possible value taken by attributes.
        year: int

        Returns
        -------
        """
        for val in self.target.loc[year, :]:
            attributes[self.transition].remove(val)
        return attributes
