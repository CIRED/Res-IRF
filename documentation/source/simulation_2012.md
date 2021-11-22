
# Simulation and sensitivity analysis

## Influence of exogenous variables

The influence of technological progress, energy prices and aggregate household income is assessed by comparing the
reference scenario to alternatives in which they are frozen one after the other[^other].
[^other]: Changing the order in which the variables are frozen has little impact, which suggests that non-linearities are
not too important in the model.

```{eval-rst}
.. csv-table:: Scenario specification
   :name: scenario_specification_exoegenous
   :file: table/simulation/simulation_exogenous_variables.csv
   :header-rows: 1
```

{numref}`consumption_actual_phebus_exogenous` illustrates the resulting effect on the aggregate final energy consumption
for heating, the main output of the model. It shows that energy consumption decreases autonomously from 15% to 30%,
depending on the scenarios considered, between 2012 and 2050. Freezing technological progress increases energy
consumption by 12% in 2050 relative to the reference scenario. Additionally, freezing energy prices further increases
energy consumption by 20%. Lastly, freezing income decreases energy consumption by another 15%.

```{figure} img/simulation_2012/exogenous/consumption_actual.png
:name: consumption_actual_phebus_exogenous

Evolution of final actual energy consumption
```

{numref}`flow_renovation_phebus_exogenous` and {numref}`stock_performance_phebus_exogenous` illustrate effects on the
intensive and extensive margin of renovation. {numref}`flow_renovation_phebus_exogenous` shows that freezing
technological progress significantly reduces the annual flow of renovations. Additionally freezing energy prices
reinforces this effect, while freezing income has no additional effect. The declining trend observed in the all-frozen
scenario illustrates the depletion of the potential for profitable renovations as a result of past renovations. The
comparison with the other scenarios shows that technological progress and, to a lesser extent, increasing energy prices
are augmenting this potential. {numref}`stock_performance_phebus_exogenous` shows that the scenarios with the highest
number of renovations generate fewer numbers in the low-efficiency labels (G to C) and more in high-efficiency ones (B
and A).

```{figure} img/simulation_2012/exogenous/flow_renovation.png
:name: flow_renovation_phebus_exogenous

Renovation flows
```

```{figure} img/simulation_2012/exogenous/stock_performance.png
:name: stock_performance_phebus_exogenous

Evolution of energy performance
```

The aggregate heating intensity decreases with the theoretical budget share dedicated to heating {numref}`heating_intensity_phebus_exogenous`,
which in turn decreases with the energy performance of the dwelling, decreases with household income and increases with
the price of energy. The effect of the energy price on this quantity is ambiguous since, in addition to the direct
effect just mentioned, there is the indirect effect discussed in the previous paragraph that the energy price stimulates
renovations. The fact that the curve with non-frozen energy prices crosses that with frozen energy prices (both with
frozen technological change) suggests that the direct effect dominates in the short term while the indirect effect
dominates in the long term. The comparison of the first two scenarios indicates that freezing technological change
reduces heating intensity, an effect due, as we have seen previously, to a lesser improvement in energy efficiency.
Finally, the comparison of the last two scenarios illustrates the positive effect of income growth on heating intensity.
It is important to note that, at the same time, the growth in aggregate income increases the surface area of dwellings
to be heated.

```{figure} img/simulation_2012/exogenous/heating_intensity.png
:name: heating_intensity_phebus_exogenous

Evolution of heating intensity
```

To sum up:
- The reduction in energy consumption is to a large extent autonomous, that is, important even when all key drivers are
  frozen;
- The rise in energy prices stimulates renovation and reduces intensity of heating; the former effect tends to take over
  the latter in the long term;
- Technological progress has a pure effect of improving energy efficiency;
- Income growth increases energy consumption by increasing both the area to be heated and the intensity of heating.

{numref}`energy_poverty_phebus_exogenous` displays the evolution of the share of dwellings in fuel poverty, as measured
by the income-to-price ratio that counts the households spending more than 10% of their income on heating. It shows
that, when all key drivers are frozen, this count consistently declines. The increase in income naturally accelerates
this trend, while the increase in energy prices does the opposite. In comparison, technological change has a modest
effect on reducing fuel poverty.

```{figure} img/simulation_2012/exogenous/energy_poverty.png
:name: energy_poverty_phebus_exogenous

Number of households dedicating over 10% of their income to heating
```

## Influence of the determinants of capitalization

```{eval-rst}
.. csv-table:: Scenario specification capitalization
   :name: scenario_specification_capitalization
   :file: table/input_2012/investment_horizon_2012.csv
   :header-rows: 1
```

Here we compare the four scenarios specified in {numref}`scenario_specification_capitalization` (the scenario without
the landlord-tenant dilemma corresponding to full capitalization). The absence of capitalization results in a reduction
in the number and quality of renovations {numref}`flow_renovation_phebus_capitalization` and {numref}`stock_performance_phebus_capitalization`; 
even if this trend induces a counter-rebound effect {numref}`heating_intensity_phebus_capitalization`, it results in an increase in
aggregate energy consumption {numref}`consumption_actual_phebus_capitalization`. The absence of capitalization in rents has a similar effect, although much
weaker. In contrast, under the assumption of full capitalization, whereby landlords have the same willingness to invest
as owner-occupiers, energy efficiency improvements and, ultimately, energy savings are more important than in reference.


```{figure} img/simulation_2012/capitalization/flow_renovation.png
:name: flow_renovation_phebus_capitalization

Renovation flows
```

```{figure} img/simulation_2012/capitalization/stock_performance.png
:name: stock_performance_phebus_capitalization

Evolution of energy performance
```

```{figure} img/simulation_2012/capitalization/heating_intensity.png
:name: heating_intensity_phebus_capitalization

Evolution of heating intensity
```

```{figure} img/simulation_2012/capitalization/consumption_actual.png
:name: consumption_actual_phebus_capitalization

Evolution of final actual energy consumption
```

## Influence of credit constraints

To assess the influence of credit constraints, we set discount rates as follows:
- **Reference**: the discount rates are those described in {numref}`discount_rate_existing`;
- **No credit constraints**: all discount rates are equal to 7%;
- **Public investment**: all discount rates are equal to 4%.

As shown in the following figures, these variants have relatively modest effects on renovation dynamics and aggregate
consumption. Credit constraints therefore do not seem to play a major role in the model. This result is consistent with
the conclusions of the sensitivity analysis to which Res-IRF 2.0 was subjected {cite:ps}`brangerGlobalSensitivityAnalysis2015`.

```{figure} img/simulation_2012/creditconstraint/flow_renovation.png
:name: flow_renovation_phebus_creditconstraint

Renovation flows
```

```{figure} img/simulation_2012/creditconstraint/stock_performance.png
:name: stock_performance_phebus_creditconstraint

Evolution of energy performance
```

```{figure} img/simulation_2012/creditconstraint/heating_intensity.png
:name: heating_intensity_phebus_creditconstraint

Evolution of heating intensity
```

```{figure} img/simulation_2012/creditconstraint/consumption_actual.png
:name: consumption_actual_phebus_creditconstraint

Evolution of final actual energy consumption
```