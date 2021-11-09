
# Setting up public policies 

The parameterization elements detailed below are summarized {numref}. 

```{eval-rst}
.. csv-table:: Summary of key policy parameters.
   :name: public_policies_summary
   :file: table/policies/policies_summary.csv
   :header-rows: 1
   :stub-columns: 1
```

## Carbon tax (CAT)

The carbon tax is applied as of 2014 to natural gas and heating oil. Its revenues are not recycled to
households. The instrument is parameterized as follows:
- Its rate (€/tCO2) is the same as in the TECV law: €30.50 in 2017, €39 in 2018, €47.5 in 2019 and 56€ in 2020. From
  2021 onwards, a growth rate of 6%/year will be applied to reach the target value of €100 in 2030. The tax rate then
  evolves at a rate of 4%/year, as recommended by the Quinet report (2008). This rate is subject to the same 20% VAT
  that applies to energy prices.
- The carbon contents to which the tax applies are 271 gCO2/kWh PCI for heating oil and 206 gCO2/kWh PCI for natural
  gas. The latter will decrease at a rate of 1%/year from 2020 onwards in order to take into account the objectives of
  renewable gas penetration (leading to a 26% share in 2050).

The purpose of the carbon tax is to send a price signal to investors to redirect long-term investments. In accordance
with this principle, the future rate of the tax is announced several years in advance by the government. Such a tax
takes full effect if investors form perfect expectations - they take into account the chronicity of the tax in their
profitability calculations. In practice, anticipatory behavior is closer to the myopia that prevails in the Res-IRF
model. Therefore, we define the following scenario variants, which limit the impact of the instrument:
- Scenario TC: myopically anticipated carbon tax
- Scenario TC+: perfectly anticipated carbon tax

## Zero-interest loan (ZIL)

The ZIL is represented as of 2012 as a subsidy equal to the interest on a consumer credit[^credit] used for an investment of
an equivalent amount. Among the different subsidies evaluated, the ZIL has the particularity of targeting work that
generates substantial jumps in performance. In the model, the targeting is based on the "minimum overall energy
performance" requirements, interpreted as follows:
- Post-work consumption ceiling of 150 kWh/m²/year if pre-work consumption exWCOds 180 kWh/m²/year. Interpreted as a
  minimum final D label for jumps from the initial G to E label.
- Consumption ceiling after work of 80 kWh/m²/year if consumption before work is less than 180 kWh/m²/year. Interpreted
  as a minimum final label B for jumps from the initial label D and C. 

[^credit]: An alternative would be to take as a reference the interest rates of a real estate loan, which are generally lower than
those of a consumer loan. However, this hypothesis is only relevant for a minority of situations where the renovation
work is coupled with the purchase of the property. 

By construction, the assumptions made about the terms of the consumer credit define the amount of the subsidy. Two
alternative scenarios are selected:

- ZIL scenario: consumer credit terms are those given by OPEN (2016), i.e., an interest rate of 3% over 5 years. In
  addition, in order to incorporate the heterogeneous financing constraints faced by households, we add to each
  household category a loan ceiling corresponding to the average amount borrowed under the ZIL in the data of the
  Société de Gestion du Fonds de Garantie à l'Action Sociale (SGFGAS), from 16,800€ for categories C1 to 21,000€ for
  categories C5. Finally, a minimum loan threshold of €5,000 is added.
- ZIL+ scenario: the instrument is closer to its theoretical setting, defined by at an interest rate of 4% (as noted by
  the Banque de France for consumer loans in 201529 ) over a 10-year life (maximum authorized duration of an ZIL), with
  a loan amount capped at €30,000 and no minimum threshold. Converted into an ad valorem subsidy, these two variants
  correspond to subsidy rates of 9% for the ZIL and 23% for the ZIL+.

## Tax credit for energy transition

The CITE is represented from 2012 onwards as an ad valorem subsidy at a single rate of 17%, which corresponds to the
average rate of assistance reported by OPEN30. The difference between this value and the official rate of The difference
between this value and the official rate of 30% reflects the fact that, in the model, the subsidy rate applies to the
full cost of the energy performance (supply and labor) when the official CITE rate applies to the cost of supply only (
except for opaque wall insulation). The instrument is modeled as a subsidy received immediately by the beneficiary,
without taking into account the maximum 12-month delay inherent in any tax credit31. In order to take into account the
current discussions on restricting the instrument to the most efficient measures (by cancelling the In order to take
into account the current discussions on restricting the instrument to the most efficient measures (by cancelling the
eligibility of windows, for example), two variants are modeled:
ISCED scenario: non-targeted subsidy CITE+ scenario: targeted subsidy like the ZIL

## White certificate obligations (WCO)

The WCO are represented from 2012 onwards as a hybrid instrument coupling an energy efficiency subsidy whose cost is
passed on by the obliged energy suppliers as a tax on energy sales. These two components are modeled as follows:

- The amount of the subsidy is defined for each operation by an amount of discounted cumulative energy savings {numref}`cumac_label_transition`
  multiplied by a WCO price (€/kWh cumulated). The latter is the subject of a scenario described below. The operations
  considered are those relating to the envelope and thermal systems of residential buildings [^buildings]. 
  {numref}`example_who_prices` relates the amounts thus calculated to the investment costs for the year 2012 to illustrate
  the ad valorem rates associated with such a subsidy. While the average rate is 5%, the WCO scale is "regressive" in
  the sense that the highest subsidy rates are for the least expensive operations; indeed, although the amounts granted
  increase with the energy performance achieved, this increase is less than the underlying increase in investment costs.
  The amount of the WCO subsidy is capped so that the total subsidy rate, including the CITE, TVAr and ZIL, does
  not exceed 100%.
- The tax (in € per kWh sold) is calculated by multiplying the official obligation coefficients (kWh cumac per kWh sold)
  by the price of WCO (€/kWh cumac). This tax is restricted to sales of domestic fuel oil, natural gas and electricity
  and increased by VAT at 20%. For the post-2020 period, where the obligation coefficients are not yet defined, an
  increase of 1%/year is considered. This assumption is equivalent to a constant amount of obligation assuming that
  energy sales follow the trend decline of 1%/year.

[^buildings]: https://www.ecologique-solidaire.gouv.fr/operations-standardisees

```{eval-rst}
.. csv-table:: Cumulative kWh/m² per label transition
   :name: cumac_label_transition
   :file: table/policies/cumac_label_transition.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Amount in €/m² per label transition, for a CEE price of 4€/MWh cumac
   :name: example_who_prices
   :file: table/policies/example_who_prices.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Subsidy rate for a CEE price of 4€/MWh cumac
   :name: example_who_subsidy_rate
   :file: table/policies/example_who_subsidy_rate.csv
   :header-rows: 1
   :stub-columns: 1
```


Subsidies under the WCO precariousness scheme are allocated according to the owner's income in the private sector and
according to the occupant's income in the social sector33. The taxes, on the other hand, are applied taxes are applied
uniformly to all households.

The price of the WCO is the main determinant of the impact of the instrument. Conventional EWCs apply to households in
the C3 to C5 category and precarious EWCs apply to C1 and C2 households. The amount of the subsidy and the WCO produced
are doubled for C1 households to reflect the bonus granted to very modest households. The precariousness WCOs are
counted from 2015 (and not 2016 as in reality) in order to facilitate the comparison between periods 3 and 4. In line
with recent trends, it is assumed that the price of conventional and precariousness WCO is identical and capped at 20
€/MWh cumac34. The price chronicle is subject to two scenario variants:

- WCO scenario: 4€ from 2012 to 2016, 5€ in 2017 and then 2%/year increase from 2018 until the 20 €/MWh cumac cap.
- WCO+ scenario: 4€ from 2012 to 2016, 5€ in 2017 then 15€ in 2018, increasing by 2%/year from 2019 until the ceiling of
  20 €/MWh cumac.

Since the proposed modeling only covers the residential building sector, two important looping mechanisms are missing
from a more complete evaluation of the instrument: the important looping mechanisms are missing from a more complete
evaluation of the instrument:

- In theory, the WCO price should result from a market equilibrium and reflect the opportunity cost of the constraint
  associated with the target. Taking these mechanisms into account implies representing all market actors, i.e. all
  sectors covered by the scheme, including agriculture, industry and transport. Since the exercise proposed here focuses
  on the residential sector, these adjustments cannot be taken into account. The price of WCO is therefore defined
  exogenously in the residential sector, without any explicit link to the total target.
- In theory, the tax component of the WCO is defined by each obligated supplier in such a way that the revenue it
  generates balances its subsidy expenses. To take this balance into account, it is again necessary to represent all
  sectors, since an obligated fuel seller may, for example, pass on the cost of subsidies granted in the residential
  sector to the price of fuel. In the exercise proposed here, the subsidies and taxes that apply to the residential
  sector are The subsidies and taxes that apply to the residential sector are defined exogenously and independently.

  
## Reduced rate VAT (VATr)

Renovation measures are subject to a reduced VAT rate of 5.5%, instead of the normal rate of 10% which applies in the
building sector. This assumption is embodied in our cost matrix.

## Building code (BCO)

Investment choices in new construction are limited to BBC and BEPOS levels between 2012 and 2019, then to zero energy
standard only from 2020.

