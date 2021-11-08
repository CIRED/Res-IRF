
# Input Res-IRF version 3.0

## Building stock used in version 3.0

The model is calibrated on base year 2012. What we refer to as existing dwellings corresponds to the stock of dwellings
available in 2012, minus annual demolitions. What we refer to as new dwellings is the cumulative sum of dwellings
constructed after 2012.

Previous versions of Res-IRF were parameterized on data published in 2008 by the {cite:ps}`anahModelisationPerformancesThermiques2008`. 
A major step forward for the time, this database aggregated data of varying quality from different sources. A
number of extrapolations made up for missing (e.g., the number of dwellings built before 1975) or imprecise (e.g.,
occupancy status of decision-makers) data.

The version 3.0 of the model is now mainly based on data from the Phébus survey (Performance de l’Habitat, Équipements,
Besoins et USages de l’énergie). Published in 2014, the Phébus data represented a substantial improvement in knowledge
of the housing stock and its occupants. Specifically, a more systematic data collection procedure allowed for new
information (in particular on household income) and improved accuracy of previously available information. These
advances now permit assessment of the distributional aspects of residential energy consumption.

The Phébus survey has two components:
- The so-called “Clode” sample details the characteristics of dwellings, their occupants and their energy expenditure.
  Specific weights are assigned to each household type to ensure that the sample is representative of the French
  population. 
- The so-called “EPC” sample complements, for a subsample of 44% of households in the Clode sample, socio-economic data
  with certain physical data, including the energy consumption predicted by the EPC label. In Res-IRF, specific weights
  are assigned to this sub-sample, based on Denjean (2014).

To parameterize the model, we matched the two components in a single database. Without further specification, the
matched database is the one we refer to when we mention Phébus in the text.

In addition to the Phébus data, we calibrate correction parameters so that the model outputs at the initial year are
consistent with the data produced by the Centre d'études et de recherches économiques sur l’énergie (CEREN). The CEREN
data differ from the Phébus ones in their scope and the methodology used to produce them. They however serve as a
reference for most projections of energy consumption in France.

**Overview of the database**
```{eval-rst}
.. csv-table:: Overview of the database
   :name: databases_overview
   :file: table/databases_overview.csv
   :header-rows: 1
   :stub-columns: 1

```

**Overview of the content of the databases**
```{eval-rst}
.. csv-table:: Overview of the database content
   :name: databases_overview_content
   :file: table/databases_overview_content.csv
   :header-rows: 1
   :stub-columns: 1

```

The model contains 1,080 types of dwellings divided into:
- Nine energy performance levels – EPC labels A to G for existing dwellings, Low Energy Building (LE) and Net Zero
  Energy Building (NZ) levels for new dwellings;
- Four main heating fuels – electricity, natural gas, fuel oil and fuel wood
- Three types of occupancy status – homeowners, landlords or social housing managers, 
- Two types of housing type: single- and multi-family dwellings;
- Five categories of household income, the boundaries of which are aligned with those of INSEE quintiles.

### Scope

Res-IRF 3.0 covers 23.9 million principal residences in metropolitan France among the 27.1 million covered by the
Phébus-Clode survey for the year 2012. This scope differs from that of other databases {numref}`databases_overview`.
It was delineated by excluding from the Phébus sample: those dwellings heated with fuels with low market shares, such as
liquefied petroleum gas (LPG) and district heating; some dwellings for which it was not possible to identify a principal
energy carrier; some dwellings for which the Phébus data were missing.

```{figure} img/input_2012/buildingstock_2012_absolute.png
:name: buildingstock_2012_absolute

Building stock 2012
```

### Energy performance

The number of dwellings in each EPC band is directly given by Phébus-DPE. 

```{figure} img/input_2012/buildingstock_ep_2012_percent.png
:name: buildingstock_ep_2012_percent

Building stock 2012 by Energy Performance
```

### Building characteristics and occupancy status

{numref}`decision_maker_distribution` specifies the joint distribution of building characteristics (singe- and
multi-family dwellings) and types of investors (owner-occupied, landlord, social housing manager).

```{eval-rst}
.. csv-table:: Joint distribution of building and investor characteristics in Res-IRF 3.0
   :name: decision_maker_distribution
   :file: table/decision_maker_distribution.csv
   :header-rows: 1
   :stub-columns: 1

```

### Heating fuel

The model covers energy use for heating from electricity, natural gas, fuel oil and fuel wood. This scope covers 16% of
final energy consumption in France. We consider only the main heating fuel used in each dwelling. To identify it from
the Phébus-Clode database, we proceed as follows:
1. We retain the main heating fuel when declared as such by the respondents.
2. When several main fuels are declared, we assign to the dwelling a heating fuel according to the following order of
   priority: district heating > collective boiler > individual boiler > all-electric > heat pump > other.
3. When no main fuel is reported, we retain the main fuel declared as auxiliary, determined with the following order
   of priority: *electric heater > all-electric > mixed base > fixed non-electric > chimney*.

```{figure} img/input_2012/energy_consumption_phebus.png
:name: energy_consumption_phebus

Energy consumption in Phébus
```

{numref}`energy_consumption_phebus` compares the total consumption of each fuel in the Phébus database and in the
model. It shows that retaining only one fuel for each dwelling leads us to consider much less electricity and wood
consumption than reported in Phébus. This is due for the most part to our exclusion of auxiliary heating, which
predominantly uses electricity and wood, and to a lesser extent to our exclusion of the specific electricity consumption
that is reported in Phébus.

### Household income

A major advance of version 3.0, the introduction of income categories was intended to capture heterogeneity in:
- the propensity of owners to invest in energy retrofits,
- the intensity of use of heating infrastructure by occupants.
The level of detail of the Phébus database made this development possible. Yet since the income data it contains only
relates to occupants, additional data were needed to set income parameters for landlords.

#### Occupants

The disposable income of occupants – owner-occupiers and tenants – is segmented into five categories delineated by the
same income boundaries as those defining income quintiles in France, according to the national statistical office for
2012. The use of these quintiles instead of those intrinsic in the Phébus sample ensures consistency between homeowners’
and tenants’ income (see next section), without introducing too strong biases, as shown in {numref}`income_categories`.
Each dwelling is then allocated the average income for its category. {numref}`income_energy_performance`
illustrates the distribution of occupant income in the different EPC bands. A clear correlation appears between
household income and the energy efficiency of their dwelling.[^dwelling]

[^dwelling]: The low number of dwellings labelled A and B in Phébus makes income distribution statistics less accurate in these bands.


```{figure} img/input_2012/income_energy_performance.png
:name: income_energy_performance

Distribution of income categories within EPC bands. Source: Phébus
```

```{eval-rst}
.. csv-table:: Income categories used in Res-IRF 3.0
   :file: table/income_categories.csv
   :name: income_categories
   :header-rows: 1
   :stub-columns: 1

```

#### Owners

Homeowners income overlaps with occupants. Yet Phébus does not contain any information on the income of landlords,
which we had to reconstitute by other means. We matched the Phébus-DPE data with INSEE data pre-processed by the Agence
nationale pour l’information sur le logement {cite:ps}`anilBailleursLocatairesDans2012`. The resulting landlords income
distribution is described in {numref}`income_owners_occ_status` and compared to that of tenants. Here again, significant
disparities appear, with households whose annual income falls below €34,210 representing 80% of tenants but only 20% of
owner-occupiers.

```{figure} img/input_2012/income_owners_occ_status.png
:name: income_owners_occ_status

Distribution of tenants income categories by occupancy-status.
```

To build this figure, some adjustments are needed to translate into income categories the {cite:ps}`anilBailleursLocatairesDans2012` 
data that are expressed in terms of living standard[^standard].

[^standard]: This metric divides household income by consumption units – 1 for the first adult, 0.5 for any other person older
than 14 and 0.3 for any person under that age. It is generally thought to better represent the financing capacity of a
household than does income.

## Complete list of inputs

We use the term input to name any factor that is given a numerical value 
in the model. Model inputs fall into three categories {cite:ps}`brangerGlobalSensitivityAnalysis2015`:
  - Exogenous input trajectories (EI) representing future states of the world:
  energy prices, population growth and GDP growth.
  - Calibration targets (CT), which are empirical values the model aims to
  replicate for the reference year. They include hard-to-measure aggregates
  such as the reference retrofitting rate and the reference energy
  label transitions.
  - All other model parameters (MP), which reflect current knowledge on
  behavioral factors (discount rates, information spillover rates, etc.) and
  technological factors (investment costs, learning rates, etc.)


```{eval-rst}
.. csv-table:: Complete list of inputs
   :name: input_list_2012
   :file: table/input_list_2012.csv
   :header-rows: 1
   :stub-columns: 1
```


### Exogenous input

- Energy prices: based on a scenario from ADEME using assumptions from the Directorate General for Energy and Climate (
  DGEC) and the European Commission. The scenario used is equivalent to an average annual growth rate of
  fuel prices after tax of 1.42% for natural gas, 2.22% for fuel oil, 1.10% for electricity and 1.20% for fuel wood over
  the period. These lead to an average annual growth rate of the price index of 1.47%/year.
- Population growth[^growth]: based on a projection from {cite:ps}`inseeProjectionsPopulationPour2006` equivalent to an average
  annual growth rate of 0.3%/year over the period 2012-2050.
- Growth in household income: extrapolates the average trend of 1.2%/year given by INSEE uniformly
  across all income categories.

[^growth]: The population is adjusted by a factor of 23.9/27.1 to take into account the difference in scope between
Res-IRF and Phébus. The resulting average household size is 2.2 persons per dwelling in 2013, a value consistent with
{cite:ps}`inseeMenagesToujoursNombreux2017`; it decreases with income to reach 2.05 in 2050.

### Calibration target

#### Construction 

Market shares used to calibrate intangible costs for construction.

```{eval-rst}
.. csv-table:: Market shares of construction in 2012.
   :name: market_share_construction_2012
   :file: table/ms_construction_ini_2012.csv
   :header-rows: 2
   :stub-columns: 2
```

#### Intensive margin

Intangible costs are calibrated so that the life-cycle cost model, fed with the investment costs, matches the market
shares reported here {numref}`market_share_2012`

```{eval-rst}
.. csv-table:: Market shares of energy efficiency upgrades in 2012. Source: PUCA (2015)
   :name: market_share_2012
   :file: table/ms_renovation_ini_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

In the absence of any substantial improvement in the quality of the data available, the matrix remains unchanged from
version 2.0 of the model.

#### Extensive margin

Parameter ρ (of renovation function) is calibrated, for each type of decision-maker and each initial label (i.e., 6x6=36
values), so that the NPVs calculated with the subsidies in effect in 2012 {cite:ps}`giraudetExploringPotentialEnergy2012` 
reproduce the renovation rates described in {numref}`renovation_share_ep_2012`and {numref}`renovation_rate_dm_2012` and 
their aggregation represents 3% (686,757 units) of the housing stock of the initial year.

```{eval-rst}
.. csv-table:: Renovation share by energy performance label. Source: PUCA (2015)
   :name: renovation_share_ep_2012
   :file: table/renovation_share_ep_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Renovation rate by type of dwelling. Source : OPEN (2016) and USH (2017)
   :name: renovation_rate_dm_2012
   :file: table/renovation_rate_dm_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

#### Aggregated energy consumption

To ensure consistency with the CEREN data, which is the reference commonly used in modelling exercises, Res-IRF is
calibrated to reproduce the final energy consumption given by CEREN for each fuel in the initial year. The resulting
conversion coefficients applied to the Phebus Building Stock are listed in {numref}`calibration_energy_2012`.

```{eval-rst}
.. csv-table:: Calibration of total final actual energy consumption
   :name: calibration_energy_2012
   :file: table/ceren_energy_consumption_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

### Dwelling Stock Variation Factors

```{eval-rst}
.. csv-table:: Initial Floor area construction (m2/dwelling)
   :name: area_construction_2012
   :file: table/area_construction_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Floor area construction elasticity
   :name: area_construction_elasticity_2012
   :file: table/area_construction_elasticity_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Maximum Floor area construction (m2/dwelling)
   :name: area_construction_max_2012
   :file: table/area_construction_max_2012.csv
   :header-rows: 1
   :stub-columns: 1
```


```{eval-rst}
.. csv-table:: Rotation and Mutation rate (%/year)
   :name: rotation_mutation_rate_2012
   :file: table/rotation_mutation_rate_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

### Investment cost factors

```{eval-rst}
.. csv-table:: Renovation costs used in Res-IRF 3.0 (€/m2). Source: Expert opinion
   :name: cost_renovation_2012
   :file: table/cost_renovation_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

The matrix equally applies to single- and multi-family dwellings, in both private and social housing. In the absence of
any substantial improvement in the quality of the data available, the matrix remains unchanged from version 2.0 of the
model. 


```{eval-rst}
.. csv-table:: Switching-fuel costs used in Res-IRF 3.0 (€/m2). Source: Expert opinion
   :name: switching_fuel_cost
   :file: table/cost_switch_fuel_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Construction costs (€/m2)
   :name: cost_construction_2012
   :file: table/cost_construction_2012.csv
   :header-rows: 2
   :stub-columns: 1
```

### Existing Dwelling Stock Factors 

```{eval-rst}
.. csv-table:: Initial Floor area (m2/dwelling)
   :name: area_existing_2012
   :file: table/area_existing_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Income (€/year)
   :name: income_2012
   :file: table/income_2012.csv
   :header-rows: 1
```

### Other factors

```{eval-rst}
.. csv-table:: Discount rates (%/year). Source: Expert opinion
   :name: discount_rate_existing_2012
   :file: table/discount_rate_existing_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Investment horizon (years). Source: Expert opinion
   :name: investment_horizon_2012
   :file: table/investment_horizon_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

Considering that the quality of new constructions results from decisions made by building and real estate professionals
rather than by future owners, we subject these decisions in the model to private investment criteria, reflected by a
discount rate of 7% and a time horizon of 35 years.

```{eval-rst}
.. csv-table:: Discount rates construction (%/year). Source: Expert opinion
   :name: discount_rate_construction_2012
   :file: table/discount_rate_construction_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

## Appendix
```{figure} img/input_2012/buildingstock_2012_percent.png
:name: buildingstock_2012_percent

Building stock 2012 (%)
```
