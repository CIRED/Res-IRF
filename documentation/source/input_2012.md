
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
- Six types of investors – owner-occupiers, landlords or social housing managers, each in single- and multi-family
  dwellings;
- Five categories of household income, the boundaries of which are aligned with those of INSEE quintiles.

### Scope

Res-IRF 3.0 covers 23.9 million principal residences in metropolitan France among the 27.1 million covered by the
Phébus-Clode survey for the year 2012. This scope differs from that of other databases (see {numref}`databases_overview`).
It was delineated by excluding from the Phébus sample: those dwellings heated with fuels with low market shares, such as
liquefied petroleum gas (LPG) and district heating; some dwellings for which it was not possible to identify a principal
energy carrier; some dwellings for which the Phébus data were missing.

```{figure} img/input_2012/buildingstock_2012_absolute.png
:name: buildingstock_2012_absolute

Building stock 2012
```

```{figure} img/input_2012/buildingstock_2012_percent.png
:name: buildingstock_2012_percent

Building stock 2012 (%)
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
- the propensity of owners to invest in energy retrofits and 
- (ii) the intensity of use of heating infrastructure by occupants.
The level of detail of the Phébus database made this development possible. Yet since the income data it contains only
relates to occupants, additional data were needed to set income parameters for landlords.

#### Occupants

The disposable income of occupants – owner-occupiers and tenants – is segmented into five categories delineated by the
same income boundaries as those defining income quintiles in France, according to the national statistical office for
2012. The use of these quintiles instead of those intrinsic in the Phébus sample ensures consistency between homeowners’
and tenants’ income (see next section), without introducing too strong biases, as shown in {numref}`income_categories`. 
2013. Each dwelling is then allocated the average income for its category. {numref}`income_energy_performance` 
2014. illustrates the distribution of occupant income in the different EPC bands. A clear correlation appears between 
2015. household income and the energy efficiency of their dwelling.[^dwelling]

[^dwelling]: The low number of dwellings labelled A and B in Phébus makes income distribution statistics less accurate in these bands.


```{figure} img/input_2012/income_energy_performance.png
:name: income_energy_performance

Distribution of income categories within EPC bands (Source: Phébus)
```

```{eval-rst}
.. csv-table:: Income categories used in Res-IRF 3.0
   :file: table/income_categories.csv
   :name: income_categories
   :header-rows: 1
   :stub-columns: 1

```

#### Owners

Homeowners’ income overlaps with occupants’. Yet Phébus does not contain any information on the income of landlords,
which we had to reconstitute by other means. We matched the Phébus-DPE data with INSEE data pre-processed by the Agence
nationale pour l’information sur le logement {cite:ps}`anilBailleursLocatairesDans2012`. The resulting landlords’ income
distribution is described in Figure 7 and compared to that of tenants. Here again, significant disparities appear, with
households whose annual income falls below €34,210 representing 80% of tenants but only 20% of owner-occupiers.


```{figure} img/input_2012/income_owners_occ_status.png
:name: income_owners_occ_status

Distribution of income categories for landlords and tenants in privately rented housing
```

To build this figure, some adjustments are needed to translate into income categories the {cite:
ps}`anilBailleursLocatairesDans2012` data that are expressed in terms of living standard[^standard].

[^standard]: This metric divides household income by consumption units – 1 for the first adult, 0.5 for any other person older
than 14 and 0.3 for any person under that age. It is generally thought to better represent the financing capacity of a
household than does income.

## Input

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
   :name: input_calibration_target
   :file: table/input_calibration_target.csv
   :header-rows: 1
   :stub-columns: 1
```


```{bibliography}
:style: unsrt
```