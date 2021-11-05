## Specification of Version 4.0
### Data used in version 4.0
As explained in the previous section, the core of Res-IRF is a stock matrix that describes the structure of the
residential stock at an original year. Until recently, the model was based on the image of the residential stock derived
from the 2012 Phébus survey data. This is what will be called the 2012 version of Res-IRF.

For greater accuracy of the model projections, a new stock matrix corresponding to the image of the residential stock in
2018 was imported. This matrix was built up using data provided by the SDES as part of the research agreement between
the CGDD and the CIRED in which this work is included.

Another dataset was also provided concerning the correspondence between the incomes of landlords and those of
tenants in the private rental stock.

These two data sets are described in the following sections. They have been constituted by the SDES from data coming
from the Fidéli 2018 database, from the DPE 2017 and 2018 database of the ADEME and from the Enerter model of the
company Energies Demain for the dwellings built before 1948 (2).


### Building stock
The first dataset incorporated into Res-IRF is the new stock matrix (MAT1) representing the residential stock in 2018. It describes the number of units in the residential stock by various segments.

This matrix segments the 28.6 million dwellings in the residential stock (primary residences only) into five categories:
* Energy performance level: DPE label ranging from G to A;
* Type of housing: individual or collective;
* Type of owner: owner-occupier, owner-lessor, social housing, free rental housing;
* Income level of the owner: income deciles of the tenant;
* Heating energy: natural gas, electricity, "other energy".

These data had to be reprocessed before they could be imported into Res-IRF. The following changes were made:
* Deletion of housing rented free of charge: this category, present in the data provided, is not relevant in Res-IRF and represents a limited number of people;
* Conversion of owner deciles to owner quintiles: As Res-IRF is currently adapted to handle only income quintiles and not deciles, the numbers of deciles belonging to the same quintile have been grouped together;
* Treatment of heating energy data: the only energy categories available in the data received were natural gas, electricity and the "other energies" category. However, Res-IRF includes four types of energy: electricity, gas, but also homes heated with oil and wood. It was therefore necessary to determine the share of fuel oil and wood in the "other energy" category. For this purpose, other data on the structure of the stock in 2018 from CEREN were used (37). For each type of dwelling (single-family or multi-family), these data make it possible to obtain the percentage of dwellings heated with wood and fuel oil among those heated neither with electricity nor with gas. The assumptions used are presented in the following table.

**Proportion of dwellings heated with wood and oil among dwellings not heated with gas or electricity, by type of dwelling**

|| **Wood fuel** | **Oil fuel** |
| ----------------------- | --------- |--------- |
| **Multi-family**  | 27%       | 30,8% |
| **Single-family** | 25,9%     | 6,2% |

*Note for the reader: among collective dwellings that are heated neither by gas nor by electricity, 27% are heated by wood and 30.8% are heated by oil.*

* Shift from tenant income to owner income: the income levels expressed in the stock matrix provided by the SDES were initially those of tenants. However, for Res-IRF to function properly, it is necessary to start by importing into the model the numbers of private rental housing stock according to the incomes of owners, as explained in section 2.2.4. It was therefore necessary to match the incomes of tenants with the incomes of owners in the private rental housing stock by means of a pass-through matrix.


#### Scope
This new structure is compared to the old version of Res-IRF, calibrated from the 2012 Phebus data. We look at both the
original 2012 fleet matrix and the projection that this 2012 calibrated version gives for the year 2018. For the
analysis of the results that follows, it is important to keep in mind that we are not comparing staff of identical size.

Indeed, the stock matrix used in 2012 had to be truncated even more than the one used in 2018 to be usable in Res-IRF:
it indicated a residential stock of 23.9 million dwellings in 2012 and the projection for 2018 reaches 25.5 million
dwellings. In comparison, the new data show a stock of 26.7 million dwellings in 2018.

**Comparison Phebus-2012 building stocks and SDES-2018**
![phebus_sdes_percent][phebus_sdes_percent]
![phebus_sdes_millions][phebus_sdes_millions]

Note:
* SDES building stock got a decile segmentation for income class. We aggregate them in figures in order to make the
  comparison possible.
* All images come from comparison.iynb a Jupyter notebook available in the project.

#### Energy performance
The number of dwellings in each EPC band is directly given by Phébus-DPE and SDES database. Figure compares the
distribution of EPC labels in versions 3.0 and 4.0 of the model. The stock in the latter is more energy efficient in
year 2018 than was that of the former in year 2012, with less dwellings in lower EPC bands (G-E) and more in upper
bands (D-A).

![phebus_sdes_energy_performance_percent][phebus_sdes_energy_performance_percent]

#### Heating Energy
The model covers energy use for heating from electricity, natural gas, fuel oil and fuel wood. This cope covers 16% of
final energy consumption in France.[^f04b] We consider only the main heating fuel used in each dwelling.

[^f04b]: Residential buildings in France contribute 26% of total energy consumption, 67% of which is devoted to heating,
91% of which is covered by electricity, natural gas, heating oil and wood energy (ADEME, 2015).

![phebus_sdes_energy_percent][phebus_sdes_energy_percent]

#### Household income


#### Renovation rate
In Res-IRF 3.0, the renovation rates used to calibrate Res-IRF were differentiated by decision maker and by ECD label.

Due to the lack of robustness of the data on renovation rates by EPC and the problematic results they generated with the
new image of the stock (SDES 2018), the renovation rates are now only differentiated by decision-makers. The latter have
been recalculated from data already used in the old parameterization of the renovation rates but also with other
additional data sources.

The 2012 version of Res-IRF (calibrated on Phébus) replicated the renovation rates observed in the French housing stock
according to the DPE and the decision maker of a dwelling. This meant using a matrix of 6 * 6 = 36 renovation rates,
which the calibration of parameters of the model (rho parameters of the renovation rate function) should make it
possible to replicate in the original year.