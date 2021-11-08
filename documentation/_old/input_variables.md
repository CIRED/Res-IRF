# Input


Buildings stock.

## Building stock
- **initial stock by attributes**:
  - Energy performance label
  - Heating energy (Power, Natural gas, Oil fuel, Wood fuel)
  - Housing type (Multi-family, Single-family)
  - Occupancy status (Landlords, Homeowners, Social-housing)
  - Income class tenant 
  - Income class owner

NB:  
- The model has been written to let user add or drop stock attributes. It will be necessary to rewrite some function in
  order to do so. 
- Users can also estimate attributes to populate their dataset and use the current version.

> Data from SDES-2018, Phebus-2012

### Numeric value for stock attributes
  - income
  - area
  - construction area
  - energy performance certificate primary consumption
  - energy performance certificate construction primary consumption
  - rate final energy in total primary consumption
  - rate heating energy in total final energy consumption
  - rate heating energy in total construction final energy consumption


## Exogenous input trajectories (EI)

- *Population growth* : based on a projection from INSEE (2006) equivalent to an average annual growth rate of 0.3%/year
  over the period 2012-2050.
- *Growth in household income*: extrapolates the average trend of 1.2%/year given by INSEE (see paragraph 2.3.1) uniformly
  across all income categories.
- *Energy prices*: based on a scenario from ADEME using assumptions from the Directorate General for Energy and Climate (
  DGEC) and the European Commission (see Figure 15). The scenario used is equivalent to an average annual growth rate of
  fuel prices after tax of 1.42% for natural gas, 2.22% for fuel oil, 1.10% for electricity and 1.20% for fuel wood over
  the period. These lead to an average annual growth rate of the price index of 1.47%/year.

- Available income ini
  - Total available income 
- Available income rate

- Household income rate


## Model parameters (MP)
- **End year (2040)** (python input)
  
### Buildings dynamic

- Destruction rate (0.35%/yr)
- Residual destruction rate (5%/yr)
- Rotation rate
- Mutation rate

#### Construction dynamic

Growth of multi-family building share in total stock:
- Factor share multi-family
  
Growth of area by unit in constructed buildings:
- Elasticity area construction
- Area max construction

### Demographic parameters
- Factor population housing ini
  - Number of people by housing for the calibration year
- Population housing min
  - Minimum number of people by housing over the years
  
  
### Market share and renovation parameters
Market share:
- Nu intangible cost
- Nu construction
- Nu label
- Nu energy

Renovation rate function:
- NPV min
- Renovation rate max
- Renovation rate min

### Learning by doing parameters
- Learning by doing renovation (10%)
  - Reduction of renovation cost when knowledge doubles
- Learning years renovation (10 years)
  - Calculate initial knowledge 
  
- Learning information rate renovation (25%)
  - Reduction of intangible cost when knowledge doubles
- Information rate max renovation (80%)
  - Maximum share of reduction

- Learning by doing construction (15%)
- Cost construction lim

- Learning information rate construction (25%)
- Information rate max construction (95%)

> Learning by doing and information acceleration can be activated or de-activated.

### Cost
- construction cost
- renovation cost
- switching-fuel cost
- CO2 content
  - CO2 content initial (gCO2/kWh)
  - CO2 content rate (%/year)  
- policies detailed
    - carbon tax data
  
### Behavioral parameters
- investment discount rate
- construction investment discount rate
- **envelope investment horizon** 
- **heater investment horizon**


## Calibration Targets
- market share renovation initial
  - Observed market share for the calibration year
- renovation rate initial
- market share construction
