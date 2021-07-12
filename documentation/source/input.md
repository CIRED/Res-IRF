
# Input

> Res-IRF integrate a detailed description of the energy performance of the dwelling stock with a rich description of 
> household behaviour. Res-IRF has been developed to improve the behavioural realism that integrated models of energy 
> demand typically lack. (cf. README.md)

This description implies the use of a variety of different input.
Most of these inputs are uncertain. It is then necessary to run a sensitivity analysis.

**Data that are bolded get a native integration**

We consider multiple scenarios for each input.
However, it is time-consuming to run all scenarios, and we prefer running at least 3:
- Reference
- Optimistic
- Pessimistic 

Some data source are numbers, series or dataframe included in json file.  
Others come from others files (.csv, .pkl, etc...).
Source.json file contain sources to these data as file path.

Scenario file chose which scenarios are launched.

### Building stock
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


### Parameters
- **End year (2040)** (python input)
  
### Buildings dynamic

- **Destruction rate (0.35%/yr)**
- Residual destruction rate (5%/yr)
- Rotation rate
- Mutation rate

#### Construction dynamic

- Factor share multi-family
  
- Elasticity area construction
- Area max construction

### Demographic

These data are based on verified sources (INSEE), and are not prioritize for the uncertainty.

- Stock total ini
  - Total number of buildings the initial year.
  
- Factor population housing ini
  - Number of people by housing for the calibration year
- Population housing min
  - Minimum number of people by housing over the years
  
- Available income ini
  - Total available income 
- Available income rate

- Household income rate

- projection population


### Market share and renovation parameters
As other parameters (intangible cost and rho) are calibrated based on observed data, we don't prioritize the one bellow:
- Nu intangible cost
- Nu construction
- Nu label
- Nu energy
- NPV min
- Renovation rate max
- Renovation rate min

### Learning by doing

All these parameters seem to have been estimated 

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

### Cost and prices
- **energy prices**
  - energy price w/ taxes initial (€/kWh)
  - energy price rate (%/year)
- CO2 content
  - CO2 content initial (gCO2/kWh)
  - CO2 content rate (%/year)  
- construction cost
- **renovation cost**
- **switching-fuel cost**
- policies detailed
    - carbon tax data


### Behavioral parameters
- investment discount rate
- construction investment discount rate
- **envelope investment horizon** 
- **heater investment horizon**

### Numeric value for stock attributes
  - income
  - area
  - construction area
  - energy performance label primary consumption
  - energy performance label construction primary consumption
  - rate final energy in total primary consumption
  - rate heating energy in total final energy consumption
  - rate heating energy in total construction final energy consumption


### Observed data    
- market share renovation initial
  - Observed market share for the calibration year

- renovation rate initial
  

### Construction distribution 
- Housing type share total
- Occupancy status share housing type
- Energy performance share total construction
- Housing type share total construction
- Heating energy share housing type

### Others
- colors 