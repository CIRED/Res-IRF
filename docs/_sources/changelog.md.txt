
# Modification from Res-IRF 3.0

Modifications are listed in the table and the major changes are developed if necessary in a separate section.

```{eval-rst}
.. csv-table:: Code modification
   :name: code_modificaiton
   :file: table/changelog_resirf.csv
   :header-rows: 1
```

## Renovation rate function

The renovation rate $τ_i$ of dwellings labelled i is then calculated as a logistic function of the NPV:

$$τ_i=\frac{τ_{max}}{(1+(τ_{max}/τ_{min} -1) e^{-ρ(NPV_i- NPV_{min})})}$$

- $τ_{max}$ and NPV_{min} are constant based on own assumption,
- $NPV_i$ is a result from Res-IRF,
- $ρ$ is calibrated to replicate the observed renovation rates.

Previous work {cite:ps}`brangerGlobalSensitivityAnalysis2015` showed that the model was very sensitive to this parameter.

Res-IRF 3.0, based on the Phébus building stock, replicated the renovation rates observed in the French housing stock
according to the Energy Performance Certificate (EPC) and the Decision Maker (DM). 

Renovation rates by pair (ECD, decision maker) were determined using the following data: 
- 3% of all dwellings in the stock are renovated in the original year (source ADEME); 
- Share of renovations performed for each EPC {numref}`renovation_share`: $P(q | R)$ 
- Renovation rate by decision maker[^maker] {numref}`renovation_rate` : $P(R | d)$

[^maker]: Parameters imported in the Scilab version of Res-IRF were different from those indicated in some articles. We
present here the correct values.

```{eval-rst}
.. csv-table:: Renovation share by energy performance certificate. Source: PUCA (2015)
   :name: renovation_share
   :file: table/input_2012/renovation_share_ep_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

```{eval-rst}
.. csv-table:: Renovation rate by decision make. Source : OPEN (2016) and USH (2017)
   :name: renovation_rate
   :file: table/input_2012/renovation_rate_dm_2012.csv
   :header-rows: 1
   :stub-columns: 1
```

### Correction of renovation rate objective formula

The usual conditional probability formulas allow us to calculate rate_obj (renovation rate as a function of decision
maker and EPC) as follows:

$$\text{rate_obj} = P(R | d \cap q) = \frac{P(R \cap d \cap q)}{P(d \cap q)} = \frac{P(R) P(d \cap q | R)}{P(d \cap q)}$$

Assuming independence between:

$$P(d \cap q | R) = P(q|R) P(d | q \cap R) = P(q | R) P(d|R)$$


$$\text{rate_obj} = P(R | d \cap q) = \frac{P(d) P(q) P(R|d) P(R|q)}{P(R) P(d \cap q)}$$

- $P(R)$ : aggregate renovation rate of the stock of 3%. 
- $P(R|d)$ in {numref}`renovation_rate`
- $P(R|q) = P(R \cap q) / P(q) = P(R) * P(q|R) / P(q)$ and with $P(q|R)$ in {numref}`renovation_share`
- $P(q)$, $P(d)$, $P(d \cap q)$ calculated by the building stock structure.


Previously Res-IRF 3.0 calculated :

$$\text{rate_obj} = \frac{P(R|d) P(R|q)}{P(R)}$$

**We replace this formula by the formula defined above.** We assess the marginal impact of this modification on Res-IRF
3.0. We observe a change in absolute value but not in trend.


```{figure} img/changelog/renovation_rate_formula/flow_renovation.png
:name: renovation_rate_formula_flow_renovation

Impact renovation rate formula on flow_renovation
```

```{figure} img/changelog/renovation_rate_formula/renovation_rate_decision_maker.png
:name: renovation_rate_formula_renovation_rate_decision_maker

Impact renovation rate formula on decision-maker renovation rate
```

```{figure} img/changelog/renovation_rate_formula/renovation_rate_performance.png
:name: renovation_rate_formula_renovation_rate_performance

Impact renovation rate formula on EPC renovation rate
```

```{figure} img/changelog/renovation_rate_formula/consumption_actual.png
:name: renovation_rate_formula_consumption_actual

Impact renovation rate formula on consumption actual
```

### Calibration segmentation

Parameter ρ is calibrated, for each type of decision-maker and each initial label, so that the NPVs calculated with the
subsidies in effect in 2012 (version 3.0) reproduce the observed renovation rates. In general, we don't have enough
disaggregated data. 

**How can we calibrate an individual decision function per agent with only aggregated data?**

A simple solution would be to say that each agent must replicate the observed renovation rates of his group. In this
case, we have as many decision functions as there are agents. This solution is not satisfactory because it erases all
the specificities of the agents (discount rate, investment horizon, etc...) and is equivalent to creating a model with a
representative agent. Also, a robust decision function must be unique for all agents.

```{figure} img/changelog/renovation_rate_function/function_agents.png
:name: function_agents

Individual decision functions by agents.
```

#### Agents with the same observed renovation rate should get the same renovation rate function

Let say two agents got 2 NPV of 100 €/m2 (Agent 1) and 200 €/m2 (Agent 2) and their observed renovation rate is 3%. This
occurs when, due to lack of data, we assign two different agents the same renovation rate because they share common
attributes.
- For example, Agent 1 and Agent 2 are both Homeowners in a Single-family dwelling, but one is living
in high-efficient dwelling when the other is living in a low-efficient dwelling. Agent 1 should get lower incentive to
invest than Agent 2. 
- Another example, will be between two agents of two different owner income classes. Agent 1 owner could be
D1 with expensive investment loan when Agent 2 could become to D10.

```{figure} img/changelog/renovation_rate_function/agents_same_rate.png
:name: agents_same_rate

Individual decision functions by agents of the same group.
```

These situations push us to create one renovation rate function for all agents sharing the same observed renovation rate,
this means the same attributes. The objective is to determine the parameters in such a way that the average weighted by
the number of agents reproduces the observed renovation rate.

The ideal solution would be to determine the function parameter by solving the following equation:

$$\sum_{k=1}^n w(k) f(ρ, NPV_k) = \text{rate_obj}$$

However, this solution is not analytically solvable.

We are thinking of two other ways to calibrate the renovation function for all agents sharing the same attributes:
- Calculate function parameter for each individual agent and then calculate an average parameter.
- Calculate function parameter for a fictive individual agent that got an average utility function.

```{figure} img/changelog/renovation_rate_function/agents_comparison_rate.png
:name: agents_comparison_rate

Comparison of decision functions with the same renovation rate objective.
```

- There isn't critical differences between the two methods.
- There is no indication that the weighted average of the parameters allows us to recover the average value of the
  observed renovation rate. This is an approximation.
- Although without any theoretical support, we think it is simpler to imagine the average observed renovation rate
  representing the decision of a representative agent.

We will therefore calibrate the decision function by the utility of a representative agent.

#### All agents should get the same renovation rate function

Should agents with different observed renovation rate get different renovation rate function?

```{figure} img/changelog/renovation_rate_function/agents_different_rate.png
:name: agents_different_rate

Should agents with different observed renovation rate get different renovation rate function? 
```

We believe that it is more robust to determine a single decision function for all agents. We thus define the decision
function as the best logistic function passing through the pairs (Utility, Investment rate) or (NPV, Renovation rate)
for our specific case.

Let say two agents with the following attributes:
Agent 1: NPV of 100 €/m2, and observed renovation rate of 2%
Agent 2: NPV of 200 €/m2, and observed renovation rate of 3%.

```{figure} img/changelog/renovation_rate_function/renovation_fitting_function.png
:name: renovation_fitting_function

Fitting renovation rate objective data to utility data calculated.
```

#### Conclusion

We try to replicate the observed investment decision from a utility function per agent calculated by a model. However,
we do not have sufficiently disaggregated data to perform this calibration directly.

**How can we calibrate an individual decision function per agent with only aggregated data?**

1. Define a representative agent by segmentation of the observed data, 
2. Calculate the utility function for the representative agent,
3. Fitting observed renovation rate data to the utility calculated.

Even without disaggregated data, each agent will have a different decision when calibrating. The average of the agents
per group of the same attributes will reflect the observed decision rates.

#### Evaluation

```{figure} img/changelog/renovation_rate_function/flow_renovation.png
:name: flow_renovation

Flow renovations.
```

```{figure} img/changelog/renovation_rate_function/renovation_rate_decision_maker.png
:name: renovation_rate_decision_maker

Renovation rate by decision-maker.
```

```{figure} img/changelog/renovation_rate_function/renovation_rate_energy.png
:name: renovation_rate_energy

Renovation rate by heating energy.
```

```{figure} img/changelog/renovation_rate_function/renovation_rate_income_class.png
:name: renovation_rate_income_class

Renovation rate by income class.
```

```{figure} img/changelog/renovation_rate_function/renovation_rate_performance.png
:name: renovation_rate_performance

Renovation rate by energy performance certificate.
```