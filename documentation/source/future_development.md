# Future developments
This version: December 18, 2019
Previous update: November 30, 2018
NB: the developments envisaged here are only for the France model. European and global extensions could be considered, but are not currently on the agenda.
## Deepening of the existing system
### Technological details
The central element of the model is the representation of the performance of the stock by ECD label. This is both a strength - a practical abstraction for modeling - and a weakness - as it stands, this abstraction lacks a solid empirical basis.
A priority is therefore to parameterize label transition costs more reliably. The goal would be to construct the matrix (with means and standard deviations) from a large number of combinations of explicit measures. The standard deviations of the costs would allow (i) Monte Carlo draws for sensitivity analysis and (ii) more fundamental uses, such as testing different risk aversion assumptions of investors.
This would allow us to better integrate fuel switches and label changes. It would also allow the model to be connected to the real markets of insulation, windows, etc., which is important in the longer term perspective of closure (point 3). 
**Development in progress: ANR PRCE project "PREMOCLASSE" with EDF R&D.**
### Micro-foundations
We could consider modeling demand with more explicit micro functions, based on the online appendix of the paper Giraudet et al, JAERE 2018 for demand, and Nauleau et al, Energy Economics 2015 for supply.
Tracks considered: none
### Other updates
There are a number of recently or soon to be published data that need to be incorporated into the model. This includes TREMI data from Ademe and an equivalent study on multi-family housing.
Eva Simon (DHUP) pointed out to us at the September 11, 2018 seminar that there was still room for improvement on the social housing stock parameterization, listing a number of data sources not yet exploited. 
**Path forward: No progress on this point since fall 2018. Note however the presence of Basile Pfeiffer at the DHUP**
## Integration of climate effects
### Integration of air conditioning
The objective is to have a more complete description of energy consumption related to thermal comfort, both winter and summer. This development involves three steps:
- Integrate the air conditioning consumptions in the model. This requires a good knowledge of the costs of the equipment, of the energy consumption specific to this use, etc. These elements are still much less well informed than for heating. Another important point is that unlike heating, where the assumption of a 100% equipment rate seems realistic, for air conditioning it is necessary to know this rate and the determinants (price, income, climate) of its evolution.
- Make heating and air conditioning consumption dependent on temperature (which is not currently the case for heating). In other words, integrate HDD and CDD into the model.
- These developments would be all the more relevant if the model were spatialized, first of all by climate zone H1/H2/H3. A point of caution, however: once spatialized, for the prospective scenarios we are obliged to make internal migration assumptions (status quo type, heliotropism, intensified urbanization...).
**Possible course of action: (i) Very preliminary contact with RTE for a possible CIFRE thesis. (ii) Could be inserted in ANR JCJC "AFRICOOL" if retained (even if not explicitly mentioned in the complete proposal). (iii) E4C. In any case, Vincent Viguié (and Samuel Juhel) are interested in these issues.**
### Inter-temporal optimization
The notion of "killing the deposit" - carrying out unambitious renovation operations that obliterate any subsequent improvement, thus contributing to maintain the stock in a poorly performing state - is quite central to the debates of the last few years on energy savings. However, it is little studied, despite its potentially important political implications; for example, one can imagine that it would lead to the recommendation of a high carbon price in the short term. Taking into account these effects (related to option value issues) requires using the model in inter-temporal optimization, like the TITAN model developed at the CGDD by Baptiste Perrissin-Fabert (and stemming from the work of Adrien Vogt-Schilb & Co).
This development would make even more sense in connection with the previous point (2.1). This would allow us to determine an optimal rate of insulation, which reconciles the problems of summer and winter comfort under climate change.
NB: Cyril's experience on the optimization of

**Planned track: (Informal solicitation of the ADEME on the subject (Marie-Laure at the time...).**
## Looping with IMACLIM-R France
### Energy supply-demand link
The looping of Res-IRF with the power generation module of IMACLIM-R-France has already been done by Mathy, Fink, Bibas (Energy Policy 2015). Res-IRF sends energy demands in kWh, to which the electricity module responds by sending back the associated CO2 emissions and electricity prices to the following year.
The exercise could be repeated and would be most relevant if it could link the version of Res-IRF augmented with air conditioning (see 2.1) and a power generation model that has a detailed load curve at the daily+seasonal level. This development would also be linked to the work of Behrang and Quentin, which would make it possible to deal more generally with trade-offs between gas and electricity.

**Possible approach: make the link with the work of Behrang and Quentin.**
### Investment and green finance
The aim here would be to complete Res-IRF with the general equilibrium part of IMACLIM-R France. More precisely, the investment demand generated by Res-IRF would impact (i) household savings and credit demand, (ii) the dynamics of the construction/energy equipment sector and (iii) public finances (tax revenues and subsidy expenditures).
Ideally, this development would include the creation of a banking sector in IMACLIM-R-France and a central bank. This would allow for the modeling of the green bond (Reminder: Philippe and Gaëtan have been asked by the Evaluation Committee). We can also imagine making the link with a real estate market (very simplified as long as it is not spatialized).

**Development in progress: FRITE project with ADEME**
## Cross-cutting: Evaluation of public policies
### Integration of health effects
Our new agreement with the CGDD focuses on the integration of the effects of thermal discomfort and local pollution (in particular particles linked to wood-energy) on health. It is planned to obtain estimates of these effects and to apply them in post-processing to the model outputs. An ENPC internship is planned for the spring on this subject.
On this subject, we could go further by having a more general economic reflection on the nature of health problems. In the standard model, consumers optimize their level of protection/vigilance to the temperature. According to the envelope theorem, comfort-related welfare changes (which can be approximated by health expenditures in the broad sense, which may include buying a sweater, etc.) are of second order to the bill savings from improved energy efficiency (assuming that the latter was previously impeded by a market failure...environmental externality or information asymmetry). If these variations are in fact not negligible, what does this reveal? A pecuniary externality linked to the fact that our health system is mutualized? A market failure in access to care, perhaps due to information asymmetries (but isn't the CMU supposed to solve them?)? Behavioral failures", whose welfare implications (and libertarian-paternalist interventions) are debated?

**Possible track: 2019 convention with CGDD. Not very conclusive for the moment...Talk to Dorothée Charlier, who is working on the subject?**
### New policies
The model has been used in the past to model the renovation obligation (in our 2011 report to the CGDD) and the modulation of real estate taxation (property tax and transfer tax) according to housing performance (Fuk Chun Wing and Kiefer, 2015). To the extent that these instruments are still being discussed (notably through the decency decree, which would prohibit F and G housing from being rented), it might be useful to repeat the exercise with the new version of the model.

**Proposed track: CNRS PEPS project with Matthieu Glachant, rejected two years in a row.**
### New data
In addition to the planned exploitation of the Eco-PTZ data, we could consider using the CITE data which will soon, if I understand correctly, be generated by the ANAH.  
**Possible approach: Collaboration with ANAH (with whom exchanges have not always been easy in the past). (ii) Recent exchanges with Antoine Bozio of the IPP, possibilities of coupling with their model of socio-fiscal system**
