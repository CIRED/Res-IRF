# Res-IRF
## Disclaimer

**_The contents of this repository are all in-progress and should not be expected to be free of errors or to perform any
specific functions. Use only with care and caution._**

## Overview

> The Res-IRF model is a tool for simulating energy consumption and energy efficiency improvements in the French residential building sector. It currently focuses on space heating as the main usage. The rationale for its development is to integrate a detailed description of the energy performance of the dwelling stock with a rich description of household behaviour. Res-IRF has been developed to improve the behavioural realism that is typically lacking in integrated models of energy demand.

## Resources

Documentation is freely available on https://lucas-vivier.github.io/Res-IRF/

A simple user interface is available on http://resirf.pythonanywhere.com/ to give an overview of Res-IRF main output.


## Installation
**Step 1**: Git **clone Res-IRF folder** in your computer.
   - Use your terminal and go to a location where you want to store the Res-IRF project.
   - `git clone https://github.com/lucas-vivier/Res-IRF.git`

**Step 2**: **Create a conda environment** from the environment.yml file:
   - The environment.yml file is in the Res-IRF folder.
   - Use the **terminal** and go to the Res-IRF folder stored on your computer.
   - Type: `conda env create -f res-irf-env.yml`

**Step 3**: **Activate the new environment**.
   - The first line of the yml file sets the new environment's name.
   - Type: `conda activate Res-IRF`

**Step 4**: **Launch Res-IRF**
   - Launch from Res-IRF root folder:
   - `python project/main.py -n scenario.json`
   - `scenario.json` is the path to the configuration file
   
## Usage example
Project includes libraries, scripts and notebooks.  
`/project` is the folder containing scripts, notebooks, inputs and outputs.  

The standard way to launch Res-IRF:  

**Step 1: Launch Res-IRF main script.**  
The model creates results in a folder in project/output.  
Folder name is by default `ddmmyyyy_hhmm` (launching date and hour).
Results are mainly .pkl (serialize format by the pickle library), .csv file and some .png graphs.

A configuration file must be declared.
An example of configuration file is in the `input` folder under the name of `scenario.json`.
The Res-IRF script use Multiprocessing tool to launch multiple scenarios in the same time. 

In the `output/ddmmyyyy_hhmm` folder:
- One folder for each scenario declared in the configuration file with detailed outputs:
    - `summary_input.csv` summary of main input
    - `summary.csv` summary of main output
    - `detailed.csv` detailed output readable directly with an Excel-like tool
    - `summary_policies` results of public policies impact and cost
    - `.pkl` files contain exhaustive output but need to be read with a programing language
    - copy of `parameters.json` and scenario used for the run
- `.png` graphs that compare scenarios

**Step 2 : Launch one of the Jupyter Notebook analysis tool (in progress)**  
There are 4 main notebooks:
- `ui_unique.ipynb`: macro and micro output analysis.
- `quick_comparison.ipyb`: macro and micro output comparison.
- `ui_comparison.ipyb`: macro and micro output comparison.
- `policy_indicator.ipyb`: macro and micro output comparison and calculation of efficiency and effectiveness. 

Notebook templates are stored in `project/nb_template_analysis`.  
**Users should copy and paste the template notebook in another folder to launch it.**

## About the authors

The development of the Res-IRF model was initiated at CIRED in 2008. Coordinated by Louis-Gaëtan Giraudet, it involved
over the years, in alphabetic order, Cyril Bourgeois, Frédéric Branger, François Chabrol, David Glotin, Céline Guivarch,
Philippe Quirion, and Lucas Vivier.

## Meta

Lucas Vivier – [@VivierLucas](https://twitter.com/VivierLucas) – vivier@centre-cired.fr

Distributed under the GNU GENERAL PUBLIC LICENSE. See ``LICENSE`` for more information.

[https://github.com/lucas-vivier/Res-IRF](https://github.com/lucas-vivier/Res-IRF)