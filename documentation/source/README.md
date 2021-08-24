# Res-IRF
## Disclaimer
**_The contents of this repository are all in-progress and should not be expected to be free of errors or to perform any specific functions. Use only with care and caution._**

## About the authors
The development of the Res-IRF model was initiated at CIRED in 2008. Coordinated by Louis-Gaëtan Giraudet, it involved over the years, in alphabetic order, Cyril Bourgeois, Frédéric Branger, David Glotin, Céline Guivarch and Philippe Quirion.

## Overview
> The Res-IRF model is a tool for **simulating energy consumption for space heating** in the French residential sector.  Its main characteristic is to integrate a detailed description of the energy performance of the dwelling stock with a rich description of household behaviour. Res-IRF has been developed to improve the behavioural realism that integrated models of energy demand typically lack.

## Installation
**Step 1**: Git **clone Res-IRF folder** in your computer.
   - Use your terminal and go to a location where you want to store the Res-IRF project.
   - `git clone https://github.com/lucas-vivier/Res-IRF.git`

**Step 2**: **Create a conda environment** from the environment.yml file:
   - The environment.yml file is in the Res-IRF folder.
   - Use the **terminal** and go to the Res-IRF folder stored on your computer.
   - Type: `conda env create -f environment.yml`

**Step 3**: **Activate the new environment**.
   - The first line of the yml file sets the new environment's name.
   - Type: `conda activate myenv`

**Step 4**: **Launch Res-IRF**
   - `python main.py`

## Usage example
Project include different scripts and notebook to run the output and analyse its output.  
The standard way to launch Res-IRF:  

**Step 1: Launch Res-IRF main script.**  
The model create results in a folder in project/output.  
Folder name is by default `ddmmyyyy_hhmm` (launching date and hour).
Results are mainly .pkl (serialize format by the pickle library) or .csv file, and are not designed to be directly readable.  
NB: One file 'summary.csv' summarize important outputs.

**Step 2: Launch one of the Jupyter Notebook analysis tool**  
There are 3 main notebooks:
- `user_interface.ipynb`: macro and micro output analysis.
- `compare.ipyb`: macro and micro output comparison.
- `assess_public_poclies.ipyb`: macro and micro output comparison and calculation of efficiency and effectiveness. 

## Documentation

Documentation is available on https://lucas-vivier.github.io/Res-IRF/

## Meta

Lucas Vivier – [@VivierLucas](https://twitter.com/VivierLucas) – vivier@centre-cired.fr

Distributed under the GNU GENERAL PUBLIC LICENSE. See ``LICENSE`` for more information.

[https://github.com/lucas-vivier/Res-IRF](https://github.com/lucas-vivier/Res-IRF)