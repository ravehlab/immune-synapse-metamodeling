# immune-synapse-metamodeling
Source code for metamodeling of the immune synapse

For paper Neve-Oz et al., 2021 (name TBD)

<ADD DETAILS HERE>
  
Metamodel_py:
=============================

Folders:
=============================
Model1/ - Kinetic segregation (KSEG).
Model2/ - Lck activation (LCKA).
Model3/ - TCR phosphorylation (TCRP).
Coupled_model/ - Coupling (C).

=============================
Every model has the same folder names:

Folders:
=============================
Input/ - Raw input data for the model.
Processing/ - File for aranging the input data so it can be use by the surrogate model.
Surrogate/ - Files for learning and training the model.
Output/ - Files to be used by the coupled model.

=============================
Every model follow the same process:

Process:
=============================
1. Pre-processing
	1.1 Reading raw training data from 'Input/'.
	1.2 Cropping, selecting and scaling the raw data. Assigning values and units for x 			and y axes.
    1.3 Arange training data in pandas dataFrames.
    1.4 Plot training data.

2. Pre-modeling
    2.1 Define fit equations and parameters.
    2.2 Get fit parameters.
    2.3 Create fitted data.
    2.4 Plot fitted data.

3. Create model info
	3.1 Define 'Random Variable' (RV) class.
	3.2 Define 'Model' class.
	3.3 Get untrained model info.
	3.4 Create table with untrained model info.
	
4. Create surrogate model with pymc3.
    4.1 Create untrained pymc3 model.
    4.2 Create trained pymc3 model.
    4.3 Create a fine mesh surrogate model based on the trained parameters.

5. Save to Output/.
=============================

