# immune-synapse-metamodeling
Source code for metamodeling of the immune synapse

For paper Neve-Oz et al., 2021 (name TBD)

<ADD DETAILS HERE>
  
Metamodel_py

Folders:
=============================
Model1/ - Kinetic segregation (KSEG).
Model2/ - Lck activation (LCKA).
Model3/ - TCR phosphorylation (TCRP).
Coupled_model/ - Coupling (C).

=============================
Model1/ - Kinetic segregation:

Folders:
===================
Input/ - Raw input data for the model.
Processing/ - File for aranging the input data so it can be use by the surrogate model.
Surrogate/ - Files for learning and training the model.
Output/ - Files to be used by the coupled model.

===================

1. Pre-processing
2. Pre-modeling
2. Create model info
4. Surrogate

=============
5. Outpt

1. Input: 
	1.1 Reading raw training data from 'Input/'.
2. Processing:
	2.1 Reading raw training data for the model.
	2.2 Cropping, selecting and scaling the raw data. Assigning values and units for x 			and y axes.
	2.3 
    2.5 Plot training data.	
	
	
	
	
	
2. Pre modeling (finding initial fit parameters).
    2.1 Define fit equations and parameters.
    2.2 Get fit parameters.
    2.3 Create fitted data.
    2.4 Plot fitted data.

3. Create tables for the model.
    3.1
    3.2
    3.3
    3.4
    3.5
4. Create surrogate model with pymc3.
    4.1 Create untrained pymc3 model.
    4.2 Create trained pymc3 model.
    4.3 Create a fine mesh surrogate model based on the trained parameters.
5. Outputs.
    5.1
    5.2
    5.3
=============================

