# Immune-synapse-metamodeling
Source code for metamodeling of the immune synapse

For paper Neve-Oz et al., 2021 (name TBD)

<ADD DETAILS HERE>
  
### Metamodel_py:
#### Structure:
Metamodel/<br>
Folders:<br>
Surrogate_Models/ - <br>
Input_Models/ - <br>
Coupled_Model/ - <br>

Surrogate_Models/<br>
Folders:<br>
Model1/<br>
Model2/<br>
Model3/<br>

Surrogate_Models/Model*/<br>
Folders:<br>
Code - <br>
Processing - <br>
Output (generated) - <br>

Input_Models/<br>
Folders:<br>
Model1/<br>
Model2/<br>
Model3/<br>


Model1/ - Kinetic segregation (KSEG).<br>
Model2/ - Lck activation (LCKA).<br>
Model3/ - TCR phosphorylation (TCRP).<br>
Coupled_Model/ - Coupling (C).<br>

Every 'Model#' contains the same folders:<br>

Processing/ - File for aranging the input data so it can be use by the surrogate model.<br>
Code/ - Files for learning and training the model.<br>

Input/ - Raw input data for the model.<br>

