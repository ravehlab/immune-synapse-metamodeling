# Immune-synapse-metamodeling
Source code for metamodeling of the immune synapse

For paper Neve-Oz et al., 2021 (name TBD)

<ADD DETAILS HERE>
  
### Metamodel_py:
Metamodel<br>
Folders:<br>
Surrogate_Models/<br>
|--- Model1/<br>
|----<br>
|--- Model2/<br>
|--- Model3/<br>
Input_Models/<br>
--- Model1/<br>
--- Model2/<br>
--- Model3/<br>
Coupled_Model<br>


Model1/ - Kinetic segregation (KSEG).<br>
Model2/ - Lck activation (LCKA).<br>
Model3/ - TCR phosphorylation (TCRP).<br>
Coupled_Model/ - Coupling (C).<br>

Every 'Model#' contains the same folders:<br>

Processing/ - File for aranging the input data so it can be use by the surrogate model.<br>
Code/ - Files for learning and training the model.<br>

Input/ - Raw input data for the model.<br>

