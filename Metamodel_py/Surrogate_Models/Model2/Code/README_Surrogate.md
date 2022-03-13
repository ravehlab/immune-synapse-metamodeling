# Surrogate

### Process:
1. Pre-processing<br>
	1.1 Reading raw training data from 'Input/'.<br>
	1.2 Cropping, selecting and scaling the raw data. Assigning values and units for x 			and y axes.<br>
    1.3 Arange training data in pandas dataFrames.<br>
    1.4 Plot training data.<br>

2. Pre-modeling<br>
    2.1 Define fit equations and parameters.<br>
    2.2 Get fit parameters.<br>
    2.3 Create fitted data.<br>
    2.4 Plot fitted data.<br>

3. Create model info<br>
	3.1 Define 'Random Variable' (RV) class.<br>
	3.2 Define 'Model' class.<br>
	3.3 Get untrained model info.<br>
	3.4 Create table with untrained model info.<br>
	
4. Create surrogate model with pymc3.<br>
    4.1 Create untrained pymc3 model.<br>
    4.2 Create trained pymc3 model.<br>
    4.3 Create a fine mesh surrogate model based on the trained parameters.<br>

5. Save to Output/.


