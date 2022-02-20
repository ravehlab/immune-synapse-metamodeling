Model1:
Structure:
1. Input
2. Processing
3. Surrogate
4. Outpt

1. Input: 
	1.1 Raw input data that is used for training the Model.
2. Processing:
	2.1.1 Reading raw training data for the model.
	2.1. Pre processing the data.
    2.1.2 Crop and scale data, assign values and units for x and y axes.
    2.1.3 Arange training data in 'pandas' dataFrame.
    2.1.4 Plot training data.	
	
	
	
	
	2.2 Cropping, selecting and scaling the raw data. Assigning values values and units for x and y axes.
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
