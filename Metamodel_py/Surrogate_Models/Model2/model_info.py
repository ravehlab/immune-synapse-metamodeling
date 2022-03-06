# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 16:54:11 2022

@author: yairn
"""

import pandas as pd
# import pymc3 as pm
from IPython.display import display

"""
Every entity is a column header in the table.

id: str, # Unique, cross models, variable name.
type2: str,  # Type of variable, e.g. 'Free parameter', 'Random variable'.
shortName: str,  # Short name, e.g. 't'.
texName: str,  # LaTex name for display.
description: str,  "# Despcription of the variable, e.g. 'Time'.
distribution: str,  # Distribution type, e.g. 'Normal'.
distributionParameters: dict,  # Distribution parameters, e.g. ['mu', 'sd'].
units: str):  # Units of the variable, e.g. 'sec'.
"""


class RV:
    def __init__(self,
                 id: str,
                 type2: str,
                 shortName: str,
                 texName: str,
                 description: str,
                 distribution: str,
                 distributionParameters: dict,
                 units: str):

        self.id = id
        self.type2 = type2
        self.shortName = shortName
        self.texName = texName
        self.description = description
        self.distribution = distribution
        self.distributionParameters = distributionParameters
        self.units = units

    def get_as_dictionary(self):
        return {'ID': self.id,
                'Type': self.type2,
                'Short Name': self.shortName,
                'Latex Name': self.texName,
                'Description': self.description,
                'Distribution': self.distribution,
                'Distribution parameters': self.distributionParameters,
                'Units': self.units}

    def get_pymc3_statement():
        '''
        TASK 1
        TODO make this return a 2-tuple from a variable name
        to a pymc3 statment for creating this random variable
        (to be used as input for eval)
        '''
#         if RV.distribution == "Normal":
#             mu = RV.distributionParameters["mu"]
#             sd = RV.distributionParameters["sd"]
#             s0 = RV.id
#             if RV.shortName == "output":
#                 print(RV.shortName)
#             s1 = ("pm." + RV.distribution + "('" + RV.id  + "'" + \
#                                       ", mu=" + str(mu) + \
#                                       ", sd=" + str(sd) + ")")
#             s = (s0,s1)
#             print(eval("s[0]"),"=",eval("s[1]"))
        '''
        Example: return tuple :
        s = ('rv_alpha', 'pm.Normal("rv_alpha", mu=354, sigma=a*10+b*20)')
        so we can do eval(s[0]) = eval(s[1])
        '''
        # TODO: WRITE-ME
#       return

    @staticmethod
    def RV_from_dictionary(d: dict):
        ''' generates an RV object from a dictionary produced by
        get_as_dictionary() '''
        return RV(id=d['ID'],
                  type2=d['Type'],
                  shortName=d['Short Name'],
                  texName=d['Latex Name'],
                  description=d['Description'],
                  distribution=d['Distribution'],
                  distributionParameters=d['Distribution parameters'],
                  units=d['Units'])

#################################################
# Class model:


model2_description = """Distributions and inter distances of TCR and CD45
molecules that result from the early contact of a T cell and APC
(Antigen Presenting Cell)."""


class Model:
    #  Constructor
    def __init__(self,
                 shortName: str,
                 longName: str,
                 description: str,
                 model_id: str,
                 RV_csv_file=None,
                 data_csv_file=None):

        self.shortName = shortName
        self.longName = longName
        self.description = description
        self.model_id = model_id
        self.set_RVs_from_csv(RV_csv_file)
        # from CSV file
        self.set_data_from_csv(data_csv_file)
        # from CSV file

    # add a random variable to the model
    def add_rv(self, rv):
        self.RVs.append(rv)

    def get_dataframe(self):
        info = [rv.get_as_dictionary() for rv in self.RVs]
        df = pd.DataFrame(info)
        df.set_index('ID', drop=False)
        return df

    def to_csv(self, csv_file):
        df = self.get_dataframe()
        df.to_csv(csv_file)

    def set_RVs_from_csv(self, csv_file):
        '''
        read csv file (similar to Table S1 in metamodeling paper)
        with random variables and set this model's random variables
        and the statistical relations among them accordingly.
        If csv_file is None, set an empty list of RVs
        '''
        self.RVs = []
        if csv_file is None:
            return
        df = pd.read_csv(csv_file)
        rv_dicts = df.to_dict('records')
        print("RV dicts from csv:")
        print(rv_dicts)
        for rv_dict in rv_dicts:
            rv = RV.from_dictionary(rv_dict)
            self.add_rv(rv)

    def set_data_from_csv(self, data_csv_file):
        # TASK 2
        # df = pd.read_csv(data_csv_file)
        # display(df) # Yair
        # TODO: code for filling in table of data
        # self.data = ... # WRITE-ME
        self.trainingData = pd.read_csv(data_csv_file)

    # generate a pymc3 model from this model
    def get_as_pymc3(self):
        '''
        Go over all random variables in this model,
        and generate a PyMC3 object with cooresponding
        variable names and statistical relations among them
        '''
        # TODO (use "eval" command)
        # pm_model = pm.Model()
        # with pm_model as pm:
        #      for rv in self.RVs:
        #          pass
        #  s = rv.get_pymc3_statement()
        #  eval(s[0]) = eval(s[1])
        # return pm_model

    def update_rvs_from_pymc3(self, pymc3):  # BARAK
        # TASK 4
        # TODO: use trace from trained PyMC3 model to update
        # statements for all RVs
        return
#################################################
# Start model1_dep:


model2_DL = Model(shortName='LCKA',
                  longName='Lck activation',
                  description='Model2 description',
                  model_id='2',
                  RV_csv_file=None,
                  data_csv_file='trainingData_model2.csv')
#################################################
# Define dep untrained table:


def model2DL(df_fitParameters_DL):

    model2_DL.add_rv(
        RV(id='fp_Poff_DL_LCKA2',
           type2='Free parameter',
           shortName='Poff',
           texName="$$Poff^{LCKA}$$",
           description='Decay probability',
           distribution='Normal',
           distributionParameters={
               'mu': str(-3.),
               'sd': str(1.)},
           units='$$-$$'))

    model2_DL.add_rv(
        RV(id='fp_Diff_DL_LCKA2',
            type2='Free parameter',
            shortName='Diff',
            texName='$$Diff^{LCKA}$$',
            description='Diffusion constant',
            distribution='Normal',
            distributionParameters={
                'mu': str(-2.),
                'sd': str(1.)},
            units='$$\mum^2/sec$$'))

    model2_DL.add_rv(
        RV(id='rv_intercept_DL_LCKA2',
            type2='Random variable',
            shortName='intercept',
            texName='$$DL^{LCKA}_{intercept}$$',
            description='Interception with z axis',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_DL.loc['intercept', 'mu']),
                'sd': str(df_fitParameters_DL.loc['intercept', 'sd'])},
            units='$$nm$$'))

    model2_DL.add_rv(
        RV(id='rv_PoffSlope_DL_LCKA2',
            type2='Random variable',
            shortName='PoffSlope',
            texName='$$Poff^{LCKA}_{PoffSlope}$$',
            description='Slope in Poff direction',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_DL.loc['xSlope', 'mu']),
                'sd': str(df_fitParameters_DL.loc['xSlope', 'sd'])},
            units='$$-$$'))

    model2_DL.add_rv(
        RV(id='rv_DiffSlope_DL_LCKA2',
            type2='Random variable',
            shortName='DiffSlope',
            texName='$$Diff^{LCKA}_{DiffSlope}$$',
            description='Slope in Diff direction',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_DL.loc['ySlope', 'mu']),
                'sd': str(df_fitParameters_DL.loc['ySlope', 'sd'])},
            units='$$\mum^2/sec$$'))

    model2_DL.add_rv(
        RV(id='rv_output_DL_LCKA2',
           type2='Random variable',
           shortName='output',
           texName='$$Diff^{LCKA}_{output}$$',
           description='Decay length output',
           distribution='Normal',
           distributionParameters={'mu': str(500.), 'sd': str(100.)},
           units="$$nm$$"))

    model2_DL.to_csv("model2/Model2_DL.csv")

    return(model2_DL)
#################################################
# Display RVs tables as DataFrames:
# Display untrained table:
# df_model1_untrainedTable = model1_dep.get_dataframe()
# df_model1_untrainedTable = df_model1_untrainedTable.set_index('ID')
# display(df_model1_dep_untrainedTable):


def displayInfo(model2_DL):

    df_model2_untrainedTable = model2_DL.get_dataframe()
    df_model2_untrainedTable = df_model2_untrainedTable.set_index('ID')

    display(df_model2_untrainedTable.style.set_properties(
        **{'text-align': 'left',
           'background-color': 'rgba(200, 150, 255, 0.65)',
           'border': '1px black solid'}))

#################################################
