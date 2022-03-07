# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 16:54:11 2022

@author: yairn
"""

import pandas as pd
# import pymc3 as pm
from IPython.display import display

from Model1.Code import definitions

paths = definitions.paths
submodels = definitions.submodels
"""
Every variable is a column header in the table:

id: str, # Unique, cross models, variable name.
type2: str,  # Type of variable, e.g. 'Free parameter', 'Random variable'.
shortName: str,  # Short name, e.g. 't'.
texName: str,  # LaTex name for display.
description: str,  "# Despcription of the variable, e.g. 'Time'.
distribution: str,  # Distribution type, e.g. 'Normal'.
distributionParameters: dict,  # Distribution parameters, e.g. ['mu', 'sd'].
units: str):  # Units of the variable, e.g. 'sec'.
"""


class RV:  # Random variable
    def __init__(self,
                 id: str,
                 varType: str,
                 shortName: str,
                 texName: str,
                 description: str,
                 distribution: str,
                 distributionParameters: dict,
                 units: str):

        self.id = id
        self.varType = varType
        self.shortName = shortName
        self.texName = texName
        self.description = description
        self.distribution = distribution
        self.distributionParameters = distributionParameters
        self.units = units

    def get_as_dictionary(self):
        return {'ID': self.id,
                'Variable type': self.varType,
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

    #  @staticmethod

    def RV_from_dictionary(d: dict):
        """generates an RV object from a dictionary produced by
        get_as_dictionary() """
        return RV(id=d['ID'],
                  varType=d['Variable type'],
                  shortName=d['Short Name'],
                  texName=d['Latex Name'],
                  description=d['Description'],
                  distribution=d['Distribution'],
                  distributionParameters=d['Distribution parameters'],
                  units=d['Units'])

#################################################
# Class model:


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
        self.set_data_from_csv(data_csv_file)

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
# Start model1_depletion:


submodel_variables = Model(
    shortName='KSEG',
    longName='Kinetic segregation',
    description='Model1 description',
    model_id='1',
    RV_csv_file=None,
    data_csv_file=paths['Input']+'df_trainingData_depletion_flatten.csv')


#################################################
# Define dep untrained table:

x = definitions.x
y = definitions.y
fps = [x, y]
fitParameters = definitions.fitParameters
fitParametersNames = submodels['names'][0]['fitParametersNames']


def submodelInfo(df_submodel_fitParameters):

    for fp in fps:
        submodel_variables.add_rv(
            RV(id=fp['ID'],
               varType=fp['varType'],
               shortName=fp['shortName'],
               texName=fp['texName'],
               description=fp['description'],
               distribution=fp['distribution'],
               distributionParameters=fp['distributionParameters'],
               units=fp['units']))

    for rvName in fitParametersNames:
        submodel_variables.add_rv(
            RV(id=fitParameters[rvName]['ID'],
                varType=fitParameters[rvName]['varType'],
                shortName=fitParameters[rvName]['shortName'],
                texName=fitParameters[rvName]['texName'],
                description=fitParameters[rvName]['description'],
                distribution=fitParameters[rvName]['distribution'],
                distributionParameters=
                fitParameters[rvName]['distributionParameters'],
                units=fitParameters[rvName]['units']))



    submodel_variables.add_rv(
        RV(id='rv_output_depletion_KSEG1',
           varType='Random variable',
           shortName='output',
           texName='$$dep^{KSEG}_{output}$$',
           description='dep output',
           distribution='Normal',
           distributionParameters={'mu': '',
                                   'sd': str(20.)},
           units="$$nm$$"))

    submodel_info.to_csv(
        "Model1_Info_depletion.csv")

    return(model1_depletion_info)
#################################################

