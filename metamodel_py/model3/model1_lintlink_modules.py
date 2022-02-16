# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 17:59:39 2022

@author: yairn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import pymc3 as pm

#################################################
# Set what data to plot:


def plotData(DataToPlot, plotWhat):
    # titles and labels:
    xLabel = "$t(sec)$"
    yLabel = "$\kappa(kTnm^2)$"

    dep_Title = "Depletion \n"
    colTitles = [dep_Title]

    rowTitles = ["Training data", "Data fit",
                 "Trained parameters", "Surrogate"]

    # min and max values for the different heatmaps:
    vmins = [0]
    vmaxs = [250]

    nRows = 4  # Todo: fix
    nCols = 1  # Todo: fix

    # Plot a row of subplot if the data is not empty and if value is 'True':
    for iRow in range(nRows):
        if DataToPlot[iRow] is not None and plotWhat[iRow]:
            plotHeatmaps(data=DataToPlot[iRow],
                         nRows=nRows,
                         nCols=nCols,
                         rowTitles=rowTitles,
                         colTitles=colTitles,
                         xLabel=xLabel,
                         yLabel=yLabel,
                         vmins=vmins,
                         vmaxs=vmaxs,
                         iRow=iRow)
#################################################
# Plot heatmaps subplots


def plotHeatmaps(
        data,
        rowTitles,
        colTitles,
        xLabel,
        yLabel,
        nRows,
        nCols,
        vmins,
        vmaxs,
        iRow):

    fig = plt.figure(figsize=[4, 12])

    # im identifier:
    im = [None]*nCols
    x1, x2 = data[0]  # Free parameters of the data.
    f = data[1]  # Data that is the function of the free parameters.

    # Return the last row that is 'True' in 'plotWhat'. It sets in what row
    # to show 'xlabel':

    max_plotWhat = 4  # np.max(np.where(plotWhat))

    # plot the nRows x nCols subplots with labels, titles at
    # sceciefic locations. iCol is Column index, iRow is Row index:
    for iCol in range(nCols):
        fig.add_subplot(nRows, nCols, iRow*nCols + iCol+1)
        im[iCol] = plt.pcolor(x1, x2, f[iCol],
                              vmin=vmins[iCol], vmax=vmaxs[iCol],
                              shading='auto', cmap='Purples')
        if 1:  # iRow > 0:
            dep_levels = np.arange(25., 250., 25.)
            cs = plt.contour(x1, x2, f[iCol], dep_levels, colors='k',
                             vmin=vmins[iCol], vmax=vmaxs[iCol])
            plt.clabel(cs, dep_levels, inline=True,fmt='%.0f', fontsize=10)
            ###
            # fig, ax = plt.subplots()
            # CS = ax.contour(X, Y, Z)
            # ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
            ###

        fig.colorbar(im[iCol])

        if iRow == 0:
            plt.title(colTitles[iCol] + rowTitles[iRow])

        else:
            plt.title(rowTitles[iRow])

        if iRow == max_plotWhat:
            plt.xlabel(xLabel)

        if iCol == 0:
            plt.ylabel(yLabel)

    fig.tight_layout()

#################################################
# Read training data:


def readTrainingData():


    dep_training_data0_pixels = np.array([[21.443,2.146,1.8681,3.8138,3.9293,5.6677,5.3346,4.3738,4.2115,5.28,8.2552],
    [24.357,5.1339,6.7392,8.223,9.3518,10.359,11.295,10.972,12.868,13.388,13.722],
    [20.173,5.2959,7.333,7.8585,10.086,11.09,11.441,12.287,12.422,13.599,14.602],
    [21.048,6.5607,8.4275,10.31,11.318,12.677,13.688,14.361,14.064,14.734,15.726],
    [25.645,5.9201,9.4306,10.969,12.123,12.807,13.398,14.336,14.965,15.327,16.625],
    [21.139,7.0034,9.7752,10.405,11.668,12.448,13.244,15.181,15.767,16.214,18.283],
    [21.274,6.9867,9.6211,11.519,13.453,14.475,15.672,17.101,17.715,18.632,19.162],
    [23.961,8.0956,10.543,11.696,13.353,14.386,15.99,17.091,18.055,18.893,19.542],
    [22.423,8.268,10.87,11.669,13.451,15.054,16.721,17.912,18.917,19.362,19.769],
    [27.017,7.3613,10.131,12.155,13.812,15.167,16.219,16.971,18.294,18.95,19.332],
    [20.636,8.0815,10.922,12.998,14.427,15.393,16.38,17.264,17.915,19.277,20.381],
    [19.696,9.7434,12.274,13.932,15.715,17.327,18.2,19.712,20.157,21.347,22.44],
    [22.62,8.7923,11.49,12.903,14.329,15.736,16.811,17.696,18.532,19.141,20.687],
    [21.159,8.8581,10.963,13.294,14.779,16.134,17.312,18.822,19.915,21.127,22.049],
    [23.141,7.6794,10.896,12.606,14.96,16.412,17.242,17.534,19.21,20.28,20.491],
    [24.11,7.9982,10.606,11.875,13.188,14.998,16.305,17.323,18.873,20.375,20.659],
    [25.384,7.243,10.881,13.414,15.346,16.97,18.282,18.986,20.141,21.525,22.371],
    [21.893,8.3062,11.252,12.743,14.472,16.041,17.768,19.057,20.07,20.825,21.393],
    [23.24,7.2108,10.11,12.491,14.012,16.15,17.421,18.109,19.321,20.581,22.27],
    [23.722,9.4973,11.47,13.331,15.163,17.049,17.551,18.817,20.326,20.695,22.219]])

    return dep_training_data0_pixels
#################################################
# scaling data to nanometer and cropping:


def cropAndScaleRawData(dep_training_data0_pixels):
    nmToPixels = 10

    dep_training_data0_nm = nmToPixels*dep_training_data0_pixels

    size_k, size_t = np.shape(dep_training_data0_nm)
    # x-axis for arrays (time in seconds):
    t0 = np.linspace(0, 100, size_t)
    # y-axis for arrays (rigidity in kT*nm^2):
    k0 = np.linspace(100/size_k, 100, size_k)

    # title Crop and save training data

    # select start indices for t and k:
    t_start_index = 1
    k_start_index = 1

    selected_t_indices = np.arange(t_start_index, size_t, 1)
    # every second index:
    selected_k_indices = np.arange(k_start_index, size_k, 2)

    dep_training_data1_nm = dep_training_data0_nm[selected_k_indices, :]
    dep_training_data_nm = dep_training_data1_nm[:, selected_t_indices]

    t = t0[selected_t_indices]
    k = k0[selected_k_indices]

    [t_array, k_array] = np.meshgrid(t, k)

    if True:
        np.save("dep_training_data_nm.npy", dep_training_data_nm)

    return t_array, k_array, dep_training_data_nm

#################################################
# Training data to dataFrame:


def trainingDataToDataFrame(t_array, k_array, dep_training_data_nm):
    df_trainingData_model1 = pd.DataFrame(
        np.array([t_array.flatten(),
                  k_array.flatten(),
                  dep_training_data_nm.flatten()]).T,
        columns=['time_sec', 'k0_kTnm2', 'dep_nm'])

    df_trainingData_model1.to_csv('trainingData_model1.csv')

    df_dep_nm = df_trainingData_model1.pivot('k0_kTnm2', 'time_sec', 'dep_nm')

    return df_dep_nm
#################################################
# fitFunction for dep:


def sigxsigy(xy,
             tMin, tMax, tCenter, tDevisor,
             kMin, kMax, kCenter, kDevisor):

    x, y = xy
    fx = tMin + (tMax - tMin)/(1 + np.exp(-(x - tCenter)/tDevisor))
    fy = kMin + (kMax - kMin)/(1 + np.exp(-(y - kCenter)/kDevisor))
    f = fx + fy
    strf = "tMin + (tMax - tMin)/(1 + np.exp(-(x-tCenter)/tDev)) \
    + kMin + (kMax - kMin)/(1 + np.exp(-(y-kCenter)/kDev))"

    return f, strf
#################################################
# fitFunction for hMean:


def sigsig_z(xy,
             tScale, tCen, tDev,
             kScale, kCen, kDev):

    x, y = xy
    fx = tScale/(1 + np.exp(-(x-tCen)/tDev))
    fy = kScale/(1 + np.exp(-(y-kCen)/kDev))
    f = fx + fy
    strf = "tScale/(1 + np.exp(-(x-tCen)/tDev)) + \
        kScale/(1 + np.exp(-(y-kCen)/kDev))"

    return f

#################################################
# Get fit parameters


def get_fit_parameters(X, fitFunc, fXdata, parametersNames, p0):
    """
    Returns fit parameters and aranges them in DataFrames where the index
    (rows) are the fit parameters' names and the columns are 'mu' and 'sd'.
    """
    popt, pcov = curve_fit(fitFunc, X, fXdata, p0)
    mu = popt
    sd = np.sqrt(np.diag(pcov))

    data = {'mu': mu, 'sd': sd}
    index = parametersNames

    df = pd.DataFrame(data, index=index)

    return df
#################################################
# Fit parameters for dep:


def fit1(flatten_t, flatten_k, flatten_dep_nm):
    p0_dep = 0., 50., 0., 20., 0., 50., 0., 20.
    parametersNames_dep = ['tMin', 'tMax', 'tCenter', 'tDevisor',
                           'kMin', 'kMax', 'kCenter', 'kDevisor']

    "strf = tMin + (tMax - tMin)/(1 + np.exp(-(x-tCen)/tDevisor)) + \
        kMin + (kMax - kMin)/(1 + np.exp(-(y-kCen)/kDevisor))"

    # equation_dep = parametersNames_dep[0] + \
    #                 "+" + \
    #                 parametersNames_dep[1] + \
    #                 "*" + \
    #                 "t" + \
    #                 "+" + \
    #                 parametersNames_dep[2] +\
    #                 "*" + \
    #                 "k"

    df_fitParameters_dep = get_fit_parameters(X=(flatten_t, flatten_k),
                                              fitFunc=sigxsigy,
                                              fXdata=flatten_dep_nm,
                                              parametersNames=parametersNames_dep,
                                              p0=p0_dep)

    df_fitParameters_dep = df_fitParameters_dep.round(3)
    df_fitParameters_dep
    # print(equation_dep)


#################################################
# Fit parameters for dep:

def fit2(flatten_t, flatten_k, flatten_dep_nm):
    p0_dep = 100., 0., 20., 100., 0., 20.

    parametersNames_dep = ['tScale', 'tCenter', 'tDevisor',
                           'kScale', 'kCenter', 'kDevisor']

    df_fitParameters_dep = get_fit_parameters(X=(flatten_t, flatten_k),
                                              fitFunc=sigsig_z,
                                              fXdata=flatten_dep_nm,
                                              parametersNames=parametersNames_dep,
                                              p0=p0_dep)

    df_fitParameters_dep = df_fitParameters_dep.round(3)

    return df_fitParameters_dep
#################################################
# Fitted data:


def fittedData(df_fitParameters_dep, t, k):

    t_array, k_array = np.meshgrid(t, k)

    tScale_fit = df_fitParameters_dep.loc['tScale', 'mu']
    tCen_fit = df_fitParameters_dep.loc['tCenter', 'mu']
    tDev_fit = df_fitParameters_dep.loc['tDevisor', 'mu']
    kScale_fit = df_fitParameters_dep.loc['kScale', 'mu']
    kCen_fit = df_fitParameters_dep.loc['kCenter', 'mu']
    kDev_fit = df_fitParameters_dep.loc['kDevisor', 'mu']

    dep_fit = tScale_fit/(1 + np.exp(-(t_array - tCen_fit)/tDev_fit)) + \
        kScale_fit/(1 + np.exp(-(k_array - kCen_fit)/kDev_fit))

    return dep_fit
#################################################
# Class RV (random variable)


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

        self.id = id  # str
        self.type2 = type2  # str
        self.shortName = shortName  # str
        self.texName = texName  # str
        self.description = description  # str
        self.distribution = distribution  # str
        self.distributionParameters = distributionParameters  # array
        self.units = units  # str

    def get_as_dictionary(self):
        return {'ID': self.id,  # Unique variable name that is used by pymc3
                'Type': self.type2,  # Type of variable, e.g. 'Free parameter', 'Random variable'
                'Short Name': self.shortName,  # Short name, e.g. 't'.
                'Latex Name': self.texName,  # LaTex name for display.
                'Description': self.description,  #
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
# Class model


model1_description = """Distributions and inter distances of TCR and CD45
 molecules that result from the early contact of a T cell and APC
 (Antigen Presenting Cell)."""


class Model:
    #  Constructor
    def __init__(self,
                 shortName: str,
                 longName: str,
                 description: str,
                 model_id: str,
                 RV_csv_file=None,  # Topology of Bayes net
                 data_csv_file=None):  # Training data
        '''
        '''
        self.shortName = shortName  # str
        self.longName = longName  # str
        self.description = description  # str
        self.model_id = model_id  # str
        self.set_RVs_from_csv(RV_csv_file)  # fill in random variables from CSV file
        self.set_data_from_csv(data_csv_file)  # fill in training data from CSV file

    # add a random variable to the model
    def add_rv(self, rv):
        self.RVs.append(rv)

    def get_dataframe(self):  #
        info = [rv.get_as_dictionary() for rv in self.RVs]
        df = pd.DataFrame(info)
        df.set_index('ID', drop=False)
        return df

    def to_csv(self, csv_file):  # TODO: think about the name
        df = self.get_dataframe()
        df.to_csv(csv_file)

    def set_RVs_from_csv(self, csv_file):
        ''' 
        read csv file (similar to Table S1 in metamodeling paper) with random variables
        and set this model's random variables and the statistical relations among them
        accordingly

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
        pm_model = pm.Model()
        with pm_model as pm:
             for rv in self.RVs:
                 pass
            #    s = rv.get_pymc3_statement()
            #    eval(s[0]) = eval(s[1])
        return pm_model

    def update_rvs_from_pymc3(self, pymc3):  # BARAK
        # TASK 4 
        # TODO: use trace from trained PyMC3 model to update statements for all RVs
        return
#################################################
# Start model1_dep
import model1_modules as m1

model1_dep = Model(shortName='KSEG',  # str
                   longName='Kinetic segregation',  # str
                   description='Model1 description',  # str
                   model_id='1',  # str
                   RV_csv_file=None,
                   data_csv_file='trainingData_model1.csv') 
#################################################
# Define dep untrained table


def model1Dep(df_fitParameters_dep):

    model1_dep.add_rv(
        RV(id='fp_t_dep_KSEG1',
           type2='Free parameter',
           shortName='t',
           texName="$$t^{KSEG}$$",
           description='Time',
           distribution='Uniform',
           distributionParameters={'lower': str(0.), 'upper': str(100.)},
           units='$$sec$$'))

    model1_dep.add_rv(
        RV(id='fp_k_dep_KSEG1',
            type2='Free parameter',
            shortName='k',
            texName='$$\kappa^{KSEG}$$',
            description='Membrane rigidity',
            distribution='Uniform',
            distributionParameters={'lower': str(0.), 'upper': str(100.)},
            units='$$kTnm^2$$'))

    model1_dep.add_rv(
        RV(id='rv_intercept_dep_KSEG1',
            type2='Random variable',
            shortName='intercept',
            texName='$$dep^{KSEG}_{intercept}$$',
            description='Interception with z axis',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_dep.loc['intercept', 'mu']),
                'sd': str(df_fitParameters_dep.loc['intercept', 'sd'])},
            units='$$nm$$'))

    model1_dep.add_rv(
        RV(id='rv_tSlope_dep_KSEG1',
            type2='Random variable',
            shortName='tSlope',
            texName='$$dep^{KSEG}_{tSlope}$$',
            description='Slope in t direction',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_dep.loc['tSlope', 'mu']),
                'sd': str(df_fitParameters_dep.loc['tSlope', 'sd'])},
            units='$$sec$$'))

    model1_dep.add_rv(
        RV(id='rv_kSlope_dep_KSEG1',
            type2='Random variable',
            shortName='kSlope',
            texName='$$dep^{KSEG}_{kSlope}$$',
            description='Slope in k direction',
            distribution='Normal',
            distributionParameters={
                'mu': str(df_fitParameters_dep.loc['kSlope', 'mu']),
                'sd': str(df_fitParameters_dep.loc['kSlope', 'sd'])},
            units='$$kTnm^2$$'))

    # model1_dep.to_csv("Model1_dep.csv")

    return(model1_dep)
#################################################
# Untrained model:


def get_pm_model1_untrained(df_trainingData_model1,
                            df_model1_untrainedTable):  # model1

    pm_model1 = pm.Model()
    with pm_model1:
        dfRV = df_model1_untrainedTable
        DP = 'Distribution parameters'
        # model1 - KS (kinetic segregation) #######################
        # from class ##############################################
        # model1.D:
        t_KSEG1_obs = df_trainingData_model1.loc[:, 'time_sec'].values
        k_KSEG1_obs = df_trainingData_model1.loc[:, 'k0_kTnm2'].values

        dep_KSEG1_obs = df_trainingData_model1.loc[:, 'dep_nm'].values

        # rv_t
        ID = 'fp_t_dep_KSEG1'
        rv_t = pm.Uniform('rv_t',
                          lower=dfRV.loc[ID, DP]['lower'],
                          upper=dfRV.loc[ID, DP]['upper'],
                          observed=t_KSEG1_obs)

        # rv_k
        ID = 'fp_k_dep_KSEG1'
        rv_k = pm.Uniform('rv_k',
                          lower=dfRV.loc[ID, DP]['lower'],
                          upper=dfRV.loc[ID, DP]['upper'],
                          observed=k_KSEG1_obs)  # todo values

        # dep_KSEG ####################################################
        """TODO: read parameters values from RV table"""
        # rv_intercept_dep_KSEG1
        ID = 'rv_intercept_dep_KSEG1'
        rv_intercept_dep_KSEG1 =\
            pm.Normal(ID,
                      mu=eval(dfRV.loc[ID, DP]['mu']),
                      sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_xSlope_dep_KSEG1
        ID = 'rv_xSlope_dep_KSEG1'
        rv_xSlope_dep_KSEG1 =\
            pm.Normal(ID,
                      mu=eval(dfRV.loc[ID, DP]['mu']),
                      sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_ySlope_dep_KSEG1
        ID = 'rv_ySlope_dep_KSEG1'
        rv_ySlope_dep_KSEG1 =\
            pm.Normal(ID,
                      mu=eval(dfRV.loc[ID, DP]['mu']),
                      sd=eval(dfRV.loc[ID, DP]['sd']))

        ID='rv_output_dep_KSEG1'
        rv_output_dep_KSEG1 =\
            pm.Normal(ID,
                      mu=rv_intercept_dep_KSEG1 +
                      rv_xSlope_dep_KSEG1 * rv_t +
                      rv_ySlope_dep_KSEG1 * rv_k,
                      sd=eval(dfRV.loc[ID, DP]['sd']),
                      observed=dep_KSEG1_obs)

    return pm_model1
#################################################













