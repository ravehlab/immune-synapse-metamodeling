# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pymc3 as pm
import numpy as np
# Untrained model:


def get_pm_model4_untrained(df_trainingData_model4,
                            df_model4_untrainedTable):

    pm_model4_untrained = pm.Model()
    with pm_model4_untrained:

        dfRV = df_model4_untrainedTable
        DP = 'Distribution parameters'

        x_obs = df_trainingData_model4.loc[:, 'Decaylength_nm'].values
        y_obs = df_trainingData_model4.loc[:, 'Depletion_nm'].values
        z_obs = df_trainingData_model4.loc[:, 'RgRatio'].values

        # fp_Decaylength_TCRP4
        ID = 'fp_Decaylength_TCRP4'
        fp_Decaylength_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=x_obs)

        # fp_Depletion_TCRP4
        ID = 'fp_Depletion_TCRP4'
        fp_Depletion_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=y_obs)

        # RgRatio_TCRP
        """TODO: read parameters values from RV table"""
        # rv_DecaylengthMin_RgRatio_TCRP4
        ID = 'rv_DecaylengthMin_RgRatio_TCRP4'
        rv_DecaylengthMin_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthMax_RgRatio_TCRP4
        ID = 'rv_DecaylengthMax_RgRatio_TCRP4'
        rv_DecaylengthMax_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthCen_RgRatio_TCRP4
        ID = 'rv_DecaylengthCen_RgRatio_TCRP4'
        rv_DecaylengthCen_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthDev_RgRatio_TCRP4
        ID = 'rv_DecaylengthDev_RgRatio_TCRP4'
        rv_DecaylengthDev_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionSlope_RgRatio_TCRP4
        ID = 'rv_DepletionSlope_RgRatio_TCRP4'
        rv_DepletionSlope_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=0.2)  # eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_RgRatio_TCRP4'
        rv_output_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=rv_DecaylengthMin_RgRatio_TCRP4 +
            (rv_DecaylengthMax_RgRatio_TCRP4 -
             rv_DecaylengthMin_RgRatio_TCRP4) *
            np.exp((fp_Decaylength_TCRP4 - rv_DecaylengthCen_RgRatio_TCRP4) /
                   rv_DecaylengthDev_RgRatio_TCRP4) +
            rv_DepletionSlope_RgRatio_TCRP4*fp_Depletion_TCRP4,
            sd=0.2,  # eval(dfRV.loc[ID, DP]['sd']),
            observed=z_obs)

    return pm_model4_untrained
#################################################

# Trained model:


"""
Return model1 trained based on trace1.
If observed_t and/or observed_k are specified,
return the model conditioned on those values.
"""


def get_pm_model4_trained(df_model4_trainedTable,
                          observed_Decaylength=None,
                          observed_depletion=None):

    pm_model4_trained = pm.Model()
    with pm_model4_trained:

        dfRV = df_model4_trainedTable
        DP = 'Distribution parameters'

        rv_Decaylength = pm.Normal('rv_Decaylength', mu=100, sd=50,
                                   observed=observed_Decaylength)
        rv_Depletion = pm.Normal('rv_depletion', mu=100, sd=50,
                                 observed=observed_depletion)

        # RgRatio_TCRP
        """TODO: read parameters values from RV table"""
        # rv_DecaylengthMin_RgRatio_TCRP4
        ID = 'rv_DecaylengthMin_RgRatio_TCRP4'
        rv_DecaylengthMin_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthMax_RgRatio_TCRP4
        ID = 'rv_DecaylengthMax_RgRatio_TCRP4'
        rv_DecaylengthMax_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthCen_RgRatio_TCRP4
        ID = 'rv_DecaylengthCen_RgRatio_TCRP4'
        rv_DecaylengthCen_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthDev_RgRatio_TCRP4
        ID = 'rv_DecaylengthDev_RgRatio_TCRP4'
        rv_DecaylengthDev_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionSlope_RgRatio_TCRP4
        ID = 'rv_DepletionSlope_RgRatio_TCRP4'
        rv_DepletionSlope_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=0.2)  # eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_RgRatio_TCRP4'
        rv_output_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=rv_DecaylengthMin_RgRatio_TCRP4 +
            (rv_DecaylengthMax_RgRatio_TCRP4 -
             rv_DecaylengthMin_RgRatio_TCRP4) *
            np.exp((rv_Decaylength - rv_DecaylengthCen_RgRatio_TCRP4) /
                   rv_DecaylengthDev_RgRatio_TCRP4) +
            rv_DepletionSlope_RgRatio_TCRP4*rv_Depletion,
            sd=eval(dfRV.loc[ID, DP]['sd']))

    return pm_model4_trained
#################################################
