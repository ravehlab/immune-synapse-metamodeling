# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pymc3 as pm

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

        # rv_decaylength
        ID = 'fp_decaylength_TCRP3'
        rv_decaylength = pm.Normal('rv_decaylength',
                                   mu=eval(dfRV.loc[ID, DP]['mu']),
                                   sd=eval(dfRV.loc[ID, DP]['sd']),
                                   observed=x_obs)

        # rv_depletion
        ID = 'fp_depletion_TCRP3'
        rv_depletion = pm.Normal('rv_depletion',
                                 mu=eval(dfRV.loc[ID, DP]['mu']),
                                 sd=eval(dfRV.loc[ID, DP]['sd']),
                                 observed=y_obs)

        # RgRatio_TCRP
        """TODO: read parameters values from RV table"""
        # rv_intercept_RgRatio_TCRP3
        ID = 'rv_intercept_RgRatio_TCRP3'
        rv_intercept_RgRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=0.2)  # eval(dfRV.loc[ID, DP]['sd']))

        # rv_decaylangthSlope_RgRatio_TCRP3
        ID = 'rv_decaylengthSlope_RgRatio_TCRP3'
        rv_decaylengthSlope_RgRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=0.2)  # eval(dfRV.loc[ID, DP]['sd']))

        # rv_depletionSlope_RgRatio_TCRP3
        ID = 'rv_depletionSlope_RgRatio_TCRP3'
        rv_depletionSlope_RgRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=0.2)  # eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_RgRatio_TCRP3'
        rv_output_RgRatio_TCRP3 = pm.Normal(
            ID,
            mu=rv_intercept_RgRatio_TCRP3 +
            rv_decaylengthSlope_RgRatio_TCRP3*rv_decaylength +
            rv_depletionSlope_RgRatio_TCRP3*rv_depletion,
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
                          observed_decaylength=None,
                          observed_depletion=None):

    pm_model4_trained = pm.Model()
    with pm_model4_trained:

        dfRV = df_model4_trainedTable
        DP = 'Distribution parameters'

        rv_decaylength = pm.Normal('rv_decaylength', mu=100, sd=50,
                                   observed=observed_decaylength)
        rv_depletion = pm.Normal('rv_depletion', mu=100, sd=50,
                                 observed=observed_depletion)

        # RgRatio_TCRP
        """TODO: read parameters values from RV table"""
        # rv_intercept_RgRatio_TCRP3
        ID = 'rv_intercept_RgRatio_TCRP3'
        rv_intercept_RgRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_decaylangthSlope_RgRatio_TCRP3
        ID = 'rv_decaylengthSlope_RgRatio_TCRP3'
        rv_decaylengthSlope_RgRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_depletionSlope_RgRatio_TCRP3
        ID = 'rv_depletionSlope_RgRatio_TCRP3'
        rv_depletionSlope_RgRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_RgRatio_TCRP3'
        rv_output_RgRatio_TCRP3 = pm.Normal(
            ID,
            mu=rv_intercept_RgRatio_TCRP3 +
            rv_decaylengthSlope_RgRatio_TCRP3*rv_decaylength +
            rv_depletionSlope_RgRatio_TCRP3*rv_depletion,
            sd=eval(dfRV.loc[ID, DP]['sd']))

    return pm_model4_trained
#################################################
