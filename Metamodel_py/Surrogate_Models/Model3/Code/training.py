# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pymc3 as pm

# Untrained model:


def get_pm_model3_untrained(df_trainingData_model3,
                            df_model3_untrainedTable):

    pm_model3_untrained = pm.Model()
    with pm_model3_untrained:

        dfRV = df_model3_untrainedTable
        DP = 'Distribution parameters'

        x_obs = df_trainingData_model3.loc[:, 'Decaylength_nm'].values
        y_obs = df_trainingData_model3.loc[:, 'Depletion_nm'].values
        z_obs = df_trainingData_model3.loc[:, 'PhosRatio'].values

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

        # depletion_KSEG
        """TODO: read parameters values from RV table"""
        # rv_intercept_PhosRatio_TCRP3
        ID = 'rv_intercept_PhosRatio_TCRP3'
        rv_intercept_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_decaylangthSlope_PhosRatio_TCRP3
        ID = 'rv_decaylengthSlope_PhosRatio_TCRP3'
        rv_decaylengthSlope_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_depletionSlope_PhosRatio_TCRP3
        ID = 'rv_depletionSlope_PhosRatio_TCRP3'
        rv_depletionSlope_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_PhosRatio_TCRP3'
        rv_output_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=rv_intercept_PhosRatio_TCRP3 +
            rv_decaylengthSlope_PhosRatio_TCRP3*rv_decaylength +
            rv_depletionSlope_PhosRatio_TCRP3*rv_depletion,
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=z_obs)

    return pm_model3_untrained
#################################################

# Trained model:


"""
Return model1 trained based on trace1.
If observed_t and/or observed_k are specified,
return the model conditioned on those values.
"""


def get_pm_model1_trained(df_model1_trainedTable,
                          observed_t=None,
                          observed_k=None):

    pm_model1_trained = pm.Model()
    with pm_model1_trained:

        dfRV = df_model1_trainedTable
        DP = 'Distribution parameters'

        rv_t = pm.Normal('rv_t', mu=50, sd=20, observed=observed_t)
        rv_k = pm.Normal('rv_k', mu=50, sd=20, observed=observed_k)

        # depletion_KSEG
        # rv_intercept_depletion_KSEG1
        ID = 'rv_intercept_depletion_KSEG1'
        rv_intercept_depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_tSlope_depletion_KSEG1
        ID = 'rv_tSlope_depletion_KSEG1'
        rv_tSlope_depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_kSlope_depletion_KSEG1
        ID = 'rv_kSlope_depletion_KSEG1'
        rv_kSlope_depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_depletion_KSEG1'
        rv_output_depletion_KSEG1 = pm.Normal(
            ID,
            mu=rv_intercept_depletion_KSEG1 +
            rv_tSlope_depletion_KSEG1*rv_t +
            rv_kSlope_depletion_KSEG1*rv_k,
            sd=eval(dfRV.loc[ID, DP]['sd']))

    return pm_model1_trained
#################################################
