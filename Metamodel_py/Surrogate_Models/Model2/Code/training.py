# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pymc3 as pm

# Untrained model:


def get_pm_model_untrained(df_trainingData_model,
                           df_model_untrainedTable):

    pm_model_untrained = pm.Model()
    with pm_model_untrained:

        dfRV = df_model_untrainedTable
        DP = 'Distribution parameters'

        x_obs = df_trainingData_model.loc[:, 'Poff'].values
        y_obs = df_trainingData_model.loc[:, 'Diff'].values
        z_obs = df_trainingData_model.loc[:, 'Decaylength_nm'].values

        # rv_Poff
        ID = 'fp_Poff_decaylength_LCKA2'
        rv_Poff = pm.Normal(
            'rv_Poff',
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=x_obs)

        # rv_Diff
        ID = 'fp_Diff_decaylength_LCKA2'
        rv_Diff = pm.Normal(
            'rv_Diff',
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=y_obs)

        # decaylength_LCKA2
        """TODO: read parameters values from RV table"""
        # rv_intercept_decaylength_LCKA2
        ID = 'rv_intercept_decaylength_LCKA2'
        rv_intercept_decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_PoffSlope_decaylength_LCKA2
        ID = 'rv_PoffSlope_decaylength_LCKA2'
        rv_PoffSlope_decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DiffSlope_decaylength_LCKA2
        ID = 'rv_DiffSlope_decaylength_LCKA2'
        rv_DiffSlope_decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_decaylength_LCKA2'
        rv_output_decaylength_LCKA2 = pm.Normal(
            ID,
            mu=rv_intercept_decaylength_LCKA2 +
            rv_PoffSlope_decaylength_LCKA2*rv_Poff +
            rv_DiffSlope_decaylength_LCKA2*rv_Diff,
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=z_obs)

    return pm_model_untrained
#################################################

# Trained model:


"""
Return model1 trained based on trace1.
If observed_t and/or observed_k are specified,
return the model conditioned on those values.
"""


def get_pm_model_trained(df_model_trainedTable,
                         observed_Poff=None,
                         observed_Diff=None):

    pm_model_trained = pm.Model()
    with pm_model_trained:

        dfRV = df_model_trainedTable
        DP = 'Distribution parameters'

        rv_Poff = pm.Normal('rv_t', mu=-2., sd=1., observed=observed_Poff)
        rv_Diff = pm.Normal('rv_k', mu=-2., sd=1., observed=observed_Diff)

        # depletion_KSEG
        # rv_intercept_depletion_KSEG1
        # rv_intercept_decaylength_LCKA2
        ID = 'rv_intercept_decaylength_LCKA2'
        rv_intercept_decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_PoffSlope_decaylength_LCKA2
        ID = 'rv_PoffSlope_decaylength_LCKA2'
        rv_PoffSlope_decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DiffSlope_decaylength_LCKA2
        ID = 'rv_DiffSlope_decaylength_LCKA2'
        rv_DiffSlope_decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_decaylength_LCKA2'
        rv_output_decaylength_LCKA2 = pm.Normal(
            ID,
            mu=rv_intercept_decaylength_LCKA2 +
            rv_PoffSlope_decaylength_LCKA2*rv_Poff +
            rv_DiffSlope_decaylength_LCKA2*rv_Diff,
            sd=1.)

    return pm_model_trained
#################################################
