# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pymc3 as pm

# Untrained model:


def get_pm_model2_untrained(df_trainingData_model,
                            df_model_untrainedTable):

    pm_model_untrained = pm.Model()
    with pm_model_untrained:

        dfRV = df_model_untrainedTable
        DP = 'Distribution parameters'

        x_obs = df_trainingData_model.loc[:, 'Poff'].values
        y_obs = df_trainingData_model.loc[:, 'Diff'].values
        z_obs = df_trainingData_model.loc[:, 'Decaylength_nm'].values

        # fp_Poff_LCKA2
        ID = 'fp_Poff_LCKA2'
        fp_Poff_LCKA2 = pm.Normal(
            'rv_Poff',
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=x_obs)

        # fp_Diff_LCKA2
        ID = 'fp_Diff_LCKA2'
        fp_Diff_LCKA2 = pm.Normal(
            'fp_Diff_LCKA2',
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=y_obs)

        # decaylength_LCKA2
        """TODO: read parameters values from RV table"""
        # rv_PoffScale_Decaylength_LCKA2
        ID = 'rv_PoffScale_Decaylength_LCKA2'
        rv_PoffScale_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_PoffMu_Decaylength_LCKA2
        ID = 'rv_PoffMu_Decaylength_LCKA2'
        rv_PoffMu_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_PoffSigma_Decaylength_LCKA2
        ID = 'rv_PoffSigma_Decaylength_LCKA2'
        rv_PoffSigma_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DiffScale_decaylength_LCKA2
        ID = 'rv_DiffScale_Decaylength_LCKA2'
        rv_DiffScale_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DiffMu_Decaylength_LCKA2
        ID = 'rv_DiffMu_Decaylength_LCKA2'
        rv_DiffMu_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DiffSigma_Decaylength_LCKA2
        ID = 'rv_DiffSigma_Decaylength_LCKA2'
        rv_DiffSigma_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_Decaylength_LCKA2'
        rv_output_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=rv_PoffScale_Decaylength_LCKA2 *
            (-0.5 * ((fp_Poff_LCKA2 - rv_PoffMu_Decaylength_LCKA2) /
                     rv_PoffSigma_Decaylength_LCKA2)**2) +
            rv_DiffScale_Decaylength_LCKA2 *
            (-0.5 * ((fp_Diff_LCKA2 - rv_DiffMu_Decaylength_LCKA2) /
                     rv_DiffSigma_Decaylength_LCKA2)**2),
            sd=0.5,  # eval(dfRV.loc[ID, DP]['sd']),
            observed=z_obs)

    return pm_model_untrained
#################################################

# Trained model:


"""
Return trained model based on trace.
If observed_x and/or observed_y are specified,
return the model conditioned on those values.
"""


def get_pm_model2_trained(df_model_trainedTable,
                          observed_Poff=None,
                          observed_Diff=None):

    pm_model_trained = pm.Model()
    with pm_model_trained:

        dfRV = df_model_trainedTable
        DP = 'Distribution parameters'

        rv_Poff = pm.Normal('rv_Poff', mu=-2., sd=1., observed=observed_Poff)
        rv_Diff = pm.Normal('rv_Diff', mu=-2., sd=1., observed=observed_Diff)

        # decaylength_LCKA2
        """TODO: read parameters values from RV table"""
        # rv_PoffScale_Decaylength_LCKA2
        ID = 'rv_PoffScale_Decaylength_LCKA2'
        rv_PoffScale_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_PoffMu_Decaylength_LCKA2
        ID = 'rv_PoffMu_Decaylength_LCKA2'
        rv_PoffMu_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_PoffSigma_Decaylength_LCKA2
        ID = 'rv_PoffSigma_Decaylength_LCKA2'
        rv_PoffSigma_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DiffScale_Decaylength_LCKA2
        ID = 'rv_DiffScale_Decaylength_LCKA2'
        rv_DiffScale_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DiffMu_Decaylength_LCKA2
        ID = 'rv_DiffMu_Decaylength_LCKA2'
        rv_DiffMu_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DiffSigma_Decaylength_LCKA2
        ID = 'rv_DiffSigma_Decaylength_LCKA2'
        rv_DiffSigma_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_Decaylength_LCKA2'
        rv_output_Decaylength_LCKA2 = pm.Normal(
            ID,
            mu=rv_PoffScale_Decaylength_LCKA2 *
            (-0.5 * ((rv_Poff - rv_PoffMu_Decaylength_LCKA2) /
                     rv_PoffSigma_Decaylength_LCKA2)**2) +
            rv_DiffScale_Decaylength_LCKA2 *
            (-0.5 * ((rv_Diff - rv_DiffMu_Decaylength_LCKA2) /
                     rv_DiffSigma_Decaylength_LCKA2)**2),
            sd=0.5)  # eval(dfRV.loc[ID, DP]['sd']))

    return pm_model_trained
#################################################
