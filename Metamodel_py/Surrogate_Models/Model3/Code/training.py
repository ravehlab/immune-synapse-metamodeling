# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pymc3 as pm
import numpy as np

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

        # fp_Decaylength_TCRP3
        ID = 'fp_Decaylength_TCRP3'
        fp_Decaylength_TCRP3 = pm.Normal(ID,
                                         mu=eval(dfRV.loc[ID, DP]['mu']),
                                         sd=eval(dfRV.loc[ID, DP]['sd']),
                                         observed=x_obs)

        # fp_Depletion_TCRP3
        ID = 'fp_Depletion_TCRP3'
        fp_Depletion_TCRP3 = pm.Normal(ID,
                                       mu=eval(dfRV.loc[ID, DP]['mu']),
                                       sd=eval(dfRV.loc[ID, DP]['sd']),
                                       observed=y_obs)

        # PhosRatio_TCRP
        """TODO: read parameters values from RV table"""
        # rv_DecaylengthScale_PhosRatio_TCRP3
        ID = 'rv_DecaylengthScale_PhosRatio_TCRP3'
        rv_DecaylengthScale_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthMu_PhosRatio_TCRP3
        ID = 'rv_DecaylengthMu_PhosRatio_TCRP3'
        rv_DecaylengthMu_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthSigma_PhosRatio_TCRP3
        ID = 'rv_DecaylengthSigma_PhosRatio_TCRP3'
        rv_DecaylengthSigma_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionScale_PhosRatio_TCRP3
        ID = 'rv_DepletionScale_PhosRatio_TCRP3'
        rv_DepletionScale_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionMu_PhosRatio_TCRP3
        ID = 'rv_DepletionMu_PhosRatio_TCRP3'
        rv_DepletionMu_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionSigma_PhosRatio_TCRP3
        ID = 'rv_DepletionSigma_PhosRatio_TCRP3'
        rv_DepletionSigma_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_PhosRatio_TCRP3'
        rv_output_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=rv_DecaylengthScale_PhosRatio_TCRP3*(
                np.exp(-0.5*((fp_Decaylength_TCRP3 -
                              rv_DecaylengthMu_PhosRatio_TCRP3) /
                             rv_DecaylengthSigma_PhosRatio_TCRP3)**2) +
                rv_DepletionScale_PhosRatio_TCRP3*(
                    np.exp(-0.5*((fp_Depletion_TCRP3 -
                                  rv_DepletionMu_PhosRatio_TCRP3) /
                                 rv_DepletionSigma_PhosRatio_TCRP3)**2))),
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


def get_pm_model3_trained(df_model3_trainedTable,
                          observed_Decaylength=None,
                          observed_Depletion=None):

    pm_model3_trained = pm.Model()
    with pm_model3_trained:

        dfRV = df_model3_trainedTable
        DP = 'Distribution parameters'

        rv_Decaylength = pm.Normal('rv_Decaylength', mu=100, sd=50,
                                   observed=observed_Decaylength)

        rv_Depletion = pm.Normal('rv_Depletion', mu=100, sd=50,
                                 observed=observed_Depletion)

        # PhosRatio_TCRP
        """TODO: read parameters values from RV table"""
        # rv_DecaylengthScale_PhosRatio_TCRP3
        ID = 'rv_DecaylengthScale_PhosRatio_TCRP3'
        rv_DecaylengthScale_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthMu_PhosRatio_TCRP3
        ID = 'rv_DecaylengthMu_PhosRatio_TCRP3'
        rv_DecaylengthMu_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthSigma_PhosRatio_TCRP3
        ID = 'rv_DecaylengthSigma_PhosRatio_TCRP3'
        rv_DecaylengthSigma_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionScale_PhosRatio_TCRP3
        ID = 'rv_DepletionScale_PhosRatio_TCRP3'
        rv_DepletionScale_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionMu_PhosRatio_TCRP3
        ID = 'rv_DepletionMu_PhosRatio_TCRP3'
        rv_DepletionMu_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionSigma_PhosRatio_TCRP3
        ID = 'rv_DepletionSigma_PhosRatio_TCRP3'
        rv_DepletionSigma_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_PhosRatio_TCRP3'
        rv_output_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=rv_DecaylengthScale_PhosRatio_TCRP3*(
                np.exp(-0.5*((rv_Decaylength -
                              rv_DecaylengthMu_PhosRatio_TCRP3) /
                             rv_DecaylengthSigma_PhosRatio_TCRP3)**2) +
                rv_DepletionScale_PhosRatio_TCRP3*(
                    np.exp(-0.5*((rv_Depletion -
                                  rv_DepletionMu_PhosRatio_TCRP3) /
                                 rv_DepletionSigma_PhosRatio_TCRP3)**2))),
                sd=eval(dfRV.loc[ID, DP]['sd']))

    return pm_model3_trained
#################################################
