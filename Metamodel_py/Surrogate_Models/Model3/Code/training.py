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
        s = 1.
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
        # rv_a_PhosRatio_TCRP3
        ID = 'rv_a_PhosRatio_TCRP3'
        rv_a_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_xScale_PhosRatio_TCRP3
        ID = 'rv_xScale_PhosRatio_TCRP3'
        rv_xScale_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_xCen_PhosRatio_TCRP3
        ID = 'rv_xCen_PhosRatio_TCRP3'
        rv_xCen_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_xDev_PhosRatio_TCRP3
        ID = 'rv_xDev_PhosRatio_TCRP3'
        rv_xDev_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_yScale_PhosRatio_TCRP3
        ID = 'rv_yScale_PhosRatio_TCRP3'
        rv_yScale_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_yCen_PhosRatio_TCRP3
        ID = 'rv_yCen_PhosRatio_TCRP3'
        rv_yCen_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_yDev_PhosRatio_TCRP3
        ID = 'rv_yDev_PhosRatio_TCRP3'
        rv_yDev_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_PhosRatio_TCRP3'
        rv_output_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=rv_a_PhosRatio_TCRP3 +\
                rv_xScale_PhosRatio_TCRP3/\
                (1 + np.exp(-(fp_Decaylength_TCRP3 - rv_xCen_PhosRatio_TCRP3)/
                rv_xDev_PhosRatio_TCRP3)) +\
                rv_yScale_PhosRatio_TCRP3/\
                (1 + np.exp(-(fp_Depletion_TCRP3 - rv_yCen_PhosRatio_TCRP3)/
                rv_yDev_PhosRatio_TCRP3)),
                sd=0.1,  # eval(dfRV.loc[ID, DP]['sd']),
                observed=z_obs)

    return pm_model3_untrained
#################################################

# Trained model:


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
        # rv_a_PhosRatio_TCRP3
        ID = 'rv_a_PhosRatio_TCRP3'
        rv_a_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_xScale_PhosRatio_TCRP3
        ID = 'rv_xScale_PhosRatio_TCRP3'
        rv_xScale_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_xCen_PhosRatio_TCRP3
        ID = 'rv_xCen_PhosRatio_TCRP3'
        rv_xCen_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_xDev_PhosRatio_TCRP3
        ID = 'rv_xDev_PhosRatio_TCRP3'
        rv_xDev_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_yScale_PhosRatio_TCRP3
        ID = 'rv_yScale_PhosRatio_TCRP3'
        rv_yScale_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_yCen_PhosRatio_TCRP3
        ID = 'rv_yCen_PhosRatio_TCRP3'
        rv_yCen_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_yDev_PhosRatio_TCRP3
        ID = 'rv_yDev_PhosRatio_TCRP3'
        rv_yDev_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_PhosRatio_TCRP3'
        rv_output_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=rv_a_PhosRatio_TCRP3 +\
                rv_xScale_PhosRatio_TCRP3/\
                (1 + np.exp(-(rv_Decaylength - rv_xCen_PhosRatio_TCRP3)/
                rv_xDev_PhosRatio_TCRP3)) +\
                rv_yScale_PhosRatio_TCRP3/\
                (1 + np.exp(-(rv_Depletion - rv_yCen_PhosRatio_TCRP3)/
                rv_yDev_PhosRatio_TCRP3)),
                sd=0.1)  # eval(dfRV.loc[ID, DP]['sd']))

    return pm_model3_trained
#################################################
