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
        sx = 100.
        sy = 100.
        s = 1.
        sd2 = 0.2
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
        # rv_p00_PhosRatio_TCRP3
        ID = 'rv_p00_PhosRatio_TCRP3'
        rv_p00_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p10_PhosRatio_TCRP3
        ID = 'rv_p10_PhosRatio_TCRP3'
        rv_p10_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p01_PhosRatio_TCRP3
        ID = 'rv_p01_PhosRatio_TCRP3'
        rv_p01_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p20_PhosRatio_TCRP3
        ID = 'rv_p20_PhosRatio_TCRP3'
        rv_p20_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p11_PhosRatio_TCRP3
        ID = 'rv_p11_PhosRatio_TCRP3'
        rv_p11_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p02_PhosRatio_TCRP3
        ID = 'rv_p02_PhosRatio_TCRP3'
        rv_p02_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd'])*s)

        ID = 'rv_output_PhosRatio_TCRP3'
        rv_output_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=rv_p00_PhosRatio_TCRP3 +\
                rv_p10_PhosRatio_TCRP3*fp_Decaylength_TCRP3/sx +\
                rv_p01_PhosRatio_TCRP3*fp_Depletion_TCRP3/sy +\
                rv_p20_PhosRatio_TCRP3*(fp_Decaylength_TCRP3/sx)**2 +\
                rv_p11_PhosRatio_TCRP3*(fp_Decaylength_TCRP3/sx)*(fp_Depletion_TCRP3/sy) +\
                rv_p02_PhosRatio_TCRP3*(fp_Depletion_TCRP3/sy)**2,
                sd=eval(dfRV.loc[ID, DP]['sd'])*s,
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
        sx = 100.
        sy = 100.
        s = 1
        sd2 = 1.
        """TODO: read parameters values from RV table"""
        # rv_p00_PhosRatio_TCRP3
        ID = 'rv_p00_PhosRatio_TCRP3'
        rv_p00_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=sd2)  # eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p10_PhosRatio_TCRP3
        ID = 'rv_p10_PhosRatio_TCRP3'
        rv_p10_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=sd2)  # eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p01_PhosRatio_TCRP3
        ID = 'rv_p01_PhosRatio_TCRP3'
        rv_p01_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=sd2)  # eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p20_PhosRatio_TCRP3
        ID = 'rv_p20_PhosRatio_TCRP3'
        rv_p20_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=sd2)  # eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p11_PhosRatio_TCRP3
        ID = 'rv_p11_PhosRatio_TCRP3'
        rv_p11_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=sd2)  # eval(dfRV.loc[ID, DP]['sd'])*s)

        # rv_p02_PhosRatio_TCRP3
        ID = 'rv_p02_PhosRatio_TCRP3'
        rv_p02_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=sd2)  # eval(dfRV.loc[ID, DP]['sd'])*s)

        ID = 'rv_output_PhosRatio_TCRP3'
        rv_output_PhosRatio_TCRP3 = pm.Normal(ID,
            mu=rv_p00_PhosRatio_TCRP3 +\
                rv_p10_PhosRatio_TCRP3*rv_Decaylength/sx +\
                rv_p01_PhosRatio_TCRP3*rv_Depletion/sy +\
                rv_p20_PhosRatio_TCRP3*(rv_Decaylength/sx)**2 +\
                rv_p11_PhosRatio_TCRP3*(rv_Decaylength/sx)*(rv_Depletion/sy) +\
                rv_p02_PhosRatio_TCRP3*(rv_Depletion/sy)**2,
                sd=eval(dfRV.loc[ID, DP]['sd'])*s)

    return pm_model3_trained
#################################################
