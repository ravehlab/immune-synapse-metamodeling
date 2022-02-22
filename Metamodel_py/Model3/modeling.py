# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pandas as pd
import pymc3 as pm
import model1
# Untrained model:
def get_pm_model1_untrained(df_trainingData_model1,
                            df_model1_untrainedTable):

    pm_model1 = pm.Model()
    with pm_model1:

        dfRV = df_model1_untrainedTable
        DP = 'Distribution parameters'

        t_KSEG1_obs = df_trainingData_model1.loc[:, 'time_sec'].values
        k_KSEG1_obs = df_trainingData_model1.loc[:, 'k0_kTnm2'].values
        dep_KSEG1_obs = df_trainingData_model1.loc[:, 'dep_nm'].values

        # rv_t
        ID = 'fp_t_dep_KSEG1'
        rv_t = pm.Uniform('rv_t',
                          0.,
                          100.,
                          observed=t_KSEG1_obs)

        # rv_k
        ID = 'fp_k_dep_KSEG1'
        rv_k = pm.Uniform('rv_k',
                          lower=dfRV.loc[ID, DP]['lower'],
                          upper=dfRV.loc[ID, DP]['upper'],
                          observed=k_KSEG1_obs)

        # dep_KSEG
        """TODO: read parameters values from RV table"""
        # rv_intercept_dep_KSEG1
        ID = 'rv_intercept_dep_KSEG1'
        rv_intercept_dep_KSEG1 = pm.Normal(ID,
                                           mu=eval(dfRV.loc[ID, DP]['mu']),
                                           sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_tSlope_dep_KSEG1
        ID = 'rv_tSlope_dep_KSEG1'
        rv_tSlope_dep_KSEG1 = pm.Normal(ID,
                                        mu=eval(dfRV.loc[ID, DP]['mu']),
                                        sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_kSlope_dep_KSEG1
        ID = 'rv_kSlope_dep_KSEG1'
        rv_kSlope_dep_KSEG1 = pm.Normal(ID,
                                        mu=eval(dfRV.loc[ID, DP]['mu']),
                                        sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_dep_KSEG1'
        rv_output_dep_KSEG1 = pm.Normal(ID,
                                        mu=rv_intercept_dep_KSEG1 +
                                        rv_tSlope_dep_KSEG1*rv_t +
                                        rv_kSlope_dep_KSEG1*rv_k,
                                        sd=eval(dfRV.loc[ID, DP]['sd']),
                                        observed=dep_KSEG1_obs)

    return pm_model1
#################################################


"""
def get_pm_model1_untrained(df_trainingData_model1,
                            df_model1_untrainedTable):

    pm_model1 = pm.Model()
    with pm_model1:

        dfRV = df_model1_untrainedTable
        DP = 'Distribution parameters'

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
                          observed=k_KSEG1_obs)

        # dep_KSEG
        # TODO: read parameters values from RV table
        # rv_intercept_dep_KSEG1
        ID = 'rv_intercept_dep_KSEG1'
        rv_intercept_dep_KSEG1 = pm.Normal(ID,
                                           mu=eval(dfRV.loc[ID, DP]['mu']),
                                           sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_tSlope_dep_KSEG1
        ID = 'rv_tSlope_dep_KSEG1'
        rv_tSlope_dep_KSEG1 = pm.Normal(ID,
                                        mu=eval(dfRV.loc[ID, DP]['mu']),
                                        sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_kSlope_dep_KSEG1
        ID = 'rv_kSlope_dep_KSEG1'
        rv_kSlope_dep_KSEG1 = pm.Normal(ID,
                                        mu=eval(dfRV.loc[ID, DP]['mu']),
                                        sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_dep_KSEG1'
        rv_output_dep_KSEG1 = pm.Normal(ID,
                                        mu=rv_intercept_dep_KSEG1 +
                                        rv_tSlope_dep_KSEG1*rv_t +
                                        rv_kSlope_dep_KSEG1*rv_k,
                                        sd=eval(dfRV.loc[ID, DP]['sd']),
                                        observed=dep_KSEG1_obs)

    return pm_model1
#################################################
"""

#################################################
# Run untrained model:
# with pm_model1:
#     trace1 = pm.sample(2000, chains=4);