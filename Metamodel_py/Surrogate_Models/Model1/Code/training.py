# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pymc3 as pm

# Untrained model:


def get_pm_model1_untrained(df_trainingData_model1,
                            df_untrainedTable):

    pm_model1_untrained = pm.Model()
    with pm_model1_untrained:

        dfRV = df_untrainedTable
        DP = 'Distribution parameters'

        x_obs = df_trainingData_model1.loc[:, 'time_sec'].values
        y_obs = df_trainingData_model1.loc[:, 'k0_kTnm2'].values
        z_obs = df_trainingData_model1.loc[:, 'depletion_nm'].values

        # rv_t
        ID = 'fp_t_KSEG1'  # 'fp_t_depletion_KSEG1'
        fp_t_KSEG1 = pm.Uniform(ID,
                                lower=dfRV.loc[ID, DP]['lower'],
                                upper=dfRV.loc[ID, DP]['upper'],
                                observed=x_obs)

        # rv_k
        ID = 'fp_k_KSEG1'
        fp_k_KSEG1 = pm.Uniform(ID,
                                lower=dfRV.loc[ID, DP]['lower'],
                                upper=dfRV.loc[ID, DP]['upper'],
                                observed=y_obs)

        # depletion_KSEG
        """TODO: read parameters values from RV table"""
        ###
        # for name, dict_ in dict_dict.items():

        # print('the name of the dictionary is ', name)
        # print('the dictionary looks like ', dict_)
        # df_untrainedTable['ID'].iloc[-2]
        for i, ID in enumerate(dfRV.ID.iloc[2:-1]):
            print(ID)
            print(dfRV.loc[ID, DP]['mu'])
            # ID_dict = dfRV.loc[ID, DP]
            # for dict_i in ID_dict:
            ID = pm.Normal(ID,
                           mu=eval(dfRV.loc[ID, DP]['mu']),
                           sd=eval(dfRV.loc[ID, DP]['sd']))
            exec(ID)
        dfRV.ID
        
        
        
        ###
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
            rv_tSlope_depletion_KSEG1*fp_t_KSEG1 +
            rv_kSlope_depletion_KSEG1*fp_k_KSEG1,
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=z_obs)

    return pm_model1_untrained
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

        rv_t_KSEG1 = pm.Normal('rv_t', mu=50, sd=20, observed=observed_t)
        rv_k_KSEG1 = pm.Normal('rv_k', mu=50, sd=20, observed=observed_k)

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
            rv_tSlope_depletion_KSEG1*rv_t_KSEG1 +
            rv_kSlope_depletion_KSEG1*rv_k_KSEG1,
            sd=eval(dfRV.loc[ID, DP]['sd']))

    return pm_model1_trained
#################################################
