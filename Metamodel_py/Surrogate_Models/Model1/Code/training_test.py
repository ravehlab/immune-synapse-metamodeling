#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:41:39 2022

@author: yair
"""

# -*- coding: utf-8 -*-
"""
Created on Sun "Feb  6 15:32:05 2022

@author: yairn
"""

import pymc3 as pm
import numpy as np

# Untrained model:


def get_pm_model1_untrained(df_trainingData_model1,
                            df_untrainedTable):

    pm_model1_untrained = pm.Model()
    with pm_model1_untrained:

        dfRV = df_untrainedTable
        dfRV = dfRV.set_index('ID')
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
        # for ID in dfRV.index[2:-1]:
        #     # Get the dict for distribution parameters of this ID:
        #     ID_dict = dfRV.loc[ID, DP]

        #     # Constructing the distribution parameters with 'key=value':
        #     kvs = ''  # keys, values
        #     for dict_key in ID_dict:
        #         kv = ', '+dict_key+'='+ID_dict[dict_key]
        #         kvs = kvs + kv  # print(kvs)

        #     # Constructing the pymc3 random variables for an ID as a string:
        #     pm_str = ID + ' = pm.' +\
        #         dfRV.loc[ID, 'Distribution'] +\
        #         '(' + ID + kvs + ')'

        #     right_pm = 'pm.' +\
        #         dfRV.loc[ID, 'Distribution'] +\
        #         '(' + ID + kvs + ')'

        #     print(right_pm)
        #     print(pm_str)
        #     exec(pm_str)
        ###
        # String to variable name.

        ###
        RVs = {}

        for i, ID in enumerate(dfRV.index[2:-1]):
            print(i, ID)
            # Get the dict for distribution parameters of this ID:
            ID_dict = dfRV.loc[ID, DP]

            # Constructing the distribution parameters with 'key=value':
            kvs = ''  # keys, values
            for dict_key in ID_dict:
                kv = ', '+dict_key+'='+ID_dict[dict_key]
                kvs = kvs + kv  # print(kvs)

            # Constructing the pymc3 random variables for an ID as a string:
            # pm_str = ID + ' = pm.' +\
            #     dfRV.loc[ID, 'Distribution'] +\
            #     '(' + ID + kvs + ')'

            right_pm = 'pm.' +\
                dfRV.loc[ID, 'Distribution'] +\
                '(' + ID + kvs + ')'

            RVs[ID] = right_pm
            # print(right_pm)
            # print(pm_str)

            # exec("%s = %s" % (ID, right_pm))
            # print(ID+' = ', right_pm)

        #########################################

        ID = 'rv_output_depletion_KSEG1'
        rv_output_depletion_KSEG1 = pm.Normal(
            ID,
            mu=RVs['rv_tScale_Depletion_KSEG1']/(
                1 + np.exp(-(fp_t_KSEG1 - RVs['rv_tCen_Depletion_KSEG1']) /
                           RVs['rv_tDev_Depletion_KSEG1'])) +
            RVs['rv_kScale_Depletion_KSEG1']/(
                1 + np.exp(-(fp_k_KSEG1 - RVs['rv_kCen_Depletion_KSEG1']) /
                           RVs['rv_kDev_Depletion_KSEG1'])),
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
        # rv_tScale_Depletion_KSEG1
        ID = 'rv_tScale_Depletion_KSEG1'
        rv_tScale_Depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_tCen_Depletion_KSEG1
        ID = 'rv_tCen_Depletion_KSEG1'
        rv_tCen_Depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_tDev_Depletion_KSEG1
        ID = 'rv_tDev_Depletion_KSEG1'
        rv_tDev_Depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_kScale_Depletion_KSEG1
        ID = 'rv_kScale_Depletion_KSEG1'
        rv_kScale_Depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_kCen_Depletion_KSEG1
        ID = 'rv_kCen_Depletion_KSEG1'
        rv_kCen_Depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_kDev_Depletion_KSEG1
        ID = 'rv_kDev_Depletion_KSEG1'
        rv_kDev_Depletion_KSEG1 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_depletion_KSEG1'
        rv_output_depletion_KSEG1 = pm.Normal(
            ID,
            mu=rv_tScale_Depletion_KSEG1/(
                1 + np.exp(-(rv_t_KSEG1 - rv_tCen_Depletion_KSEG1) /
                           rv_tDev_Depletion_KSEG1)) +
            rv_kScale_Depletion_KSEG1/(
                1 + np.exp(-(rv_k_KSEG1 - rv_kCen_Depletion_KSEG1) /
                           rv_kDev_Depletion_KSEG1)),
            sd=eval(dfRV.loc[ID, DP]['sd']))


    return pm_model1_trained
#################################################
