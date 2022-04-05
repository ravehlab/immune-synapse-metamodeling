#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:01:39 2022

@author: yair
"""

import numpy as np
import pandas as pd
import pymc3 as pm

Input_path = '/home/yair/Documents/Git/Metamodel_py/Coupled_Model/Input/'

# Read input data from surrogate models:
model1_df_Table_ID = pd.read_pickle(
    Input_path+"df_model1_untrainedTable_ID")
model2_df_Table_ID = pd.read_pickle(
    Input_path+"df_model2_untrainedTable_ID")
model3_df_Table_ID = pd.read_pickle(
    Input_path+"df_model3_untrainedTable_ID")
model4_df_Table_ID = pd.read_pickle(
    Input_path+"df_model4_untrainedTable_ID")

# Get random variables from tables:
DP = 'Distribution parameters'
model_Table_ID = model1_df_Table_ID
for ID in model_Table_ID.index:
    print(model_Table_ID.loc[ID, DP])

model_Table_ID = model2_df_Table_ID
for ID in model_Table_ID.index:
    print(model_Table_ID.loc[ID, DP])

model_Table_ID = model3_df_Table_ID
for ID in model_Table_ID.index:
    print(model_Table_ID.loc[ID, DP])

model_Table_ID = model4_df_Table_ID
for ID in model_Table_ID.index:
    print(model_Table_ID.loc[ID, DP])


def get_metamodel(observed_t_KSEG1=None,
                  observed_k_KSEG1=None,
                  observed_logPoff_LCKA2=None,
                  observed_logDiff_LCKA2=None,
                  observed_depletion_TCRP3=None,
                  observed_decaylength_TCRP3=None):
    ''' return a metamodel with all surrogate models '''
    DP = 'Distribution parameters'

    metamodel = pm.Model()
    with metamodel:
        # model1 - KSEG (kinetic segregation) #######################
        rv_t_KSEG1 = pm.Normal(
            'rv_t_KSEG1', mu=50, sd=20, observed=observed_t_KSEG1)
        rv_k_KSEG1 = pm.Normal(
            'rv_k_KSEG1', mu=50, sd=20, observed=observed_k_KSEG1)

        dfRV = model1_df_Table_ID
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

        # model2 - LCKA (LCK activation) ############################
        rv_Poff_LCKA2 = pm.Normal('rv_Poff', mu=-2., sd=1.,
                                  observed=observed_logPoff_LCKA2)
        rv_Diff_LCKA2 = pm.Normal('rv_Diff', mu=-2., sd=1.,
                                  observed=observed_logDiff_LCKA2)

        dfRV = model2_df_Table_ID

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
            (-0.5 * ((rv_Poff_LCKA2 - rv_PoffMu_Decaylength_LCKA2) /
                     rv_PoffSigma_Decaylength_LCKA2)**2) +
            rv_DiffScale_Decaylength_LCKA2 *
            (-0.5 * ((rv_Diff_LCKA2 - rv_DiffMu_Decaylength_LCKA2) /
                     rv_DiffSigma_Decaylength_LCKA2)**2),
            sd=0.5)  # eval(dfRV.loc[ID, DP]['sd']))

        # Coupling layer: ###########################################
        # from model1:
        rv_depletion_KSEG1_C = pm.Normal(
            'rv_depletion_KSEG1_C',
            mu=rv_output_depletion_KSEG1,
            sd=50)

        # from model2:
        rv_Decaylength_ALCK2_C = pm.Normal(
            'rv_Decaylength_ALCK2_C',
            mu=rv_output_Decaylength_LCKA2,
            sd=50)

        # model3 from coupled variables: ############################
        rv_Decaylength_LCK_TCRP3 = pm.Normal(
            'rv_Decaylength_LCK_TCRP3',
            mu=rv_Decaylength_ALCK2_C,
            sd=50)

        rv_depletion_TCRP3 = pm.Normal(
            'rv_depletion_TCRP3',
            mu=rv_depletion_KSEG1_C,
            sd=30)

        # Model 3 (TCR phosphorylation) #############################
        dfRV = model3_df_Table_ID
        DP = 'Distribution parameters'

        rv_Decaylength = pm.Normal('rv_Decaylength', mu=100, sd=50,
                                   observed=observed_decaylength_TCRP3)

        rv_Depletion = pm.Normal('rv_Depletion', mu=100, sd=50,
                                 observed=observed_depletion_TCRP3)

        # PhosRatio_TCRP
        """TODO: read parameters values from RV table"""
        # rv_DecaylengthScale_PhosRatio_TCRP3
        ID = 'rv_DecaylengthScale_PhosRatio_TCRP3'
        rv_DecaylengthScale_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthMu_PhosRatio_TCRP3
        ID = 'rv_DecaylengthMu_PhosRatio_TCRP3'
        rv_DecaylengthMu_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthSigma_PhosRatio_TCRP3
        ID = 'rv_DecaylengthSigma_PhosRatio_TCRP3'
        rv_DecaylengthSigma_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionScale_PhosRatio_TCRP3
        ID = 'rv_DepletionScale_PhosRatio_TCRP3'
        rv_DepletionScale_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionMu_PhosRatio_TCRP3
        ID = 'rv_DepletionMu_PhosRatio_TCRP3'
        rv_DepletionMu_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionSigma_PhosRatio_TCRP3
        ID = 'rv_DepletionSigma_PhosRatio_TCRP3'
        rv_DepletionSigma_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_PhosRatio_TCRP3'
        rv_output_PhosRatio_TCRP3 = pm.Normal(
            ID,
            mu=rv_DecaylengthScale_PhosRatio_TCRP3*(
                np.exp(-0.5*((rv_Decaylength -
                              rv_DecaylengthMu_PhosRatio_TCRP3) /
                             rv_DecaylengthSigma_PhosRatio_TCRP3)**2) +
                rv_DepletionScale_PhosRatio_TCRP3*(
                    np.exp(-0.5*((rv_Depletion -
                                  rv_DepletionMu_PhosRatio_TCRP3) /
                                 rv_DepletionSigma_PhosRatio_TCRP3)**2))),
            sd=eval(dfRV.loc[ID, DP]['sd']))

    return metamodel


metamodel = get_metamodel(observed_t_KSEG1=25.0,
                          observed_k_KSEG1=25.0,
                          observed_logPoff_LCKA2=-3.0,
                          observed_logDiff_LCKA2=-2.0,
                          observed_depletion_TCRP3=None,
                          observed_decaylength_TCRP3=None)

with metamodel:
    trace_metamodel = pm.sample(2000, chains=4)

# # Direction A (KSEG, LCKA to TCRP):
# pm.summary(trace_metamodel, ['rv_depletion_TCRP3',
#                              'rv_decay_length_TPCR3'])

# # Direction B (TCRP to KSEG, LCKA):
# metamodel = get_metamodel(observed_t_KSEG1=None,
#                           observed_k_KSEG1=None,
#                           observed_log_Poff_LCKA2=None,
#                           observed_log_Diff_LCKA2=None,
#                           observed_depletion_TCRP3=127,
#                           observed_decay_length_TCRP3=67)

# with metamodel:
#     trace_metamodel = pm.sample(2000, chains=4)

# vars(metamodel)
# pm.summary(trace_metamodel, ['rv_t_KSEG1',
#                              'rv_k_KSEG1',
#                              'rv_logDiff_LCKA',
#                              'rv_logPoff_LCKA'])
