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
                  observed_decay_length_TCRP3=None):
    ''' return a metamodel with all surrogate models '''
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
        rv_depletion_KSEG1_C = pm.Normal('rv_depletion_C',
                             mu=rv_output_depletion_KSEG1,
                             sd=50)

        # from model2:
        rv_decaylength_ALCK2_C = pm.Normal('rv_decay_length_aLCK_C', 
                                   mu = 1/rv_lambda_aLCK_LA2, 
                                   sd = 50) 
        # rv_depletion_C = pm.Normal('rv_depletion_C', 
        #                            mu = rv_dep_C, 
        #                            sd = 30)

        # model3 from coupled variables: ############################
        rv_decay_length_LCK_TP3 = pm.Normal('rv_decay_length_TP3',
                                        mu=rv_decay_length_aLCK_C,
                                        sd=50)

        rv_depletion_TP3 = pm.Normal('rv_depletion_TP3',
                                        mu=rv_dep_C,
                                        sd=30)

        # Model 3 (TCR phosphorylation) #############################
        # observed
        rTCR_max_diff_obs = rTCR_max_diff_array.reshape(-1)  # !!!!!!!!!!!

        # rv_decay_length_aLCK_TP3 = pm.Uniform('rv_decay_length_TP', 0, 300,
        #                                 observed=observed_decay_length_TP3)
        # rv_depletion_aLCK_TP3 = pm.Uniform('rv_depletion_TP', 0, 800,
        #                              observed=observed_depletion_TP3) 

        # rTCR_max_diff
        rv_noise_rTCR_max_diff_TP3 = pm.HalfNormal(
            'rv_noise_rTCR_max_diff_TP3',
            sd=mu_rv_noise_rTCR_max_diff_TP3)

        # decay_length:
        rv_decay_length_sigma_TP3 = pm.Normal(
            'rv_decay_length_sigma_TP3',
            mu=mu_rv_decay_length_sigma_TP3,
            sd=sd_rv_decay_length_sigma_TP3)

        rv_decay_length_mu_TP3 = pm.Normal(
            'rv_decay_length_mu_TP3',
            mu=mu_rv_decay_length_mu_TP3,
            sd=sd_rv_decay_length_mu_TP3)

        rv_decay_length_scale_TP3 = pm.Normal(
            'rv_decay_length_scale_TP3',
            mu=mu_rv_decay_length_scale_TP3,
            sd=sd_rv_decay_length_scale_TP3)

        rv_decay_length_gaussian_TP3 = rv_decay_length_scale_TP3 *\
            np.exp(-0.5*((rv_decay_length_LCK_TP3 - rv_decay_length_mu_TP3) /
                         rv_decay_length_sigma_TP3)**2)

        # depletion:
        rv_depletion_sigma_TP3 = pm.Normal('rv_depletion_sigma_TP3',
                                             mu=mu_rv_depletion_sigma_TP3,
                                             sd=sd_rv_depletion_sigma_TP3) # rv_dep_C
        rv_depletion_mu_TP3 = pm.Normal('rv_depletion_mu_TP3',
                                             mu=mu_rv_depletion_mu_TP3,
                                             sd=sd_rv_depletion_mu_TP3)
        rv_depletion_scale_TP3 = pm.Normal('rv_depletion_scale_TP3',
                                             mu=mu_rv_depletion_scale_TP3,
                                             sd=sd_rv_depletion_scale_TP3)

        rv_depletion_gaussian_TP3 = rv_depletion_scale_TP3 *\
            np.exp(-0.5*((rv_depletion_TP3 - rv_depletion_mu_TP3) /
                         rv_depletion_sigma_TP3)**2)

        rv_rTCR_max_diff_TP3 = pm.Normal('rv_rTCR_max_diff_TP3',
                                         mu=rv_decay_length_gaussian_TP3 *
                                         rv_depletion_gaussian_TP3,
                                         sd=rv_noise_rTCR_max_diff_TP3)

    return metamodel
