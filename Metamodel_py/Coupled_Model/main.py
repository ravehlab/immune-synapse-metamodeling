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
                  observed_log_Poff_LCKA2=None,
                  observed_log_Diff_LCKA2=None,
                  observed_depletion_TCRP3=None,
                  observed_decay_length_TCRP3=None):
    ''' return a metamodel with all surrogate models '''
    metamodel = pm.Model()
    with metamodel:
        ### model1 - KS (kinetic segregation) ###########################    
        # param_t = pm.Uniform('param_t', 0, 100, observed=observed_t_KS1)
        # param_k = pm.Uniform('param_k', 0, 100, observed=observed_k_KS1)
        rv_t_KSEG1 = pm.Uniform('rv_t', 0, 100, observed=observed_t_KSEG1)
        rv_k_KSEG1 = pm.Uniform('rv_k', 0, 100, observed=observed_k_KSEG1)

        dfRV = df_untrainedTable
        # dfRV = dfRV.set_index('ID')
        DP = 'Distribution parameters'

        # depletion_KSEG
        """TODO: read parameters values from RV table"""
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

        ### model2 - LA (LCK activation) ###########################    
        # param_log_Diff = pm.Uniform('param_log_Diff', -3, 0, observed= observed_log_Diff_LA2)
        # param_log_Poff = pm.Uniform('param_log_Poff', -5, 0, observed= observed_log_Poff_LA2)
        rv_log_Diff = pm.Uniform('rv_log_Diff', -3, 0, observed= observed_log_Diff_LA2)
        rv_log_Poff = pm.Uniform('rv_log_Poff', -5, 0, observed= observed_log_Poff_LA2)

        ### lambda aLCK
        noise_lambda_aLCK_LA2 = pm.HalfNormal('noise_lambda_aLCK_LA2',
                                             sd = mu_noise_lambda_aLCK_LA2) # noise 
        rv_lambda_aLCK_min_LA2 = pm.TruncatedNormal('rv_lambda_aLCK_min_LA2',
                                                   mu = mu_rv_lambda_aLCK_min_LA2,
                                                   sd = sd_rv_lambda_aLCK_min_LA2,
                                                   upper = 0.01)
        rv_lambda_aLCK_max_LA2 = pm.TruncatedNormal('rv_lambda_aLCK_max_LA2',
                                                   mu = mu_rv_lambda_aLCK_max_LA2,
                                                   sd = sd_rv_lambda_aLCK_max_LA2,
                                                   lower = 0.01)
        rv_lambda_aLCK_log_Diff_center_LA2 = pm.Normal('rv_lambda_aLCK_log_Diff_center_LA2',
                                                      mu = mu_rv_lambda_aLCK_log_Diff_center_LA2, # 0
                                                      sd = sd_rv_lambda_aLCK_log_Diff_center_LA2) # 1
        rv_lambda_aLCK_log_Diff_divisor_LA2 = pm.TruncatedNormal('rv_lambda_aLCK_log_Diff_divisor_LA2',
                                                                mu = mu_rv_lambda_aLCK_log_Diff_divisor_LA2, 
                                                                sd = sd_rv_lambda_aLCK_log_Diff_divisor_LA2,
                                                                upper = 0)
        rv_lambda_aLCK_log_Poff_center_LA2 = pm.Normal('rv_lambda_aLCK_log_Poff_center_LA2',
                                                       mu = mu_rv_lambda_aLCK_log_Poff_center_LA2, # -3
                                                       sd = sd_rv_lambda_aLCK_log_Poff_center_LA2) # 0.5
        rv_lambda_aLCK_log_Poff_divisor_LA2 = pm.TruncatedNormal('rv_lambda_aLCK_log_Poff_divisor_LA2',
                                                                mu = mu_rv_lambda_aLCK_log_Poff_divisor_LA2, #
                                                                sd = sd_rv_lambda_aLCK_log_Poff_divisor_LA2,
                                                                lower = 0) # 

        rv_tmp_x1 = (rv_log_Diff - rv_lambda_aLCK_log_Diff_center_LA2) / rv_lambda_aLCK_log_Diff_divisor_LA2
        rv_tmp_sig1 = 1.0 / (1 + np.exp(-rv_tmp_x1))
        rv_tmp_x2 = (rv_log_Poff - rv_lambda_aLCK_log_Poff_center_LA2) / rv_lambda_aLCK_log_Poff_divisor_LA2
        rv_tmp_sig2 = 1.0 / (1 + np.exp(-rv_tmp_x2))

        rv_lambda_aLCK_LA2 = pm.Normal('rv_lambda_aLCK_LA', mu=rv_lambda_aLCK_min_LA2 +\
                                  (rv_lambda_aLCK_max_LA2 - rv_lambda_aLCK_min_LA2)*\
                                  rv_tmp_sig1 * rv_tmp_sig2,
                                  sd=noise_lambda_aLCK_LA2) # , observed=lambda_aLCK_obs

        ### Coupling model ########################################
        ### Coupling model ########################### 
        """ example from previous coupled model:
        rv_dw_aLCK_C = pm.Normal('rv_dw_aLCK_C', 
                                   mu= rv_dw_aLCK_LA3, 
                                   sd=2) 
        rv_dm_CD45_C = pm.Normal('rv_dm_CD45_C', 
                                   mu= rv_dm_CD45_KS1, 
                                   sd= 3) 
        ### Model 2 (TCR phosphorylation) ################    
        # random variables x and y
        rv_dw_aLCK_TP2 = pm.Normal('rv_dw_aLCK_TP2', 
                                   mu= rv_dw_aLCK_C, 
                                   sd=2) 
        rv_dm_CD45_TP2 = pm.Normal('rv_dm_CD45_TP2', 
                                   mu= rv_dm_CD45_C, 
                                   sd=3) 
        """
        ###########################################################
        # from model1:
        """
        rv_w_TCR_C = pm.Normal('rv_w_TCR_C', 
                                   mu = rv_w_TCR_KS1, 
                                   sd = 20)         
        rv_w_CD45_C = pm.Normal('rv_w_CD45_C', 
                                   mu = rv_w_CD45_KS1, 
                                   sd = 20)
        """          
        rv_dep_C = pm.Normal('rv_dep_C', 
                                   mu = rv_dep_KS1, 
                                   sd = 50) 
        
        # from model2:
        # rv_lambda_aLCK_C = pm.Normal('rv_lambda_aLCK_C', 
        #                            mu = rv_lambda_aLCK_LA2, 
        #                            sd = 2)

        rv_decay_length_aLCK_C = pm.Normal('rv_decay_length_aLCK_C', 
                                   mu = 1/rv_lambda_aLCK_LA2, 
                                   sd = 50) 
        # rv_depletion_C = pm.Normal('rv_depletion_C', 
        #                            mu = rv_dep_C, 
        #                            sd = 30)
        
        ### model3 from coupled variables: ###############
        rv_decay_length_LCK_TP3 = pm.Normal('rv_decay_length_TP3',
                                        mu=rv_decay_length_aLCK_C,
                                        sd=50)

        rv_depletion_TP3 = pm.Normal('rv_depletion_TP3',
                                        mu=rv_dep_C,
                                        sd=30)

        ### Model 3 (TCR phosphorylation) ################
        # observed 
        rTCR_max_diff_obs = rTCR_max_diff_array.reshape(-1) #!!!!!!!!!!!
        ### !!!

        # rv_decay_length_aLCK_TP3 = pm.Uniform('rv_decay_length_TP', 0, 300,
        #                                 observed=observed_decay_length_TP3)
        # rv_depletion_aLCK_TP3 = pm.Uniform('rv_depletion_TP', 0, 800,
        #                              observed=observed_depletion_TP3) 


        ### rTCR_max_diff
        rv_noise_rTCR_max_diff_TP3 = pm.HalfNormal('rv_noise_rTCR_max_diff_TP3',
                                              sd=mu_rv_noise_rTCR_max_diff_TP3) # noise 
        
        ### decay_length:
        rv_decay_length_sigma_TP3 = pm.Normal('rv_decay_length_sigma_TP3',
                                             mu=mu_rv_decay_length_sigma_TP3,
                                             sd=sd_rv_decay_length_sigma_TP3)
        rv_decay_length_mu_TP3 = pm.Normal('rv_decay_length_mu_TP3',
                                             mu=mu_rv_decay_length_mu_TP3,
                                             sd=sd_rv_decay_length_mu_TP3)
        rv_decay_length_scale_TP3 = pm.Normal('rv_decay_length_scale_TP3',
                                             mu=mu_rv_decay_length_scale_TP3,
                                             sd=sd_rv_decay_length_scale_TP3)
        
        rv_decay_length_gaussian_TP3 = rv_decay_length_scale_TP3*\
            np.exp(-0.5*((rv_decay_length_LCK_TP3 - rv_decay_length_mu_TP3)/rv_decay_length_sigma_TP3)**2)
        
        ### depletion:
        rv_depletion_sigma_TP3 = pm.Normal('rv_depletion_sigma_TP3',
                                             mu=mu_rv_depletion_sigma_TP3,
                                             sd=sd_rv_depletion_sigma_TP3) # rv_dep_C
        rv_depletion_mu_TP3 = pm.Normal('rv_depletion_mu_TP3',
                                             mu=mu_rv_depletion_mu_TP3,
                                             sd=sd_rv_depletion_mu_TP3)
        rv_depletion_scale_TP3 = pm.Normal('rv_depletion_scale_TP3',
                                             mu=mu_rv_depletion_scale_TP3,
                                             sd=sd_rv_depletion_scale_TP3)
        
        rv_depletion_gaussian_TP3 = rv_depletion_scale_TP3*\
            np.exp(-0.5*((rv_depletion_TP3 - rv_depletion_mu_TP3)/rv_depletion_sigma_TP3)**2)       
        
        rv_rTCR_max_diff_TP3 = pm.Normal('rv_rTCR_max_diff_TP3', mu=rv_decay_length_gaussian_TP3*\
                                    rv_depletion_gaussian_TP3,                     
                                    sd=rv_noise_rTCR_max_diff_TP3) # observed=rTCR_max_diff_obs

    return metamodel


