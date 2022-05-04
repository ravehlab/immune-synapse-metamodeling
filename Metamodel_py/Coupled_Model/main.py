#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 08:01:39 2022

@author: yair
"""

import numpy as np
import pandas as pd
import pymc3 as pm
# import seaborn as sns

# Input_path = '/home/yair/Documents/Git/Metamodel_py/Coupled_Model/Input/'
# Input_path = '/home/jonah/Yair/Git/Metamodel_py/Coupled_Model/Input/'
# Input_path = '/immune-synapse-metamodeling/Metamodel_py/Coupled_Model/Input/'
Input_path = '/cs/usr/yairneveoz/immune-synapse-metamodeling/Metamodel_py/Coupled_Model/Input/'
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
if False:
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

# %% Coupled model:
def get_metamodel(observed_t_KSEG1=None,
                  observed_k_KSEG1=None,
                  observed_logPoff_LCKA2=None,
                  observed_logDiff_LCKA2=None,
                  observed_phosRatio_TCRP3=None,
                  observed_RgRatio_TCRP4=None):
    ''' return a metamodel with all surrogate models '''
    DP = 'Distribution parameters'

    metamodel = pm.Model()
    with metamodel:
        # % model1 - KSEG (kinetic segregation) ####################
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

        # % model2 - LCKA (LCK activation) #########################
        rv_logPoff_LCKA2 = pm.Normal('rv_logPoff_LCKA2', mu=-2., sd=1.,
                                  observed=observed_logPoff_LCKA2)
        rv_logDiff_LCKA2 = pm.Normal('rv_logDiff_LCKA2', mu=-2., sd=1.,
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
            (-0.5 * ((rv_logPoff_LCKA2 - rv_PoffMu_Decaylength_LCKA2) /
                     rv_PoffSigma_Decaylength_LCKA2)**2) +
            rv_DiffScale_Decaylength_LCKA2 *
            (-0.5 * ((rv_logDiff_LCKA2 - rv_DiffMu_Decaylength_LCKA2) /
                     rv_DiffSigma_Decaylength_LCKA2)**2),
            sd=0.5)  # eval(dfRV.loc[ID, DP]['sd']))

        # % Coupling layer: ########################################
        # from model1:
        rv_depletion_KSEG1_C = pm.Normal(
            'rv_depletion_KSEG1_C',
            mu=rv_output_depletion_KSEG1,
            sd=50)

        # from model2:
        rv_Decaylength_ALCK2_C = pm.Normal(
            'rv_Decaylength_ALCK2_C',
            mu=10**rv_output_Decaylength_LCKA2,
            sd=50)
        
        # from model2:
        # rv_Decaylength_ALCK2_C = pm.Normal(
        #     'rv_Decaylength_ALCK2_C',
        #     mu=rv_output_Decaylength_LCKA2,
        #     sd=50)

        # model3 from coupled variables: ############################
        rv_Decaylength_TCRP3 = pm.Normal(
            'rv_Decaylength_TCRP3',
            mu=rv_Decaylength_ALCK2_C,
            sd=50)

        rv_depletion_TCRP3 = pm.Normal(
            'rv_depletion_TCRP3',
            mu=rv_depletion_KSEG1_C,
            sd=30)

        #############################################################
        # % Model 3 (TCR phosphorylation, PhosRatio) ###############
        dfRV = model3_df_Table_ID
        DP = 'Distribution parameters'

        # rv_Decaylength = pm.Normal('rv_Decaylength', mu=100, sd=50,
        #                            observed=observed_decaylength_TCRP3)

        # rv_Depletion = pm.Normal('rv_Depletion', mu=100, sd=50,
        #                          observed=observed_depletion_TCRP3)

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
                np.exp(-0.5*((rv_Decaylength_TCRP3 -
                              rv_DecaylengthMu_PhosRatio_TCRP3) /
                             rv_DecaylengthSigma_PhosRatio_TCRP3)**2) +
                rv_DepletionScale_PhosRatio_TCRP3*(
                    np.exp(-0.5*((rv_depletion_TCRP3 -
                                  rv_DepletionMu_PhosRatio_TCRP3) /
                                 rv_DepletionSigma_PhosRatio_TCRP3)**2))),
            sd=eval(dfRV.loc[ID, DP]['sd']),
            observed=observed_phosRatio_TCRP3)

        #############################################################
        # % Model 4 (TCR phosphorylation, RgRatio) #################
        dfRV = model4_df_Table_ID

        # RgRatio_TCRP
        """TODO: read parameters values from RV table"""
        # rv_Intercept_RgRatio_TCRP4
        ID = 'rv_Intercept_RgRatio_TCRP4'
        rv_Intercept_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=0.2)  # eval(dfRV.loc[ID, DP]['sd']))

        # rv_DecaylengthSlope_RgRatio_TCRP4
        ID = 'rv_DecaylengthSlope_RgRatio_TCRP4'
        rv_DecaylengthSlope_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=0.2)  # eval(dfRV.loc[ID, DP]['sd']))

        # rv_DepletionSlope_RgRatio_TCRP4
        ID = 'rv_DepletionSlope_RgRatio_TCRP4'
        rv_DepletionSlope_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=eval(dfRV.loc[ID, DP]['mu']),
            sd=0.2)  # eval(dfRV.loc[ID, DP]['sd']))

        ID = 'rv_output_RgRatio_TCRP4'
        rv_output_RgRatio_TCRP4 = pm.Normal(
            ID,
            mu=rv_Intercept_RgRatio_TCRP4 +
            rv_DecaylengthSlope_RgRatio_TCRP4*rv_Decaylength_TCRP3 +
            rv_DepletionSlope_RgRatio_TCRP4*rv_depletion_TCRP3,
            sd=0.3, observed=observed_RgRatio_TCRP4)  # eval(dfRV.loc[ID, DP]['sd']))

    return metamodel


# %% Direction A (KSEG & LCKA to TCRP):
# Get coupled_model:
# x_obs = df_trainingData_model4.loc[:, 'Decaylength_nm'].values
# y_obs = df_trainingData_model4.loc[:, 'Depletion_nm'].values
# z_obs = df_trainingData_model4.loc[:, 'RgRatio'].values
# t_KSEG1_obs = np.linspace(10., 100., 10)  # 25.
# k_KSEG1_obs = np.linspace(10., 100., 10)  # 25.

    
metamodel = get_metamodel(observed_t_KSEG1=50.,
                          observed_k_KSEG1=25.,
                          observed_logPoff_LCKA2=-1.0,  # -1.0
                          observed_logDiff_LCKA2=-2.0,  # -2.0
                          observed_phosRatio_TCRP3=None,
                          observed_RgRatio_TCRP4=None)

with metamodel:
    trace_metamodel = pm.sample(2000, chains=4)

# pm.traceplot(trace_metamodel)
pm.plot_trace(trace_metamodel)

pm.summary(trace_metamodel, ['rv_depletion_TCRP3',
                             'rv_Decaylength_TCRP3'])

# %% Direction A, plot Kernel Density Estimation (KDE) joint plot:
df_trace_metamodel = pm.trace_to_dataframe(trace_metamodel)
# df_trace_metamodel
"""
m3 = sns.jointplot(df_trace_metamodel.loc[:,'rv_Decaylength_TCRP3'],
              df_trace_metamodel.loc[:,'rv_depletion_TCRP3'],
              kind='kde', xlim=(0, 200), ylim=(0, 200), n_levels=10,
              shade=True, cmap='Oranges', shade_lowest=True)
m3.ax_marg_x.lines[0].set_color('#fdae6b')
m3.ax_marg_y.lines[0].set_color('#fdae6b')
m3.ax_marg_x.collections[0].set_facecolor('#fdae6b')
m3.ax_marg_y.collections[0].set_facecolor('#fdae6b')
"""
# %% Direction B (TCRP to KSEG & LCKA):
metamodel = get_metamodel(observed_t_KSEG1=None,
                          observed_k_KSEG1=None,
                          observed_logPoff_LCKA2=None,
                          observed_logDiff_LCKA2=None,
                          observed_phosRatio_TCRP3=60.,
                          observed_RgRatio_TCRP4=67.)

with metamodel:
    trace_metamodel = pm.sample(2000, chains=4)

pm.plot_trace(trace_metamodel)
vars(metamodel)
pm.summary(trace_metamodel, ['rv_t_KSEG1',
                             'rv_k_KSEG1',
                             'rv_logPoff_LCKA2',
                             'rv_logDiff_LCKA2'])

# %% Direction B, plot Kernel Density Estimation (KDE) joint plot:
df_trace_metamodel = pm.trace_to_dataframe(trace_metamodel)
df_trace_metamodel
"""
m1 = sns.jointplot(df_trace_metamodel.loc[:,'rv_t_KSEG1'],
                   df_trace_metamodel.loc[:,'rv_k_KSEG1'],
                   kind='kde', xlim=(0, 100), ylim=(0, 100), n_levels=10,
                   shade=True, cmap='Purples', shade_lowest=True)
m1.ax_marg_x.lines[0].set_color('#756bb1')
m1.ax_marg_y.lines[0].set_color('#756bb1')
m1.ax_marg_x.collections[0].set_facecolor('#756bb1')
m1.ax_marg_y.collections[0].set_facecolor('#756bb1')
"""
# %% Direction B, plot Kernel Density Estimation (KDE) joint plot:
df_trace_metamodel = pm.trace_to_dataframe(trace_metamodel)
"""
m2 = sns.jointplot(df_trace_metamodel.loc[:,'rv_logPoff_LCKA2'],
              df_trace_metamodel.loc[:,'rv_logDiff_LCKA2'],
              kind='kde', xlim=(-5, 0), ylim=(-3, 0), n_levels=100,
              shade=True, cmap='Blues', shade_lowest=True)
m2.ax_marg_x.lines[0].set_color('#2b7bba')
m2.ax_marg_y.lines[0].set_color('#2b7bba')
m2.ax_marg_x.collections[0].set_facecolor('#2b7bba')
m2.ax_marg_y.collections[0].set_facecolor('#2b7bba')
# %% TODO:
"""
"""
1. Make transparent/white maps with values of the KDE jointplots and 
plot them over the surrogate models. (Matlab)
2. Get values from Razvag's paper and run the coupled model.
3. Find how to include NaN values (Model4) in order eliminate 'impossible'
values.
[X1, Y1], [X1, X2], [X1, Y2], [X2, Y1], [X2, Y2], [Y1, Y2] 
"""
# %% Direction A: Model1 and Model2 to Model3: ######################
# Run loop:
# Define pairs of free parameters:
batch = 't_k'
# batch = 't_Poff'
# batch = 't_Diff'
# batch = 'k_Poff'
# batch = 'k_Diff'
# batch = 'Poff_Diff'

# Define fixed parameters values:
fixed_t = 50.
fixed_k = 25.
fixed_logPoff = -1.
fixed_logDiff = -2.

Np = 4  # 20  # Square grid size:

# Define batch parameters values:    
batch_t = np.linspace(100./(Np), 100., Np)
batch_k = np.linspace(100./(Np), 100., Np)
batch_logPoff = np.linspace(-5., -5./Np, Np)
batch_logDiff = np.linspace(-3., -3./Np, Np)

if True: 
    if batch == 't_k':
        batch_x = batch_t
        batch_y = batch_k
        
        fixed_x = fixed_logPoff
        fixed_y = fixed_logDiff

    if batch == 't_Poff':
        batch_x = batch_t
        batch_y = batch_logPoff
        
        fixed_x = fixed_k
        fixed_y = fixed_logDiff
        
    if batch == 't_Diff':
        batch_x = batch_t
        batch_y = batch_logDiff
        
        fixed_x = fixed_k
        fixed_y = fixed_logPoff
        
    if batch == 'k_Poff':
        batch_x = batch_logPoff
        batch_y = batch_k
        
        fixed_x = fixed_t
        fixed_y = fixed_logDiff
        
    if batch == 'k_Diff':
        batch_x = batch_logDiff
        batch_y = batch_k
        
        fixed_x = fixed_t
        fixed_y = fixed_logPoff     
        
    if batch == 'Poff_Diff':
        batch_x = batch_logPoff
        batch_y = batch_logDiff
        
        fixed_x = fixed_t
        fixed_y = fixed_k    
    
    RgRatios_mean = np.zeros((Np, Np))
    RgRatios_std = np.zeros((Np, Np))

    phosRatios_mean = np.zeros((Np, Np))
    phosRatios_std = np.zeros((Np, Np))
    
    # rv_output_dep_KSEG1_mean = np.zeros((Np, Np))
    # rv_dep_C_mean = np.zeros((Np, Np))
    # rv_depletion_TCRP_mean = np.zeros((Np, Np))
    
    # rv_output_logDLALCK_ALCK2_mean = np.zeros((Np, Np))
    # rv_decayLength_C_mean = np.zeros((Np, Np))
    # rv_decayLength_TCRP_mean = np.zeros((Np, Np))

    for i,y_in in enumerate(batch_y):
        for j,x_in in enumerate(batch_x):
#             try:
            if batch == 't_k':
                cur_metamodel = get_metamodel(
                    observed_t_KSEG1 = x_in,
                    observed_k_KSEG1 = y_in,
                    observed_logPoff_LCKA2 = fixed_x, 
                    observed_logDiff_LCKA2 = fixed_y,
                    observed_RgRatio_TCRP4 = None,
                    observed_phosRatio_TCRP3 = None)

            if batch == 't_Poff':
                cur_metamodel = get_metamodel(
                    observed_t_KSEG1 = x_in,
                    observed_k_KSEG1 = fixed_y,
                    observed_logPoff_LCKA2 = y_in, 
                    observed_logDiff_LCKA2 = fixed_x,
                    observed_RgRatio_TCRP4 = None,
                    observed_phosRatio_TCRP3 = None)

            if batch == 't_Diff':
                cur_metamodel = get_metamodel(
                    observed_t_KSEG1 = x_in,
                    observed_k_KSEG1 = fixed_y,
                    observed_logPoff_LCKA2 = fixed_x, 
                    observed_logDiff_LCKA2 = y_in,
                    observed_RgRatio_TCRP4 = None,
                    observed_phosRatio_TCRP3 = None)

            if batch == 'k_Poff':
                cur_metamodel = get_metamodel(
                    observed_t_KSEG1 = fixed_x,
                    observed_k_KSEG1 = y_in,
                    observed_logPoff_LCKA2 = x_in, 
                    observed_logDiff_LCKA2 = fixed_y,
                    observed_RgRatio_TCRP4 = None,
                    observed_phosRatio_TCRP3 = None)

            if batch == 'k_Diff':
                cur_metamodel = get_metamodel(
                    observed_t_KSEG1 = fixed_x,
                    observed_k_KSEG1 = y_in,
                    observed_logPoff_LCKA2 = fixed_y, 
                    observed_logDiff_LCKA2 = x_in,
                    observed_RgRatio_TCRP4 = None,
                    observed_phosRatio_TCRP3 = None)

            if batch == 'Poff_Diff':
                cur_metamodel = get_metamodel(
                    observed_t_KSEG1 = fixed_x,
                    observed_k_KSEG1 = fixed_y,
                    observed_logPoff_LCKA2 = x_in,
                    observed_logDiff_LCKA2 = y_in,
                    observed_RgRatio_TCRP4 = None,
                    observed_phosRatio_TCRP3 = None)

            try:
                print(f"i,y_val={i,y_in}, j,x_val={j,x_in}")
                with cur_metamodel:                    
                    cur_trace_metamodel = pm.sample(2000, chains=4, progressbar = False)                    
    
                    RgRatios_mean[i,j] = cur_trace_metamodel.rv_output_RgRatio_TCRP4.mean()
                    RgRatios_std[i,j] = cur_trace_metamodel.rv_output_RgRatio_TCRP4.std()
        
                    phosRatios_mean[i,j] = cur_trace_metamodel.rv_output_PhosRatio_TCRP3.mean() 
                    phosRatios_std[i,j] = cur_trace_metamodel.rv_output_PhosRatio_TCRP3.std()
            
            except:
                print("An exception occurred")

# %%
"""
if True:
    n_ys = 20 #4 #
    n_xs = 20 #4 #
    
    if batch == 't_k':
        x1s = np.linspace(100./(n_xs), 100., n_xs)
        y1s = np.linspace(100./(n_ys), 100., n_ys) # 25. 

        x2s = -4. #
        y2s = -2. #
        
        batch_x = x1s
        batch_y = y1s
        
        fixed_x = x2s
        fixed_y = y2s
        
    if batch == 'Poff_Diff':
        x1s = 50.
        y1s = 25. 

        x2s = np.linspace(-5., -5./n_xs, n_xs)
        y2s = np.linspace(-3., -3./n_xs, n_xs)
        
        fixed_x = x1s
        fixed_y = y1s
        
        batch_x = x2s
        batch_y = y2s        
        
        
    if batch == 'Poff_k':
        x1s = 50.
        y1s = np.linspace(100./(n_ys), 100., n_ys) 

        x2s = np.linspace(-5., -5./n_xs, n_xs) 
        y2s = -2.
    
        fixed_x = x1s
        batch_y = y1s
        
        batch_x = x2s
        fixed_y = y2s        
    
    
    
    RgRatios_mean = np.zeros((n_ys, n_xs))
    RgRatios_std = np.zeros((n_ys, n_xs))

    phosRatios_mean = np.zeros((n_ys, n_xs))
    phosRatios_std = np.zeros((n_ys, n_xs))
    
    rv_output_dep_KSEG1_mean = np.zeros((n_ys, n_xs))
    rv_dep_C_mean = np.zeros((n_ys, n_xs))
    rv_depletion_TCRP_mean = np.zeros((n_ys, n_xs))
    
    rv_output_logDLALCK_ALCK2_mean = np.zeros((n_ys, n_xs))
    rv_decayLength_C_mean = np.zeros((n_ys, n_xs))
    rv_decayLength_TCRP_mean = np.zeros((n_ys, n_xs))
    
    for i,y_in in enumerate(batch_y):
        for j,x_in in enumerate(batch_x):
#             try:
            if batch == 't_k':
                cur_metamodel = get_pm_metamodel(observed_t_KSEG1 = x_in,
                                                 observed_k_KSEG1 = y_in,
                                                 observed_logPoff_LCKA2 = fixed_x, 
                                                 observed_logDiff_LCKA2 = fixed_y,
                                                 observed_RgRatio_TCRP3 = None,
                                                 observed_phosRatio_TCRP3 = None)

            if batch == 'Poff_Diff':
                cur_metamodel = get_pm_metamodel(observed_t_KSEG1 = fixed_x,
                                                 observed_k_KSEG1 = fixed_y,
                                                 observed_logPoff_LCKA2 = x_in, 
                                                 observed_logDiff_LCKA2 = y_in,
                                                 observed_RgRatio_TCRP3 = None,
                                                 observed_phosRatio_TCRP3 = None)

            if batch == 'Poff_k':
                cur_metamodel = get_pm_metamodel(observed_t_KSEG1 = fixed_x,
                                                 observed_k_KSEG1 = y_in,
                                                 observed_logPoff_LCKA2 = x_in, 
                                                 observed_logDiff_LCKA2 = fixed_y,
                                                 observed_RgRatio_TCRP3 = None,
                                                 observed_phosRatio_TCRP3 = None)


            with cur_metamodel:
                cur_trace_metamodel = pm.sample(2000, chains=4, progressbar = False);

            print(f"i,dep={i,y_in}, j,lp={j,x_in}")

            RgRatios_mean[i,j] = cur_trace_metamodel.rv_output_RgRatio_TCRP3.mean() 
            RgRatios_std[i,j] = cur_trace_metamodel.rv_output_RgRatio_TCRP3.std()

            phosRatios_mean[i,j] = cur_trace_metamodel.rv_output_phosRatio_TCRP3.mean() 
            phosRatios_std[i,j] = cur_trace_metamodel.rv_output_phosRatio_TCRP3.std()
"""
