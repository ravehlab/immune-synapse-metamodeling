# -*- coding: utf-8 -*-
"""coupled_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NlPoUAuuG2cYNhiZp4o1s-ztjWg1tIs9
"""

!pip install arviz -q #==0.6.1
# !pip install Theano==1.0.5 -q
!pip install theano-pymc -q
!pip install pymc3==3.10.0 -q

import numpy as np
import scipy as sp
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
import scipy.stats as stats
import arviz as az

print('Running on PyMC3 v{}'.format(pm.__version__))
print('Running on ArviZ v{}'.format(az.__version__))

# defining colors for the diferent entities
TCR_color = np.array([0.0, 0.6, 0.0])
CD45_color = np.array([1.0, 0.0, 0.0])
LCK_color = np.array([0.0, 0.0, 0.4])
aLCK_color = np.array([1.0, 0.6, 1.0])
pTCR_color = np.array([1.0, 0.6, 0.0])
membrane_color = 0.6*np.array([1.0, 1.0, 1.0])

"""## Metamodel containing all surrogate models and their coupling

### Abbreviations:
ac - slope of 'c' (e.g. at for slope of t)
"""

# Model 1 constants:
# surface t slope:
mu_at_dm_CD45_KS1 = 4.974 
sd_at_dm_CD45_KS1 = 0.010
# surface k slope:
mu_ak_dm_CD45_KS1 = 9.956
sd_ak_dm_CD45_KS1 = 0.020

mu_b_dm_CD45_KS = 1.943
sd_b_dm_CD45_KS = 0.763
mu_noise_dm_CD45_KS = 6.776
sd_noise_dm_CD45_KS = 0.205
# --
# Model 3 constants:
# 
mu_noise_dw_aLCK_LA = 36.861
sd_noise_dw_aLCK_LA = 1.297
mu_dw_aLCK_log_Diff_divisor_LA = 0.366
sd_dw_aLCK_log_Diff_divisor_LA = 0.025
mu_dw_aLCK_log_Poff_center_LA = -1.693
sd_dw_aLCK_log_Poff_center_LA = 0.060
mu_dw_aLCK_min_LA = 1.007	
sd_dw_aLCK_min_LA = 0.975
mu_dw_aLCK_max_LA = 346.700
sd_dw_aLCK_max_LA = 9.351
mu_dw_aLCK_log_Diff_center_LA = 0.909	
sd_dw_aLCK_log_Diff_center_LA = 0.031
mu_dw_aLCK_log_Poff_divisor_LA = -0.745
sd_dw_aLCK_log_Poff_divisor_LA = 0.050
# --
# Model 2 constants:
# dm_pTCR:
mu_dm_pTCR_aLCK_divisor_TP = 57.665	
sd_dm_pTCR_aLCK_divisor_TP = 13.519	
mu_dm_pTCR_CD45_center_TP	= 395.945	
sd_dm_pTCR_CD45_center_TP	= 79.366	
mu_noise_dm_pTCR_TP = 44.663	
sd_noise_dm_pTCR_TP = 1.036	
mu_dm_pTCR_min_TP = 209.912	
sd_dm_pTCR_min_TP = 4.592	
mu_dm_pTCR_max_TP = 47.437
sd_dm_pTCR_max_TP = 28.111	
mu_dm_pTCR_aLCK_center_TP = 107.104
sd_dm_pTCR_aLCK_center_TP = 16.94	
mu_dm_pTCR_CD45_divisor_TP = -22.562	
sd_dm_pTCR_CD45_divisor_TP = 44.569

# dw_pTCR:
mu_dw_pTCR_intercept_TP	= 75.507	
sd_dw_pTCR_intercept_TP	= 1.214
mu_noise_dw_pTCR_TP = 10.531
sd_noise_dw_pTCR_TP = 0.240
mu_dw_pTCR_aLCK_slope_TP = 0.152
sd_dw_pTCR_aLCK_slope_TP = 0.004
mu_dw_pTCR_dm_CD45_slope_TP = -0.090
sd_dw_pTCR_dm_CD45_slope_TP = 0.002

def get_metamodel(observed_t_KS1= None,
                  observed_k_KS1= None,
                  observed_log_Poff_LA3= None, 
                  observed_log_Diff_LA3= None):
    ''' return a metamodel with all surrogate models '''
    metamodel = pm.Model()
    with metamodel:
        ### model1 - KS (kinetic segregation) ###########################    
        param_t = pm.Uniform('param_t', 0, 100, 
                          observed= observed_t_KS1)
        param_k = pm.Uniform('param_k', 0, 50,
                          observed= observed_k_KS1)
        rv_at_dm_CD45_KS1 = pm.Normal('rv_at_dm_CD45_KS1', 
                                    mu=mu_at_dm_CD45_KS1, 
                                    sd=sd_at_dm_CD45_KS1) # surface t slope
        rv_ak_dm_CD45_KS1 = pm.Normal('rv_ak_dm_CD45_KS1', 
                                    mu=mu_ak_dm_CD45_KS1, 
                                    sd=sd_ak_dm_CD45_KS1) # surface k slope
        rv_b_dm_CD45_KS1 = pm.Normal('rv_b_dm_CD45_KS1', 
                                    mu=mu_b_dm_CD45_KS, 
                                    sd=sd_b_dm_CD45_KS) # surface intercept
        rv_noise_dm_CD45_KS1 = pm.HalfNormal('rv_noise_dm_CD45_KS1', 
                                            sd=mu_noise_dm_CD45_KS) # noise 
        rv_dm_CD45_KS1 = pm.Normal('rv_dm_CD45_KS1', 
                                  mu=rv_b_dm_CD45_KS1 +\
                                  rv_at_dm_CD45_KS1*param_t +\
                                  rv_ak_dm_CD45_KS1*param_k,
                                  sd=rv_noise_dm_CD45_KS1) #

        ### model3 - LA (LCK activation) ###########################    
        # random variables x and y
        param_log_Diff = pm.Uniform('param_log_Diff', 0, 3,
                                 observed= observed_log_Diff_LA3)
        param_log_Poff = pm.Uniform('param_log_Poff', -6, 0,
                                 observed= observed_log_Poff_LA3)
        
        # random variables
        noise_dw_aLCK_LA3 = pm.HalfNormal('noise_dw_aLCK_LA3', 
                                         sd=mu_noise_dw_aLCK_LA)
        # Sigmoid params
        rv_dw_aLCK_min_LA3 = pm.HalfNormal('rv_dw_aLCK_min_LA3', 
                                          sd=mu_dw_aLCK_min_LA)
        rv_dw_aLCK_max_LA3 = pm.HalfNormal('rv_dw_aLCK_max_LA3', 
                                          sd=mu_dw_aLCK_max_LA)
        rv_dw_aLCK_log_Diff_center_LA3 = pm.TruncatedNormal('rv_dw_aLCK_log_Diff_center_LA3',
                                                            mu=mu_dw_aLCK_log_Diff_center_LA,
                                                            sd=sd_dw_aLCK_log_Diff_center_LA,
                                                            lower=0)
        rv_dw_aLCK_log_Diff_divisor_LA3 = pm.Normal('rv_dw_aLCK_log_Diff_divisor_LA3',
                                                    mu=mu_dw_aLCK_log_Diff_divisor_LA,
                                                    sd=sd_dw_aLCK_log_Diff_divisor_LA)
        rv_dw_aLCK_log_Poff_center_LA3 = pm.Normal('rv_dw_aLCK_log_Poff_center_LA3',
                                                  mu=mu_dw_aLCK_log_Poff_center_LA,
                                                  sd=sd_dw_aLCK_log_Poff_center_LA)
        rv_dw_aLCK_log_Poff_divisor_LA3 = pm.TruncatedNormal('rv_dw_aLCK_log_Poff_divisor_LA3', 
                                                      mu=mu_dw_aLCK_log_Poff_divisor_LA, 
                                                      sd=sd_dw_aLCK_log_Poff_divisor_LA,
                                                      upper=0)

        rv_tmp_x1_LA3 = (param_log_Diff - rv_dw_aLCK_log_Diff_center_LA3) / rv_dw_aLCK_log_Diff_divisor_LA3
        rv_tmp_sig1_LA3 = 1.0 / (1 + np.exp(-rv_tmp_x1_LA3))
        rv_tmp_x2_LA3 = (param_log_Poff - rv_dw_aLCK_log_Poff_center_LA3) / rv_dw_aLCK_log_Poff_divisor_LA3
        rv_tmp_sig2_LA3 = 1.0 / (1 + np.exp(-rv_tmp_x2_LA3))

        rv_dw_aLCK_LA3 = pm.Normal('rv_dw_aLCK_LA3', mu=rv_dw_aLCK_min_LA3 +\
                                  (rv_dw_aLCK_max_LA3 - rv_dw_aLCK_min_LA3)*\
                                  rv_tmp_sig1_LA3 * rv_tmp_sig2_LA3,
                                  sd=noise_dw_aLCK_LA3)
        ### Coupling model ###########################    
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
        
        ### dm_pTCR:
        rv_noise_dm_pTCR_TP2 = pm.HalfNormal('rv_noise_dm_pTCR_TP2', 
                                            sd=mu_noise_dm_pTCR_TP)
        # Sigmoid params
        rv_dm_pTCR_min_TP2= pm.HalfNormal('rv_dm_pTCR_min_TP2',
                                         sd=mu_dm_pTCR_min_TP)
        rv_dm_pTCR_max_TP2= pm.HalfNormal('rv_dm_pTCR_max_TP2',
                                         sd=mu_dm_pTCR_max_TP)
        rv_dm_pTCR_aLCK_center_TP2=  pm.TruncatedNormal('rv_dm_pTCR_aLCK_center_TP2', 
                                                      mu=mu_dm_pTCR_aLCK_center_TP,
                                                      sd=sd_dm_pTCR_aLCK_center_TP, 
                                                      lower=0)
        rv_dm_pTCR_aLCK_divisor_TP2=  pm.Normal('rv_dm_pTCR_aLCK_divisor_TP2', 
                                               mu=mu_dm_pTCR_aLCK_divisor_TP,
                                               sd=sd_dm_pTCR_aLCK_divisor_TP)
        rv_dm_pTCR_CD45_center_TP2=  pm.Normal('rv_dm_pTCR_CD45_center_TP2', 
                                              mu=mu_dm_pTCR_CD45_center_TP,
                                              sd=sd_dm_pTCR_CD45_center_TP)
        rv_dm_pTCR_CD45_divisor_TP2=  pm.TruncatedNormal('rv_dm_pTCR_CD45_divisor_TP2',
                                                        mu=mu_dm_pTCR_CD45_divisor_TP,
                                                        sd=sd_dm_pTCR_CD45_divisor_TP,
                                                        upper=0)
        rv_tmp_x1_TP2= (rv_dw_aLCK_TP2 - rv_dm_pTCR_aLCK_center_TP2) / rv_dm_pTCR_aLCK_divisor_TP2
        rv_tmp_sig1_TP2=  1.0 / (1 + np.exp(-rv_tmp_x1_TP2))
        rv_tmp_x2_TP2= (rv_dm_CD45_TP2 - rv_dm_pTCR_CD45_center_TP2) / rv_dm_pTCR_CD45_divisor_TP2
        rv_tmp_sig2_TP2=  1.0 / (1 + np.exp(-rv_tmp_x2_TP2))
        
        rv_dm_pTCR_TP2 = pm.Normal('rv_dm_pTCR_TP2',\
                        mu= rv_dm_pTCR_min_TP2 + 
                        (rv_dm_pTCR_max_TP2-rv_dm_pTCR_min_TP2) * rv_tmp_sig1_TP2 * rv_tmp_sig2_TP2,\
                        sd=rv_noise_dm_pTCR_TP2)

        ### dw_pTCR:
        rv_noise_dw_pTCR_TP2 = pm.HalfNormal('rv_noise_dw_pTCR_TP2', 
                                            sd=mu_noise_dw_pTCR_TP) # noise 
        
        # linear params
        rv_dw_pTCR_intercept_TP2=  pm.Normal('rv_dw_pTCR_intercept_TP2', 
                                            mu=mu_dw_pTCR_intercept_TP, 
                                            sd=sd_dw_pTCR_intercept_TP)
        rv_dw_pTCR_aLCK_slope_TP2=  pm.TruncatedNormal('rv_dw_pTCR_aLCK_slope_TP2', 
                                                      mu=mu_dw_pTCR_aLCK_slope_TP,
                                                      sd=sd_dw_pTCR_aLCK_slope_TP, 
                                                      lower=0)
        rv_dw_pTCR_dm_CD45_slope_TP2=  pm.TruncatedNormal('rv_dw_pTCR_dm_CD45_slope_TP2', 
                                                         mu=mu_dw_pTCR_dm_CD45_slope_TP, 
                                                         sd=sd_dw_pTCR_dm_CD45_slope_TP,
                                                         upper= 0)
        
        rv_dw_pTCR_TP2 = pm.Normal('rv_dw_pTCR_TP2', 
                                  mu=rv_dw_pTCR_intercept_TP2 +\
                                  rv_dw_pTCR_aLCK_slope_TP2*rv_dw_aLCK_TP2 +\
                                  rv_dw_pTCR_dm_CD45_slope_TP2*rv_dm_CD45_TP2,
                                  sd=rv_noise_dw_pTCR_TP2) 


    return metamodel

metamodel= get_metamodel(observed_t_KS1= 75.0,
                         observed_k_KS1= 34.0,
                         observed_log_Poff_LA3= -3.0,
                         observed_log_Diff_LA3= 1.0
                         )
gv_metamodel = pm.model_to_graphviz(metamodel)
display(gv_metamodel)

from google.colab import files
gv_metamodel.render("metamodel_graph", format="png")
files.download("metamodel_graph.png") # Download locally from colab

with metamodel:
    trace_metamodel = pm.sample(2000, chains=4)

pm.traceplot(trace_metamodel);

# !cat /proc/cpuinfo

pm.summary(trace_metamodel).round(2)

vars(metamodel)

pm.summary(trace_metamodel, ['rv_dm_pTCR_TP2', 'rv_dw_pTCR_TP2'])

# metamodel= get_metamodel(observed_t_KS1= 75.0,
#                          observed_k_KS1= 34.0,
#                          observed_log_Poff_LA3= -3.0,
#                          observed_log_Diff_LA3= 1.0
# model1:
# dm_CD45:

# model2:

# model3:

mu_rv_dm_pTCR_TP2 = trace_metamodel.rv_dm_pTCR_TP2.mean()
sd_rv_dm_pTCR_TP2 = trace_metamodel.rv_dm_pTCR_TP2.std()
mu_rv_dw_pTCR_TP2 = trace_metamodel.rv_dw_pTCR_TP2.mean()
sd_rv_dw_pTCR_TP2 = trace_metamodel.rv_dw_pTCR_TP2.std()

print('mu_rv_dm_pTCR_TP2 =', mu_rv_dm_pTCR_TP2)
print('sd_rv_dm_pTCR_TP2 =', sd_rv_dm_pTCR_TP2)
print('mu_rv_dw_pTCR_TP2 =', mu_rv_dw_pTCR_TP2)
print('sd_rv_dw_pTCR_TP2 =', sd_rv_dw_pTCR_TP2)

xx = np.linspace(0, 1000, 201)

# TCR:
d_TCR = np.zeros(xx.shape)
d_TCR[xx < 250] = 1
# CD45:
mu_rv_dw_aLCK_TP2 = trace_metamodel.rv_dw_aLCK_TP2.mean()	
mu_rv_dm_CD45_TP2 = trace_metamodel.rv_dm_CD45_TP2.mean()
d_CD45 = np.zeros(xx.shape)
d_CD45[(xx > mu_rv_dm_CD45_TP2-250) & (xx < mu_rv_dm_CD45_TP2+250)] = 1
# aLCK:
d_aLCK = 2*np.exp(-0.5*((xx - mu_rv_dm_CD45_TP2)/mu_rv_dw_aLCK_TP2)**2)
# pTCR:
d_pTCR = np.exp(-0.5*((xx - mu_rv_dm_pTCR_TP2)/mu_rv_dw_pTCR_TP2)**2)


fig, ax3=plt.subplots(figsize=[12,3])

ax3.plot(xx, d_TCR, '-', color=TCR_color)
ax3.plot(xx, d_CD45, '-', color=CD45_color)
ax3.plot(xx, d_aLCK, '-', color=aLCK_color)
ax3.plot(xx, d_pTCR, '-', color=pTCR_color)

