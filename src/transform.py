import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import numba as nb
import pandas as pd
import scipy.integrate as integrate
import os
import pathlib
import tqdm
from . import writers
#import pickle
#pickle.HIGHEST_PROTOCOL = 4

#==================================================================================================================================================================
# Helper functions
#==================================================================================================================================================================

@nb.jit(nb.int64(nb.int64))
def get_cur_factor(rng_cur):
    cur_scale_10   = 10
    cur_scale_100  = 100
    cur_scale_1000 = 1000

    cur_scale_factor_10   = 10000
    cur_scale_factor_100  = 1000
    cur_scale_factor_1000 = 100
    cur_scale_factor_max  = 10

    if rng_cur >= 0:
        factor = cur_scale_factor_max
        if rng_cur < cur_scale_10:
            factor = cur_scale_factor_10
        elif (rng_cur >= cur_scale_10) & (rng_cur < cur_scale_100):
            factor = cur_scale_factor_100
        elif (rng_cur >= cur_scale_100) & (rng_cur < cur_scale_1000):
            factor = cur_scale_factor_1000
        else:
            factor = cur_scale_factor_max
    
    #Or if rng_cur is negative
    else:
        d_rng_cur = 0 #mA
        rng_cur = abs(rng_cur)
        if (rng_cur>0) & (rng_cur<999): #Negative current between 0 and 999 is mA
            d_rng_cur = rng_cur
        elif (rng_cur>=1000) & (rng_cur<999999): #Negative current between 1000 and 999999 is mA ##Is this wrong?
            d_rng_cur = rng_cur
        elif (rng_cur>=1000000) & (rng_cur<=999999999):# Negtive current between 1000000-999999999 is uA
            d_rng_cur = rng_cur / 1000000000.0

        factor = cur_scale_factor_max
        if (d_rng_cur<0.01): #10uA
            factor = 100000000
        elif (d_rng_cur>=0.01) & (d_rng_cur<0.1): #100uA
            factor = 10000000
        elif (d_rng_cur>=0.1) & (d_rng_cur<1.0): #1000uA
            factor = 1000000
        elif (d_rng_cur>=1.0) & (d_rng_cur<10.0): #10mA
            factor = 100000
        elif (d_rng_cur>=10.0) & (d_rng_cur<100.0): #100mA
            factor = 10000
        elif (d_rng_cur>=100.0) & (d_rng_cur<1000.0): #1000mA
            factor = 1000
        elif (d_rng_cur>=1000.0): #
            factor = 100.0
        else:
            factor = 10.0   
    #Factor returns transformation to milli_ampere
    return factor

@nb.njit(nb.float64[:](nb.int64[:], nb.int64[:]), parallel=True)
def transform_current(test_cur, rng_cur):
    n  = len(test_cur)
    current = np.zeros(n)
    for i in nb.prange(n):
        current[i] = 1e-3 * test_cur[i] / get_cur_factor(rng_cur[i])
    return current

@nb.njit(nb.float64[:](nb.int64[:], nb.int64[:], nb.int64[:]),parallel=True)
def transform_capacity(cap, factor_cap, rng_cur):
    #factor_cap==0: Unit is [mAs]
    #factor_cap==1: Unit is [Ah]
    #factor_cap==2: Unit is [mAh]
    n=len(cap)
    capacity = np.zeros(n)
    for i in nb.prange(n):
        #Get the current-transformation factor
        cur_factor = get_cur_factor(rng_cur[i])

        #Get corrent time-unit converter
        if factor_cap[i] == 0: #Transform to real units and from mAs to Ah
            cap_factor = 1e3 * 3600
        elif factor_cap[i] == 1: #Transform to real units
            cap_factor = 1
        elif factor_cap[i] == 2: #Transform to real units and from mAh to Ah
            cap_factor = 1e3
            
        capacity[i] = cap[i]/cur_factor/cap_factor
    return capacity

@nb.njit(nb.float64[:](nb.int64[:], nb.int64[:], nb.int64[:]),parallel=True)
def transform_energy(eng, factor_eng, rng_cur):
    #factor_cap==0: Unit is [mWs]
    #factor_cap==1: Unit is [Wh]
    #factor_cap==2: Unit is [mWh]
    n=len(eng)
    energy = np.zeros(n)
    for i in nb.prange(n):
        #Get the current-transformation factor
        cur_factor = get_cur_factor(rng_cur[i])

        #Get corrent time-unit converter
        if factor_eng[i] == 0: #Transform to real units and from mAs to Ah
            eng_factor = 1e3 * 3600
        elif factor_eng[i] == 1: #Transform to real units
            eng_factor = 1
        elif factor_eng[i] == 2: #Transform to real units and from mAh to Ah
            eng_factor = 1e3
            
        energy[i] = eng[i]/cur_factor/eng_factor
    return energy  


def transform_testtime(testtime):
    return testtime / 1000

def transform_voltage(test_vol):
    return test_vol / 10000


def transform_temperature(test_tmp):
    return test_tmp/10    

@nb.njit(parallel=True)
def step_to_comm(steps):
    n=len(steps)
    stepstr = ['UNKNOWN']*n
    for i in nb.prange(n):
        if steps[i]==1:
            stepstr[i]='CC_Chg'
        elif steps[i]==2:
            stepstr[i]='CC_Dchg'
        elif steps[i]==3:
            stepstr[i]='N/A'
        elif steps[i]==4:
            stepstr[i]='Pause'
        elif steps[i]==5:
            stepstr[i]='Cycle'
        elif steps[i]==6:
            stepstr[i]='End'
        elif steps[i]==7:
            stepstr[i]='CCCV_Chg'
        elif steps[i]==8:
            stepstr[i]='CP_Dchg'
        elif steps[i]==9:
            stepstr[i]='CP_Chg'
        elif steps[i]==10:#WHAT IS 11-19??
            stepstr[i]='CR_Dchg'
        elif steps[i]==20:
            stepstr[i]='CCCV_Dchg'
        else:
            pass
    return stepstr     


#===============================Transformation functions==============================================================
def data(obj):

    if isinstance(obj, writers.FileHandler):
        data = obj.read()
    elif isinstance(obj, pd.DataFrame):
        data = obj
    else:
        raise TypeError(f'transform.data should be given FileHandler or pd.DataFrame, not {type(obj)}')

    if data.empty:
        return data

    data.loc[data.index, 'test_cur'] = transform_current(data['test_cur'].values, data['cur_step_range'].values)
    data.loc[data.index, 'test_vol'] = transform_voltage(data['test_vol'].values)
    #
    data.loc[data.index, 'test_capchg']  = transform_capacity(data['test_capchg'].values,  data['factor_capchg'].values,  data['cur_step_range'].values)
    data.loc[data.index, 'test_capdchg'] = transform_capacity(data['test_capdchg'].values, data['factor_capdchg'].values, data['cur_step_range'].values)
    #
    data.loc[data.index, 'test_engchg']  = transform_energy(data['test_engchg'].values,  data['factor_engchg'].values,  data['cur_step_range'].values)
    data.loc[data.index, 'test_engdchg'] = transform_energy(data['test_engdchg'].values, data['factor_engdchg'].values, data['cur_step_range'].values)   
    #
    data.loc[data.index, 'test_tmp'] = transform_temperature(data.test_tmp.values)
    #
    data.loc[data.index, 'step_type'] = step_to_comm(data.step_type.values)   
    data.loc[data.index, 'test_time'] = transform_testtime(data.test_time.values)


    renames = {'test_cur': 'Current [A]',
                'test_vol': 'Voltage [V]',
                'test_time':'Steptime [s]',
                'test_tmp': 'Temperature [degC]',
                'test_atime':'Time [datetime]',
                'step_id':'line',
                'step_type':'comm',
                'cycle':'Cycle #'}
    
    
    data = data.rename(columns=renames, inplace=False)

    data.loc[:,'Capacity [Ah]'] = data.test_capchg - data.test_capdchg
    data = data.drop(['test_capchg', 'test_capdchg'], inplace=False,axis=1)

    data.loc[:,'Energy [Wh]'] = data.test_engchg - data.test_engdchg
    data = data.drop(['test_engchg', 'test_engdchg'], inplace=False,axis=1)

    data.loc[:, 'Power [W]'] = data.loc[:,'Current [A]']*data.loc[:,'Voltage [V]']

    #Drop not-needed columns
    for col in ['test_ir', 'factor_capchg','factor_capdchg', 'factor_engchg','factor_engdchg', 'dataupdate']:
        if col in data.columns:
            data = data.drop([col],axis=1,inplace=False)
    return data