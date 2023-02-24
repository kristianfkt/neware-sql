import numpy as np
import numba as nb
import pandas as pd
import scipy.integrate as integrate
import logging
import os
import pathlib
import tqdm
from . import writers
#import pickle
#pickle.HIGHEST_PROTOCOL = 4
logger = logging.getLogger('Neware Transformer')
logging.basicConfig(level=logging.WARNING)

#==================================================================================================================================================================
# Helper functions
#==================================================================================================================================================================

@nb.jit(nb.int64(nb.int64))
def _get_cur_factor(rng_cur):
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
def _transform_current(test_cur,rng_cur):
    n  = len(test_cur)
    current = np.zeros(n)
    for i in nb.prange(n):
        current[i] = 1e-3 * test_cur[i] / _get_cur_factor(rng_cur[i])
    return current

@nb.njit(nb.float64[:](nb.int64[:], nb.int64[:], nb.int64[:]),parallel=True)
def _transform_capacity(cap, factor_cap, rng_cur):
    #factor_cap==0: Unit is [mAs]
    #factor_cap==1: Unit is [Ah]
    #factor_cap==2: Unit is [mAh]
    n=len(cap)
    capacity = np.zeros(n)
    for i in nb.prange(n):
        #Get the current-transformation factor
        cur_factor = _get_cur_factor(rng_cur[i])

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
def _transform_energy(eng, factor_eng, rng_cur):
    #factor_cap==0: Unit is [mWs]
    #factor_cap==1: Unit is [Wh]
    #factor_cap==2: Unit is [mWh]
    n=len(eng)
    energy = np.zeros(n)
    for i in nb.prange(n):
        #Get the current-transformation factor
        cur_factor = _get_cur_factor(rng_cur[i])

        #Get corrent time-unit converter
        if factor_eng[i] == 0: #Transform to real units and from mAs to Ah
            eng_factor = 1e3 * 3600
        elif factor_eng[i] == 1: #Transform to real units
            eng_factor = 1
        elif factor_eng[i] == 2: #Transform to real units and from mAh to Ah
            eng_factor = 1e3
            
        energy[i] = eng[i]/cur_factor/eng_factor
    return energy  

@nb.njit(parallel=True)
def _step_to_comm(steps):
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

#==================================================================================================================================================================
# Helper functions
#==================================================================================================================================================================

class Transformer:
    def __init__(self):
        return
#===============================Transformation functions==============================================================
    def transform(self, obj):
        if isinstance(obj, writers.BaseWriter):
            data = obj.read()
        elif isinstance(obj, pd.DataFrame):
            data = obj
        else:
            raise TypeError(f'Transformer made for use with pandas DataFrames. Please pass a file or preloaded data')

        if data.empty:
            logger.info('Data empty. returning empty DataFrama')
            return data

        #Functions for direct transformation
        transformers = {
            'test_cur':    self.transform_current,
            'test_vol':    self.transform_voltage,
            'test_capchg': self.transform_capacity_charge,
            'test_capdchg':self.transform_capacity_discharge,
            'test_engchg': self.transform_energy_charge,
            'test_engdchg':self.transform_energy_discharge,            
            'test_tmp':    self.transform_temperature,
            'step_type':   self.transform_steptype,
            'test_time':   self.transform_testtime}

        renames = {'test_cur': 'Current [A]',
                   'test_vol': 'Voltage [V]',
                   'test_time':'Steptime [s]',
                   'test_tmp': 'Temperature [degC]',
                   'test_atime':'Time [datetime]',
                   'step_id':'line',
                   'step_type':'comm',
                   'cycle':'Cycle #'}
        
        cols = data.columns

        keys = list(transformers.keys())
        logging.debug(f'Transforming')
        for col in cols:
            if col in keys:
                logging.debug(f'Transforming {col}')
                data.loc[:, col] = transformers[col](data)
        
        logging.debug(f'Renaming columns')
        data = data.rename(columns=renames, inplace=False)

        data.loc[:,'Capacity [Ah]'] = data.test_capchg - data.test_capdchg
        data = data.drop(['test_capchg', 'test_capdchg'], inplace=False,axis=1)

        data.loc[:,'Energy [Wh]'] = data.test_engchg - data.test_engdchg
        data = data.drop(['test_engchg', 'test_engdchg'], inplace=False,axis=1)

        data.loc[:, 'Power [W]'] = data.loc[:,'Current [A]']*data.loc[:,'Voltage [V]']

        logging.debug(f'Dropping excess columns')
        #Drop not-needed columns
        for col in ['test_ir', 'factor_capchg','factor_capdchg', 'factor_engchg','factor_engdchg', 'dataupdate']:
            if col in data.columns:
                data = data.drop([col],axis=1,inplace=False)
        return data

    def transform_testtime(self,data):
        return data.test_time.values / 1000


    def transform_voltage(self,data):
        return data.test_vol.values / 10000
    
    def transform_current(self,data):
        return _transform_current(data.test_cur.values, data.cur_step_range.values)

    def transform_temperature(self,data):
        return data.test_tmp.values/10

    def transform_capacity_charge(self,data):
        return _transform_capacity(data['test_capchg'].values,  data['factor_capchg'].values,  data['cur_step_range'].values)
    
    def transform_capacity_discharge(self,data):
        return _transform_capacity(data['test_capdchg'].values, data['factor_capdchg'].values, data['cur_step_range'].values)

    def transform_energy_charge(self,data):
        return _transform_energy(data['test_engchg'].values,  data['factor_engchg'].values,  data['cur_step_range'].values)
    
    def transform_energy_discharge(self,data):
        return _transform_energy(data['test_engdchg'].values, data['factor_engdchg'].values, data['cur_step_range'].values)   

    def transform_steptype(self,data):
        return _step_to_comm(data.step_type.values)   


# #===============================Adding- functions==============================================================
#     def add(self, obj, cell=None):
#         if isinstance(obj, (core.ParquetFile, core.CSVFile, core.HDFFile, core.ExcelFile)):
#             data = obj.read()
#         elif isinstance(obj, pd.DataFrame):
#             data = obj
#         else:
#             raise TypeError(f'Transformer made for use with pandas DataFrames. Please pass a file or preloaded data')

#         if data.empty:
#             logger.info('Data empty. returning empty DataFrama')
#             return data

#         #Time columns
#         tau = data['Time [datetime]']  - data['Time [datetime]'].min()
#         data.loc[:, 'Time [s]']     = tau.values.astype(np.int64)/int(1e9)
#         data.loc[:, 'Time [h]']     = data.loc[:, 'Time [s]']/3600
#         data.loc[:, 'Steptime [h]'] = data.loc[:, 'Steptime [s]']/3600
#         dt_mode = data.loc[:, 'Steptime [s]'].diff().mode()[0]
#         data.loc[:, 'Testtime [s]'] = np.arange(0, dt_mode*len(data), dt_mode)
#         data.loc[:, 'Testtime [h]'] = data.loc[:, 'Testtime [s]']/3600     

#         if isinstance(cell, str):
#             if cell in list(self.cells.keys()):
#                 capacity = self.cells[cell]['capacity']
#                 energy = self.cells[cell]['energy']
#                 data.loc[:,'DoD_cap'] = data.loc[:,'Capacity [Ah]']/capacity
#                 data.loc[:,'DoD_eng'] = data.loc[:,'Energy [Wh]']/energy
#                 data.loc[:,'C-rate']  = data.loc[:,'Current [A]']/capacity
#                 data.loc[:,'P-rate']  = data.loc[:,'Power [W]']/energy
#                 data.loc[:,'Throughput [Ah]'] = integrate.cumtrapz(data['Current [A]'].abs(), x=data['Testtime [h]'], initial=0)
#                 data['EFC']             = data['Throughput [Ah]']/2/capacity

        #Add nans
        # data.loc[:, 'Efficiency_cap'] = np.nan
        # data.loc[:, 'Efficiency_eng'] = np.nan        
        # #Add current- and energy efficiency
        # chg = ['CC_Chg',  'CP_Chg',  'CCCV_Chg',  'CR_Chg']
        # dhg = ['CC_Dchg', 'CP_Dchg', 'CCCV_Dchg', 'CR_Dchg']   
        # #filth = (data['comm'].isin(chg) | data['comm'].isin(dhg)) & (~data['comm'].isin(['Pause'])) & (data['line'].diff(periods=-1) != 0)
        # filth_c = (data['comm'].isin(chg)) & (~data['comm'].isin(['Pause'])) & (data['line'].diff(periods=-1) != 0)
        # filth_d = (data['comm'].isin(dhg)) & (~data['comm'].isin(['Pause'])) & (data['line'].diff(periods=-1) != 0)
        # idx_c = filth_c[filth_c].index
        # idx_d = filth_d[filth_d].index

        # n = min([len(idx_c), len(idx_d)])
        # if n==0:
        #     return data

        # data.loc[idx_d[0:n], 'Efficiency_cap'] = -data.loc[idx_d[0:n],'Capacity [Ah]'].values[0:n]/data.loc[idx_c[0:n],'Capacity [Ah]'].values
        # #_add_efficiency(idxs.values, numba.typed.List(data.comm.astype(str).to_list()), data['Capacity [Ah]'].values)
        # data.loc[idx_d[0:n], 'Efficiency_eng'] = -data.loc[idx_d[0:n],'Energy [Wh]'].values/data.loc[idx_c[0:n],'Energy [Wh]'].values
        return data

        


# class CellParser:
#     def __init__(self, file_format='parquet'):
#         self.cellsprops = core.cell_properties()
#         self.dataparser = DataParser()
#         self.set_file_handler(file_format)
#         return
    
#     def set_file_handler(self, file_format):
#         formats = {
#             'parquet':core.ParquetFile,
#             'parq':core.ParquetFile,
#             'hdf':core.HDFFile,
#             'h5':core.HDFFile,
#             'csv':core.CSVFile,
#             'txt':core.CSVFile,
#             'excel':core.ExcelFile,
#             'xlsx':core.ExcelFile
#         }
#         if callable(file_format):
#             self.save_file_handler = file_format
#             logger.debug(f'CellPArser configured for {file_format}')
#         elif isinstance(file_format, str):
#             save_format = file_format.lower()
#             if save_format in list(formats.keys()):
#                 self.save_file_handler = formats[save_format]
#                 logger.debug(f'CellParser configured for {formats[save_format]} with {save_format} key')
#             else:
#                 raise ValueError(f'Unknown file_format {save_format}. Should be {list(formats.keys())}')
#         else:
#             raise TypeError(f'file_format should be string or callable')
#         return        

#     def join_tests(self):
#         cells = {}
#         all_cell_names = list(self.cellsprops.keys())
#         basenames = {}
#         for root,dirs,files in os.walk(core.Paths().tests):
#             R=pathlib.Path(root)
#             logger.debug(f'Root is {R}')
#             for dir in dirs:
#                 logger.debug(f'Dir is {dir}')
#                 D = R.joinpath(dir)
#                 if (D.is_dir()) & (dir in all_cell_names):
#                     logger.debug(f'Adding {dir} to cells')
#                     cells[dir] = {}
#                     basenames[dir] = {}
#             for file in files:
#                 F = core.FileParser().parse(R,file)
#                 if isinstance(F, self.save_file_handler):
#                     name =   str(F.name).split('-')[4]
#                     serial = str(F.name).split('-')[5]
#                     if name in list(cells.keys()):
#                         if serial in list(cells[name].keys()):
#                             cells[name][serial].append(F)
#                         else:
#                             cells[name][serial] = [F]
#                             basenames[name][serial] = '-'.join(str(F.name).split('-')[:6])
#         logger.debug(f'Found cells {list(cells.keys())}')
        
#         pbar_cell = tqdm.tqdm(cells.items())
#         pbar_serial = tqdm.tqdm([])
#         for cell,serials in pbar_cell:
#             pbar_cell.set_description(f'{cell}')
#             pbar_serial.total=len(serials.items())
#             pbar_serial.refresh()
#             pbar_serial.reset()
#             for serial,files in serials.items():
#                 pbar_serial.set_description(f'{cell}-{serial}')
#                 self.__join_tests__(cell,files,basenames[cell][serial])
#                 pbar_serial.update(1)
#         pbar_serial.close()
#         pbar_cell.close()
#         return

#     def __join_tests__(self, cell, files, basename):
#         logger.debug(f'Joining {len(files)} to {basename}')
#         paths = core.Paths(cell=cell)
#         paths.make()
#         path = paths.cells
#         data=pd.DataFrame()
#         for file in files:
#             logger.debug(f'Readning {file}')
#             data = data.append(file.read(), ignore_index=True)
#         data = data.sort_values('Time [datetime]').reset_index(drop=True)
#         data = self.dataparser.add(data, cell=cell)
#         File = self.save_file_handler(path, basename)
#         File.write(data)
#         return


if __name__=='__main__':
    pass
    # CellParser().join_tests()