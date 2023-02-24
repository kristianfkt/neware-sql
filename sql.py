"""
Import dependencies
"""
import sqlalchemy
import pandas as pd
import pathlib
import logging
import multiprocessing as mp
import tqdm
import sys
import numpy as np

from . import transformers
from . import writers

logger = logging.getLogger('Neware SQL')
logging.basicConfig(level=logging.DEBUG)




#====================================================================================
def path(root=None):
    """
    Paths are organized
    
    ./neware/neware/
    ./neware/data/
    ./neware/tests/
    
    where root as standard is './neware/
    """
    pth = pathlib.Path(__file__).resolve().parent.parent
    if isinstance(root, str):
        pth = pth.joinpath(root)
    return pth

#====================================================================================
def get_query(connection_string, query, retry=True):
    logger.debug(f'Fetching query {query}')
    engine            = sqlalchemy.create_engine(connection_string)    
    if ('FROM None' in query) | ('FROM nan' in query):
        logger.debug(f'NaN or None in query {query}')
        data = pd.DataFrame()
    else:
        try:
            data = pd.read_sql(query, engine, chunksize=None)
        except Exception as e:
            if retry is True:
                logger.warning(f'query {query} failed with error {e}. Retrying')
                data = get_query(connection_string, query, retry=False)
            else:
                logger.warning(f'query {query} failed with error {e}. Returning empty')
                data = pd.DataFrame()
    return data    

#====================================================================================
class Downloader:

    """
    Dowloads data between seq_id_min and seq_id_max and combines main and aux tables for a test
    Returns transformed data on the fly
    """

    def __init__(self, connection_string, parallel=False):
        self.connection_string = connection_string
        self.parallel          = parallel
        self.transform         = transformers.Transformer().transform

    def get_seq_id_max(self, test):
        """
        Gets max seq_id value for the queries in a test. Returns 0's upon errors. 
        """
        seq_id_max = {}
        for key,query in test.queries.items():
            query_result = get_query(self.connection_string, query.replace('*', 'MAX(seq_id)'))
            if query_result.empty:
                val = 0
            else:
                val = query_result.values[0][0]
                if val is None:
                    val=0
            seq_id_max[key] = val
        return seq_id_max     

    def get_test(self, test, from_seq_id=0, to_seq_id=None):

        seq_id_str = f' AND seq_id > {from_seq_id}'

        if isinstance(to_seq_id, int):
            seq_id_str = seq_id_str + f' AND seq_id < {to_seq_id}'

        data_main_first  = get_query(self.connection_string, test.queries['main_first']  + seq_id_str)
        data_main_second = get_query(self.connection_string, test.queries['main_second'] + seq_id_str)
        data_aux_first   = get_query(self.connection_string, test.queries['aux_first']   + seq_id_str)
        data_aux_second  = get_query(self.connection_string, test.queries['aux_second']  + seq_id_str)     

        #Insert temperatures in first table if main is not empty
        if not data_main_first.empty:
            if (data_aux_first.empty) | (len(data_main_first.index) != len(data_aux_first.index)):
                logger.debug(f'No first aux data in {test.filename}. Settting t_raw=NaN')
                data_main_first['test_tmp'] = np.nan #missing
            elif len(data_main_first.index) == len(data_aux_first.index):
                logger.debug(f'Joining first main and aux temperature in {test.filename}.')
                data_main_first['test_tmp'] = data_aux_first['test_tmp']
        else:
            logger.debug(f'Main first data is empty: {test.filename}')

        #Insert temperatures in second table if main is not empty
        if not data_main_second.empty:
            if (data_aux_second.empty) | (len(data_main_second.index) != len(data_aux_second.index)):
                logger.debug(f'No second aux data in {test.filename}. Settting t_raw=NaN')
                data_main_second['test_tmp'] = np.nan #missing
            elif len(data_main_second.index) == len(data_aux_second.index):
                logger.debug(f'Joining second main and aux temperature in {test.filename}.')
                data_main_second['test_tmp'] = data_aux_second['test_tmp']
        else:
            logger.debug(f'Main second data is empty: {test.filename}')


        #Combine main and second tables
        #Append second table to the first
        if (data_main_second.empty) & (not data_main_first.empty):
            data = data_main_first
        elif (not data_main_second.empty) & (data_main_first.empty):
            data = data_main_second
        elif (not data_main_second.empty) & (not data_main_first.empty):
            data = pd.concat([data_main_first,data_main_second], ignore_index=True)
        else:
            data = pd.DataFrame()

    
        if not data.empty:
            #Make sure data is sorted according to real-time and seq_id
            data = data.sort_values(['test_atime', 'seq_id']).reset_index(drop=True)
            data = self.transform(data)
            data.loc[:, 'dev_uid'] = test.channel.unit.device.id
        return data


#====================================================================================
class TestParser:
    def __init__(self, save_format='parquet', filename=None, location=None, identify=None):
        """
        save_format: Used to load the correct writer for saving data.
                str: 
                callable: must accept write(data), append(data), read()

        filename:
                None: batch_on is used as filename
                callable: Accepts a Test and returns a string without file extension
        location: 
                None: tests is saved under /data/
                str: tests is saved under /location/
                callable:  Accepts a Test and returns a string. Location created on the fly
        identify:
                None: Does noting
                callable: Accepts a Test and returns cell_name. Data is then saved under loaction/cell_name
        """
        if callable(save_format):
            self.write = save_format
        self.writer   = writers.writer(save_format.lower())
        self.filename = filename
        self.identify = identify
        self.location = location
        return

#----------------------------------------------------------------------------------------
#    Default parsers below
#-----------------------------------------------------------------------------------------
    def location_parser(self, test):
        """
        Sets the save location for a datafile
        """
        #No location callback: Use standard
        if self.location is None:
            location = path(root='tests')
            #Xheck for cell_name
            if isinstance(test.cell_name, str):
                    location = location.joinpath(test.cell_name)

        elif callable(self.location):
            location = self.location(test)
        else:
            pass

        if isinstance(location, str):
            location = pathlib.Path(location)

        test.save_location = location
        return location

    def filename_parser(self, test):
        if self.filename is None:
            filename = f'{test.info.batch_no}-{test.channel.unit.device.id}-{test.channel.unit.id}-{test.channel.id}-{test.id}'
        elif callable(self.filename):
            filename = self.filename(test)
        else:
            raise TypeError(f'Unknown type filename {type(sef.filename)}')
        test.filename = filename
        return filename

    def identify_parser(self, test):
        if self.identify is None:
            identity = None
        elif isinstance(self.identify, str):
            identity = self.identify

        elif callable(self.identify):
            identity = self.identify(test)
        else:
            raise TypeError(f'Unknown type identify {type(self.identify)}')
        test.cell_name = identity
        return identity
        
        
    def parse(self,test):
        # self.filename_parser(test)
        # self.identify_parser(test)
        # self.location_parser(test)
        # test.writer = self.writer(test.save_location, test.filename)
        return
    
class Repo:
    """
    Repo is initialized to download all sequentially data as parquet
    Data is as standard saved in ./neware/tests/. 
    Use of a callback location function and/or identify is adviced
    n_rows can be set to some low value to only download the first n_rows of a dataset. 
    """
    def __init__(self, connection_string, save_format='parquet', filename=None, location=None, identify=None, parallel=False, download=None, n_rows=None):

        self.connection_string = connection_string
        self.__downloader__    = Downloader(connection_string, parallel=parallel)

        self.parallel          = parallel
        self.download          = download
        
        if isinstance(n_rows, int):
            logger.info(f'Downloading only the first {n_rows} rows.')
        
        self.n_rows=n_rows

        self.__testparser__   = TestParser(save_format=save_format, filename=filename, location=location, identify=identify)
        self.__tests__        = self.__get_tests__()
        self.__notes__        = self.__get_notes__()        
        self.__table__        = self.__make_table__()
        self.__save_table__()

        self.devices      = self.__get_devices__()
        self.units        = [unit for device in self.devices for unit in device.units]
        self.channels     = [channel for unit in self.units for channel in unit.channels]
        self.tests        = [test for channel in self.channels for test in channel.tests]
        #self.parse()
        self.cells        = [test.cell_name for test in self.tests]
        return

    def __get_tests__(self):
        return pd.concat([get_query(self.connection_string, 'SELECT * FROM test'), get_query(self.connection_string, 'SELECT * FROM h_test')], axis=0, ignore_index=True)

    def __get_notes__(self):
        return get_query(self.connection_string, 'SELECT * FROM test_note')

    def __make_table__(self):
        tests = self.__tests__.drop_duplicates()
        notes = self.__notes__
        test_columns    = ['main_first_table', 'main_second_table', 'aux_first_table', 'aux_second_table','start_time','end_time']
        for idx,row in notes.iterrows():
            testrow = tests.query(f'test_id=={row.test_id} and dev_uid=={row.dev_uid} and unit_id=={row.unit_id} and chl_id=={row.chl_id}')
            if testrow.empty:
                continue
            else:
                #If more than one testrow matches tableroq
                if len(testrow)>1:
                    #Check for different first tables
                    check_first  = any(testrow.main_first_table.values[0]!=tab for tab in testrow.main_first_table.values)
                    check_second = any(testrow.main_second_table.values[0]!=tab for tab in testrow.main_second_table.values)
                    if check_first | check_second:
                        raise Exception('Different main first or second tables')
                notes.loc[idx,test_columns] = testrow[test_columns].values[0]
                #Add queries
        notes.loc[:,'main_first_query']  = 'SELECT * FROM ' + notes['main_first_table'].astype(str)  + ' WHERE test_id=' + notes['test_id'].astype(str) + ' AND unit_id=' + notes['unit_id'].astype(str) + ' AND chl_id=' + notes['chl_id'].astype(str)
        notes.loc[:,'main_second_query'] = 'SELECT * FROM ' + notes['main_second_table'].astype(str) + ' WHERE test_id=' + notes['test_id'].astype(str) + ' AND unit_id=' + notes['unit_id'].astype(str) + ' AND chl_id=' + notes['chl_id'].astype(str)
        notes.loc[:,'aux_first_query']   = 'SELECT * FROM ' + notes['aux_first_table'].astype(str)   + ' WHERE test_id=' + notes['test_id'].astype(str) + ' AND unit_id=' + notes['unit_id'].astype(str) + ' AND chl_id=' + notes['chl_id'].astype(str)
        notes.loc[:,'aux_second_query']  = 'SELECT * FROM ' + notes['aux_second_table'].astype(str)  + ' WHERE test_id=' + notes['test_id'].astype(str) + ' AND unit_id=' + notes['unit_id'].astype(str) + ' AND chl_id=' + notes['chl_id'].astype(str)        

        #Sort rows
        notes = notes.sort_values(['dev_uid', 'unit_id', 'chl_id', 'start_time']).reset_index(drop=True)

        #Remove duplicate queries. Keep the latest one
        notes           = notes.drop_duplicates(subset=['main_first_query'],  keep='last')
        notes           = notes.drop_duplicates(subset=['main_second_query'], keep='last')      
        return_columns = ['pair_id', 'test_id', 'dev_uid', 'unit_id', 'chl_id', 'batch_no', 'test_name', 'step_name', 'creator', 'info', 
                          'update_time', 'main_first_table', 'main_second_table', 'aux_first_table', 'aux_second_table', 'start_time', 
                          'end_time', 'main_first_query', 'main_second_query', 'aux_first_query','aux_second_query']  
        return notes.loc[:,return_columns]

    def __save_table__(self):
        self.__table__.to_csv(path(root='data'), index=False)
        return

    def __get_devices__(self):
        devices = []
        dev_uid = get_query(self.connection_string, f'SELECT dev_uid FROM bts612_chlmap').dev_uid.unique()
        for id in  dev_uid:
            devices.append(Device(self, id))
        return devices

    def active(self):
        return [test for test in self.tests if test.active]


    def parse(self):
        self.__table__['filename'] = None
        if not self.parallel:
            print('Starting')
            for test in tqdm.tqdm(self.tests, desc=f'Parsing {len(self.tests)} tests'):
                self.__parse__(test)
        else:
            if self.parallel:
                njobs = mp.cpu_count()
            elif isinstance(self.parallel, int):
                njobs = self.parallel
            else:
                raise ValueError('Parallel should be bool or int')
            with mp.Pool(njobs) as pool:
                chunksize = int( len(self.tests)/njobs )
                bar = pool.imap_unordered(func=self.__parse__, iterable=self.tests, chunksize=chunksize)
                for _ in tqdm.tqdm(bar, total=len(self.tests), desc=f'Parsing {len(self.tests)} tests in {njobs} parallels as {chunksize} chunks'):
                    continue
        return

    def __parse__(self, test):
        #test.parse()
        filth = (self.__table__.dev_uid==test.info.dev_uid) & (self.__table__.unit_id==test.info.unit_id) & (self.__table__.chl_id==test.info.chl_id) & (self.__table__.test_id==test.info.test_id)
        self.__table__.loc[filth, 'filename'] = test.filename            
        return

    def update(self, active_only=True):
        """
        List tests to be downloaded
        All or callback
        """
        if self.download is None:
            tests = self.tests
        elif callable(self.download):
            tests = [test for test in self.tests if self.download(test)]

        if active_only is True:
            tests = [test for test in tests if test.active]
        
        """
        Sequential
        """
        if self.parallel is False:
            bar = tqdm.tqdm(tests)
            for test in bar:
                bar.set_description(test.filename)
                self.__update__(test)
        
        else:
            if self.parallel is True:
                njobs = mp.cpu_count()
            elif isinstance(self.parallel, int):
                njobs = self.parallel
            else:
                raise TypeError(f'parallel should be True/False or int:number of jobs, not {type(self.parallel)}')            
            
            with mp.Pool(njobs) as pool:
                chunksize=1 #We don't know the size of each download
                bar = pool.imap_unordered(func=self.__update__, iterable=tests, chunksize=chunksize)
                for _ in tqdm.tqdm(bar, total=len(tests), desc=f'Getting {len(tests)} tests in {njobs} parallels'):
                    continue
        return

    def __update__(self,test):
        test.update(n_rows = self.n_rows)
        return
        
        

class Device:
    def __init__(self, repo, id):
        self.id   = id
        self.repo = repo
        self.units = self.__get_units__()
        return

    def __get_units__(self):
        units = []
        unit_id = get_query(self.repo.connection_string, f'SELECT unit_id FROM bts612_chlmap WHERE dev_uid={self.id}').unit_id.unique()
        for id in  unit_id:
            units.append(Unit(self, id))
        return units

class Unit:
    def __init__(self, device, id):
        self.device       = device
        self.id           = id
        self.channels     = self.__get_channels__()
        return
    def __get_channels__(self):
        channels = []
        chl_id   = get_query(self.device.repo.connection_string, f'SELECT chl_map_id FROM bts612_chlmap WHERE dev_uid={self.device.id} AND unit_id={self.id} AND aux_chl_map_id=0').chl_map_id.unique()
        for id in chl_id:
            channels.append(Channel(self, id))
        return channels

class Channel:
    def __init__(self, unit, id):
        self.unit      = unit
        self.id        = id
        self.tests     = self.__get_tests__()
        return
    def __get_tests__(self):
        tests = []
        table = self.unit.device.repo.__table__.query(f'dev_uid=={self.unit.device.id} and unit_id=={self.unit.id} and chl_id=={self.id}') 
        for _,row in table.iterrows():
            tests.append(Test(self, row))
        return tests

class Test:
    def __init__(self, channel, info):
        self.channel  = channel
        self.info     = info
        self.id       = info.test_id
        
        self.downloader = self.channel.unit.device.repo.__downloader__

        #Parser and properties to be parsed
        self.__parser__    = self.channel.unit.device.repo.__testparser__
        self.filename      = self.__parser__.filename_parser(self)
        self.cell_name     = self.__parser__.identify_parser(self)
        self.save_location = self.__parser__.location_parser(self)
        self.writer        = self.__parser__.writer(self.save_location, self.filename)

        if isinstance(info.end_time, pd._libs.tslibs.nattype.NaTType):
            self.active=True
        else:
            self.active=False

        self.queries = {'main_first': info['main_first_query'],
                        'main_second':info['main_second_query'],
                        'aux_first':  info['aux_first_query'],
                        'aux_second': info['aux_second_query']}
        return

    def parse(self):
        #self.__parser__.parse(self)
        return
        

    def download(self, n_rows=None):
        try:
            sql_rows = self.get_rows()
            logger.info(f'{self.filename}: Downloading {sql_rows} rows')
            data =  self.downloader.get_test(self, from_seq_id=0, to_seq_id=n_rows)
            if data.empty:
                logger.info(f'No data returned for {self.filename}.')
                return
            else:
                self.writer.write(data)
                logger.info(f'{self.filename}: Donwnload finished')
        except Exception as e:
            print(f'Failed {self.filename}')
            print(e)
        return

    def get_rows(self):
        rows = self.downloader.get_seq_id_max(self)
        return max([rows['main_first'], rows['main_second']])        

    def update(self, n_rows=None):
        if not self.writer.exists:
            return self.download(n_rows=n_rows)
        
        try:
            sql_rows = self.get_rows()
            raw_rows = self.writer.read(columns='seq_id')['seq_id'].max()
            
            if sql_rows>raw_rows:
                logger.info(f'{self.filename}: Updating from {raw_rows} to {sql_rows}. {sql_rows-raw_rows} rows total')                
                data = self.downloader.get_test(self, from_seq_id=raw_rows, to_seq_id=n_rows)
                if data.empty:
                    logger.info(f'No data returned for {self.filename}.')
                    return
                else:
                    self.writer.append(data)
                logger.info(f'{self.filename}: Update finished')
            else:
                logger.info(f'{self.filename}:Up to date.')     
        except Exception as e:
            print(f'Failed {self.filename}')
            print(e)

        return   

#============================================================================================



if __name__=='__main__':
    pass