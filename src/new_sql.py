import typing
import numpy as np
import pandas as pd
import sqlalchemy

def get_query(connection_string, query, retry=True, engine=None):
    """
    Engine is created upon every query instance by default to allow parallel execution of queries
    multiprocessing can not pickle an existing connection

    #If None or nan appears in the query, empty dataframe is returned
    Upon error, it makes a second attempt by default before returning empty dataframe upon the second failure.
    """
    if engine is None:
        engine = sqlalchemy.create_engine(connection_string)    

    if ('FROM None' in query) | ('FROM nan' in query):
        data = pd.DataFrame()
    
    else:
        try:
            data = pd.read_sql(query, engine, chunksize=None)
        except Exception as e:
            if retry is True:
                data = get_query(connection_string, query, retry=False)
            else:
                data = pd.DataFrame()
    return data  


def get_sql_engine(connection_string):
    return sqlalchemy.create_engine(connection_string)    


def get_table(connection_string):
    
    #Get test notes
    notes = get_query(connection_string, "SELECT * FROM test_note")
    
    #Get tests and h_tests
    tests = pd.concat([
        get_query(connection_string, "SELECT * FROM test"),
        get_query(connection_string, "SELECT * FROM h_test")], ignore_index=True)
    
    #Remove duplicates
    tests.drop_duplicates(inplace=True)

    #Those with no registered end-time are considered active
    tests['active'] = False
    tests.loc[pd.isnull(tests.end_time), 'active'] = True

    #Return merged tests
    table = tests.merge(notes, on=['dev_uid', 'unit_id', 'chl_id', 'test_id'])
    table.loc[:,'main_first_query']  = 'SELECT * FROM ' + table['main_first_table'].astype(str)  + ' WHERE test_id=' + table['test_id'].astype(str) + ' AND unit_id=' + table['unit_id'].astype(str) + ' AND chl_id=' + table['chl_id'].astype(str)
    table.loc[:,'main_second_query'] = 'SELECT * FROM ' + table['main_second_table'].astype(str) + ' WHERE test_id=' + table['test_id'].astype(str) + ' AND unit_id=' + table['unit_id'].astype(str) + ' AND chl_id=' + table['chl_id'].astype(str)
    table.loc[:,'aux_first_query']   = 'SELECT * FROM ' + table['aux_first_table'].astype(str)   + ' WHERE test_id=' + table['test_id'].astype(str) + ' AND unit_id=' + table['unit_id'].astype(str) + ' AND chl_id=' + table['chl_id'].astype(str)
    table.loc[:,'aux_second_query']  = 'SELECT * FROM ' + table['aux_second_table'].astype(str)  + ' WHERE test_id=' + table['test_id'].astype(str) + ' AND unit_id=' + table['unit_id'].astype(str) + ' AND chl_id=' + table['chl_id'].astype(str)   
    return table


def download_test(connection_string:str, test:pd.Series, from_seq_id:typing.Union[int,type(None)]=None, to_seq_id:typing.Union[int,type(None)]=None):
    #to_seq_id is good for development purposes. Download only first 100 rows for instance
    if from_seq_id is None:
        from_seq_id = 0

    seq_id_str = f' AND seq_id > {from_seq_id}'
    if isinstance(to_seq_id, int):
        seq_id_str = seq_id_str + f' AND seq_id < {to_seq_id}'

    data_main_first  = get_query(connection_string, test['main_first_query']  + seq_id_str)
    data_main_second = get_query(connection_string, test['main_second_query'] + seq_id_str)
    data_aux_first   = get_query(connection_string, test['aux_first_query']   + seq_id_str)
    data_aux_second  = get_query(connection_string, test['aux_second_query']  + seq_id_str)     

    #Insert temperatures in first table if main is not empty
    if not data_main_first.empty:
        if (data_aux_first.empty) | (len(data_main_first.index) != len(data_aux_first.index)):
            data_main_first['test_tmp'] = np.nan #missing

        elif len(data_main_first.index) == len(data_aux_first.index):
            data_main_first['test_tmp'] = data_aux_first['test_tmp']

    #Insert temperatures in second table if main is not empty
    if not data_main_second.empty:
        if (data_aux_second.empty) | (len(data_main_second.index) != len(data_aux_second.index)):
            data_main_second['test_tmp'] = np.nan #missing

        elif len(data_main_second.index) == len(data_aux_second.index):
            data_main_second['test_tmp'] = data_aux_second['test_tmp']
    
    if data_main_second.empty:
        data = data_main_first.copy()
    else:
        data = pd.concat([data_main_first,data_main_second], ignore_index=True)

    if not data.empty:
        data.loc[:, 'dev_uid'] = test.dev_uid

    return data        
