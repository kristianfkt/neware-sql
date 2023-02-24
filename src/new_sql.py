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
    logger.debug(f'Fetching query {query}')
    if engine is None:
        engine = sqlalchemy.create_engine(connection_string)    

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


def get_table(connection_string):
    
    #Get test notes
    notes = get_query(connection_string, "SELECT * FROM test_note")
    
    #Get tests and h_tests
    tests = pd.concat([
        get_query(connection_string, "SELECT * FROM test"),
        get_query(connection_string, "SELECT * FROM _test")], ignore_index=True)
    
    #Remove duplicates
    tests.drop_duplicates(inplace=True)

    #Those with no registered end-time are considered active
    tests['active'] = False
    tests.loc[pd.isnull(tests.end_time), 'active'] = True

    #Return merged tests   
    return tests.merge(notes, on=['dev_uid', 'unit_id', 'chl_id', 'test_id'])


def download_test(connection_string:str, test:pd.Series, from_seq_id:typing.Union[int,type(None)]=None, to_seq_id:typing.Union[int,type(None)]=None):
    #to_seq_id is good for development purposes. Download only first 100 rows for instance
    if from_seq_id is None:
        from_seq_id = 0

    seq_id_str = f' AND seq_id > {from_seq_id}'
    if isinstance(to_seq_id, int):
        seq_id_str = seq_id_str + f' AND seq_id < {to_seq_id}'

    data_main_first  = get_query(connection_string, test.queries['main_first']  + seq_id_str)
    data_main_second = get_query(connection_string, test.queries['main_second'] + seq_id_str)
    data_aux_first   = get_query(connection_string, test.queries['aux_first']   + seq_id_str)
    data_aux_second  = get_query(connection_string, test.queries['aux_second']  + seq_id_str)     

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

    data = pd.concat([data_main_first,data_main_second], ignore_index=True)
    data.loc[:, 'dev_uid'] = test.dev_uid
    return data        
