import typing
import pathlib
import pandas as pd
import multiprocessing
import tqdm.auto as tqdm

from src import new_sql as sql
from src import writers
from src import transformers

def download_worker(arg):
    test, kwargs = arg
    test.download(**kwargs)
    return
class Test:
    def __init__(self, test, file_format, file_path, file_name, connection_string):
        self.test = test
        self.connection_string= connection_string
        
        if file_path is None:
            pathlib.Path(__file__).resolve().parent.joinpath('data')

        if callable(file_path):
            file_path = file_path(self.test)

        if file_name is None:
            file_name = f"{test.dev_uid:.0f}-{test.unit_id:.0f}-{test.chl_id:.0f}-{test.test_id:.0f}"

        if callable(file_name):
            file_name = file_name(self.test)

        self.writer = writers.get_writer(file_format)(file_path, file_name)
        return

    def download(self, update=True, n_rows=None):
        if update:
            max_seq_id = self.get_max_seq_id()
        
        data = transformers.transform(
            sql.download_test(self.connection_string, self.test, from_seq_id=max_seq_id, to_seq_id=n_rows))
        
        if update:
            self.writer.append(data)
        else:
            self.writer.write(data)
        return

    def get_max_seq_id(self):
        if self.writer.exists:
            max_seq_id = self.writer.read(columns='seq_id').max()
        else:
            max_seq_id = 0 
        return max_seq_id

class Repo:
    def __init__(self, 
            connection_string:str,
            file_format:typing.Union[str, callable]='parquet',
            file_path:typing.Union[str, callable, None]=None,
            file_name:typing.Union[callable, None]=None,
            download:typing.Union[callable,None]=None
            ):

        self.connection_string = connection_string
        self.file_format = file_format

        if file_path is None:
            file_path = pathlib.Path(__file__).resolve().parent.joinpath('data')
        
        self.table = sql.get_table(self.connection_string)
        self.tests = [Test(test, file_format, file_path, file_name, connection_string) for _,test in self.table.iterrows()]
        
        if download is None:
            download = lambda *args:True
        self.check_download = download
        return
    
    def download(self, 
        update:bool=True,
        n_rows:typing.Union[None, int]=None,
        parallel:typing.Union[bool,int]=False
        ):
        download_list = [test for test in self.tests if self.check_download(test.test)]
        download_worker_args = [(test, {'update':update, 'n_rows':n_rows}) for test in download_list]

        if parallel is False:
            for arg in tqdm.tqdm(download_worker_args, desc='Downloading data'):
                download_worker(arg)

        else:
            if isinstance(parallel, int):
                n_jobs = parallel
            else:
                n_jobs = multiprocessing.cpu_count()
            
            with multiprocessing.Pool(n_jobs) as pool:
                bar = pool.imap_unordered(func=download_worker, iterable=download_worker_args)
                for _ in tqdm.tqdm(bar, desc='Downloading data'):
                    continue
        return    
