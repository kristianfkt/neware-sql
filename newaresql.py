import pathlib
import pandas as pd

from src import new_sql as sql
from src import writers
from src import transformers

class Test:
    def __init__(self):
        return

    def download(self):
        data = transformers.transform(
            sql.download_test(self.connection_string, self.test))
        self.writer.write(data)
        return




class Repo:
    def __init__(self, 
            connection_string:str,
            file_format:typing.Union[str,callable]='parquet',
            file_path:typing.Union[str, callable, None]=None,
            file_name:typing.Union[callable, None]=None
            ):

        self.connection_string = connection_string
        self.file_format = file_format
        

        if file_path is None:
            file_path = pathlib.Path(__file__).resolve().parent.joinpath('data')
        
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        
        if isinstance(file_path, pathlib.Path):
            self.__filepath__ = file_path
            get_filepath = self.__get_filepath
        elif callable(file_path):
            self.__filepath__ = None
            get_filepath = file_path

        self.get_filepath = get_filepath        

        
        if file_name is None:
            get_filename = self.__get_filename
        elif callable(file_name):
            get_filename = file_name

        self.get_filename = get_filename

        self.table = sql.get_table(self.connection_string)
        self.tests = [test for _,test in self.table.iterrows()]
        return


    def __get_filepath__(self, test):
        return self.__filepath__
    
    def __get_filename__(self, test):
        return f"{test.dev_uid:.0f}-{test.unit_id:.0f}-{test.chl_id:.0f}-{test.test_id:.0f}"

    
