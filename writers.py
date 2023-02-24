import pathlib
import pandas as pd

def writer(format):
    formats = {
        'parquet':ParquetWriter,
        'hdf':HDFWriter,
        'csv':CSVWriter,
        'feather':FeatherWriter,
        'excel':ExcelWriter,
        'latex':LaTeXWriter
    }
    return formats[format]


class BaseWriter:
    def __init__(self, root, name, extension):
        """
        Make sure root as a pathlib.Path oject
        """
        if isinstance(root, str):
            root = pathlib.Path(root).resolve()
        self.root = root
        self.name = name
        
        #Create /root/name.extension object
        if not '.' in extension:
            extension = f'.{extension}'
        self.path   = self.root.joinpath(name).with_suffix(extension)
        return
    
    @property
    def exists(self):
        return self.path.exists()

    def makeroot(self):
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)
        return

    def delete(self):
        if self.exists:
            self.path.unlink()
        return

class ParquetWriter(BaseWriter):
    def __init__(self, root, name):
        super().__init__(root, name, '.parquet')
        return

    def write(self, data):
        self.makeroot()
        data.to_parquet(self.path)
        return
    
    def append(self, data):
        self.write(pd.concat([self.read(), data], axis=0, ignore_index=True))
        return
    
    def read(self, columns=None):
        if isinstance(columns, str):
            columns=[columns]
        return pd.read_parquet(self.path, columns=columns)


class CSVWriter(BaseWriter):
    def __init__(self, root, name):
        super().__init__(root, name, '.csv')
        return

    def write(self, data):
        self.makeroot()
        data.to_csv(self.path, mode='w', header=True)
        return
    
    def append(self, data):
        if not self.exists:
            self.write(data)
        else:
            data.to_csv(self.path, mode='a', header=False)
        return
    
    def read(self, columns=None):
        if isinstance(columns, str):
            columns=[columns]        
        return pd.read_csv(self.path, usecols=columns)

class HDFWriter(BaseWriter):
    def __init__(self, root, name):
        super().__init__(root, name, '.hdf')
        return

    def write(self, data, key='data'):
        self.makeroot()
        data.to_hdf(self.path, key, mode='w', append=False, format='table', index=None, data_columns=None, complevel=9)
        return
    
    def append(self, data, key='data'):
        if not self.exists:
            self.write(data)
        else:
            data.to_hdf(self.path, key, mode='r+', append=True, format='table', index=None, data_columns=None, complevel=9)
        return
    
    def read(self, columns=None):
        if isinstance(columns, str):
            columns=[columns]        
        return pd.read_hdf(self.path, columns=columns)
    
class FeatherWriter(BaseWriter):
    def __init__(self, root, name):
        super().__init__(root, name, '.feather')
        return

    def write(self, data):
        self.makeroot()
        data.to_feather(self.path)
        return
    
    def append(self, data):
        self.write(pd.concat([self.read(), data], axis=0, ignore_index=True))
        return
    
    def read(self, columns=None):
        if isinstance(columns, str):
            columns=[columns]
        return pd.read_feather(self.path, columns=columns)

class ExcelWriter(BaseWriter):
    def __init__(self, root, name):
        super().__init__(root, name, '.xlsx')
        return

    def write(self, data):
        self.makeroot()
        data.to_excel(self.path, sheet_name='data')
        return
    
    def append(self, data):
        self.write(pd.concat([self.read(), data], axis=0, ignore_index=True))
        return
    
    def read(self, columns=None):
        if isinstance(columns, str):
            columns=[columns]
        return pd.read_excel(self.path, sheet_name='data', columns=columns)

class LaTeXWriter(BaseWriter):
    def __init__(self, root, name):
        super().__init__(root, name, '.tex')
        return

    def write(self, data):
        self.makeroot()
        with open(self.path, 'w') as file:
            file.write(data.to_latex(index=False))
        return
    
    def append(self, data):
        self.write(pd.concat([self.read(), data], axis=0, ignore_index=True))
        return
    
    def read(self, columns=None):
        if isinstance(columns, str):
            columns=[columns]
        data = pd.read_csv(self.path, skiprows=2, skipfooter=2, delimiter='&', engine='python').drop(0, axis=0)
        data.columns = [col.replace('\\', '').strip() for col in data.columns]

        if isinstance(columns, list):
            data = data.loc[:, columns]
        return data

if  __name__=='__main__':
    pass