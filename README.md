# Neware SQL
Read data directly from the MySQL database. 
This first version is set at `0.1.0`.  

# Quick start
The package is not yet in PyPi, so you'll need to clone the repository and pip install it from source. 
Connecting the the MySQL database requires
  - host
  - port
  - user
  - password
  - database

The actual connection is handled by an SQLAlchemy engine.
Credentials must be provided explicitly, or made available through `os.getenv(f'BTS_{cred.upper()}')`. 

## Pre check
``` 
from newaresql.connect import Connector

with Connector(host=.., port=..., user=..., password=..., database=...,) as connector:
  try:
      ver = connector.version
      print(f"Build version {ver} is running on the database")
  except Exception as e:
      print(f"Faile to get build version with error {e}")

  try:
      tables = conn.tables
      print(f"Found the following {len(tables)} tables in the database:")
      print("\n".join(tables))
  except Exception as e:
      print(f"Faile to fetch list of table names with error {e}")
```

## Simple example
```
import newaresql

with newaresql.connect(host=.., port=..., user=..., password=..., database=...,): as connection:
    tests = newaresql.list_tests(connection=connection)
    data = newaresql.get_data(tests[0], connection=connection)
```

or, if you don't like context managers

```
import newaresql as neware



tests = neware.list_tests(credentials=dict(host=.., port=..., user=..., password=..., database=...,))
data = neware.get_data(tests[0], credentials=dict(host=.., port=..., user=..., password=..., database=...,))
```

or, if credentials are set at enviroment variables

```
import newaresql as neware

tests = neware.list_tests()
data = neware.get_data(tests[0])
```

```
import newaresql

with newaresql.connect(): as connection:
    tests = newaresql.list_tests(connection=connection)
    data = newaresql.get_data(tests[0], connection=connection)
```
# Contributions needed
- BTS build versions and device types. `newaresql` currently supports BTS build 0760 (device type 24) and 0800 (device type 24 and 26). 
- Testing. Does it work for you? 
- 
# Code layout
Database connectivity is implemented in `connect.py`, using SQLAlchemy's connection engine and `polars.read_database` to execute most queries.\
In our experience, main- and auxillary data merge can be *excessively*  slow on the server side.\
The connector therefore implements `get_main_data()` and `get_aux_data`, and `strean_main_data()` and `stream_aux_data` separately. The auxillary data table can also be twice the height of the main data tables, as is the case for type 26 devices with 2 auxillary channels. 

Schemas of main- and aux data, *i.e* dictionaries of column names to python type, for different BTS-builds and device types are registered under `\schemas\`, and is used by the connector to ensure consistend data types when fetching the raw data.\
Data transformations from Neware's integer values to actual measurements is implemented in `transform.py` for different BTS-server builds and device types.
Conversion between Neware column names and BDF labels and machine codes are implemented in `bdf.py`.

# TO-DO
- Some columns, such as `test_cur`, requires `cur_step_range` in order to transform BTS-data to actual data. Take this into consideration so that all required columns are queried, where redundant columns are later discarded after use. 
- For step aggregation: Map out column/variable categories, *i.e.* "Current / A" is data column, while 'Step Type / 1' is of some other category.
  - This can be added to the BDF Field level. 
- Add test-statistics summary to the connector class, i.e. min/max seq_id, cycle, *etc*.
- Figure out automatic versioning or something. 
- Generate documentation. 
- Complete docstrings. 
- Add a clone-submodule where polars writes each test to files. Support parquet, csv, feather/ipc and ndjson. 