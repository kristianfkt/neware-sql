# Neware SQL
Read data directly from the MySQL database. 
This first version is set at `0.1.0`.  

# Quick start
Connecting the the MySQL database requires
  - host
  - port
  - user
  - password
  - database

The actual connection is handled by an sqlalchemy engine. Credentials must be provided explicitly, or made availalbe through `os.getenv(f'BTS_{cred.upper()'})`. 




The connector class pulls the raw data. It must be transformed to be usefull. Data can also be extended to include columns such as `step_count`, and renamed according to the Battery Data Format (bdf) 
```
import newaresql as neware

credentials = dict(
  host:str = ...,
  port:int = ...,    
  user:str = ...,
  password:str = ...,
  database:str =...,  
)
with newaresql.connect(**credentials): as connection:
    tests = neware.list_tests(connection=connection)
    data = neware.get_data(tests[0], connection=connection)
```

or

```
import newaresql as neware

credentials = dict(
  host:str = ...,
  port:int = ...,     
  user:str = ...,
  password:str = ...,
  database:str =...,  
)

tests = neware.list_tests(credentials=credentials)
data = neware.get_data(tests[0], credentials=credentials)
```

or

```
import newaresql as neware

tests = neware.list_tests()
data = neware.get_data(tests[0])
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