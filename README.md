# Neware SQL
Read data directly from the MySQL database. 

# Quick start
The package is not yet in PyPi, so you will need to clone the repository and pip install it from source. \
Connecting the the MySQL database requires
  - host
  - port
  - user
  - password
  - database

The actual connection is handled by an SQLAlchemy engine.
Credentials must be provided explicitly, or made availabe as enviroment variables on the form `BTS_HOST`, `BTS_PORT`, *etc*.  


## Simple example
The preffered way is the establish a re-useable connection ... 

```
import newaresql.bts as bts

credentials = dict(host=..., port=..., username=..., password=..., database=...)

with bts.connect(**credentials): as connection:
    tests = newaresql.list_tests(connection=connection)
    data = newaresql.get_data(tests[0], connection=connection)
```

but credentials can also be used-directly
```
import newaresql

credentials = dict(host=..., port=..., user=..., password=..., database=...)

tests = newaresql.list_tests(credentials=credentials)
data = newaresql.get_data(tests[0], credentials=credentials)
```

# Contributions needed
- BTS build versions and device types. `newaresql` currently supports BTS build 0760 (device type 24) and 0800 (device type 24 and 26). 
- Testing. Does it work for you? 
  
# Code layout
Database connectivity is implemented in `connect.py`, using SQLAlchemy's connection engine and `polars.read_database` to execute most queries.\
In our experience, main- and auxillary data merge can be *excessively*  slow on the server side.\
The connector therefore implements `get_main_data()` and `get_aux_data`, and `strean_main_data()` and `stream_aux_data` separately. The auxillary data table can also be twice the height of the main data tables, as is the case for type 26 devices with 2 auxillary channels. 

Schemas of main- and aux data, *i.e* dictionaries of column names to python type, for different BTS-builds and device types are registered under `\schemas\`, and is used by the connector to ensure consistend data types when fetching the raw data.\
Data transformations from Neware's integer values to actual measurements is implemented in `transform.py` for different BTS-server builds and device types.
Conversion between Neware column names and BDF labels and machine codes are implemented in `bdf.py`.

# TO-DO
- Implement the SQL- and Blob sinks for clone
- Implement the SQL- and Blob sinks for export
- Separate main- and aux data export. 
- Decide on metadata vs tests
- Add a dashboard