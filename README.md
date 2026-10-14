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


## Getting started
With context management
```
import newaresql

# newaresql will look for os.getenv(NEWARE_BTS_HOST), etc if credential is not supplied
credentials = dict(host=..., port=..., username=..., password=..., database=...)

with newaresql.connect(**credentials): as connection:
    tests = connection.list_tests()
    data = connection.get_data(tests[0])
```

Or without
```
import newaresql

# newaresql will look for os.getenv(NEWARE_BTS_HOST), etc if credential is not supplied
credentials = dict(host=..., port=..., username=..., password=..., database=...)

tests = newaresql.list_tests(**credentials)
data = newaresql.get_data(tests[0], **credentials)
```

# Contributions needed
- BTS build versions and device types. `newaresql` currently supports BTS build 0760 (device type 24) and 0800 (device type 24 and 26). 
- Testing. Does it work for you? 
  
# Code layout
The `Connector` class in `connector.py`  uses a SQLAlchemy's engine to handle connections and `polars.read_database` to execute queries.  
The connector exposes methods
 - `query(query,...)`: Passes a query along to pl.read_database 
 - `list_tests()`: Returns a list of availalbe tests as dictionaries
 - `get_tests()`: Returns a dataframe overview of availalbe tests
 -  `get_main(test, ...)`: Fetches the `main` data, *i.e.* time-current-voltage derived data. 
 -  `get_aux(test, ...)`: Fetches the `aux` data, *i.e.* temperature data.   
 -  `get_data(test, ...)`: Fetches `main` and `aux` data merged.
 -  `get_stats(test)`: Returns some test statistics as a dictionary.  

Data transformations and schemas for different BTS build versions and device types are implemented der `transformations/` and `schemas/`.  
Conversion between Neware column names and BDF labels and machine codes are implemented in `bdf.py`.

# TODO/Future: 
- [ ] Move some functionality into namespace newaresql.bts - IF we extend the package with more functionality.
- [ ] Add step- and cycle aggregation.
- [ ] Add clone support in newaresql.clone
- [ ] Add export support in newaresql.export.
- [ ] Add explorative dashboard. 