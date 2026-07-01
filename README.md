# Neware SQL
Read data directly from the MySQL database. 
I'm setting version to `0.1.0`, with all that entails. 

# Quick start
Connecting the the MySQL database required
  - host
  - port
  - user
  - password
  - database
newaresql uses sqlalchemy, and will look for credentials in `os.getenv(f'BTA_{key.upper()'})` if not explicitly provided. 



The connector class pulls the raw data. It must be transformed to be usefull. Data can also be extended to include columns such as `step_count`, and renamed according to the Battery Data Format (bdf) 
```
import newaresql as neware

host:str = ...
port:int = ...
user:str = ...
password:str = ...
database:str = ...

with neware.connect(host=host,port=port,user=user,password=password,database=database) as conn:
  tests = conn.get_tests().to_dicts()
  test = tests[0] # or 1, or 2, or ...
  
  main_data =neware.transform_main(
    conn.get_main_data(test),
    conn.version,
    test['dev_uid']
  )
  
  aux_data =neware.transform_aux(
    conn.get_aux_data(test),
    conn.version,
    test['dev_uid']
  )  
  
  all_data = (
      main_data.drop('test_tmp')
      .join(
        aux_data.select('seq_id','auxchl_id','test_tmp'),
        how='left',
        on='seq_id')
      )
    
  data = neware.to_bdf(
    neware.extend_data(all_data)
    )
```

# Design choices
In our experience, main- and auxillary data merge can be *excessively*  slow on the server side. The auxillary data table can also be twice the height of the main data tables, as is the case for type 26 devices.This makes pulling both together in streaming mode messy without a server side join. The connector class therefore queries the main and auxillary data separately. 

The connector class does not transform data, it simply queries it. Maybe we should abstract up a level and have the user interact with a `Test()` class that contains both the test metadata and a reference to the connector. This would make it easy to hide all the non-optional data transformations from the user. 

# Contributions needed
- BTS build versions and device types. `newaresql` currently supports BTS build 0760 (device type 24) and 0800 (device type 24 and 26). 
- Testing. Does it work for you? 
  
# Consideration
- We try to limit the table operations to `SELECT` and `UNION ALL` statements.
- We use the term `stream` for chunked queries because we (author) like the sound of it. Might be replaced with `chunk` if we add some actual streaming.  
- Main and auxillary data are queried separately. 
  - Auxillary data table may be twice the size of the main data table when multiple auxillary channels are supported. This throws a wrench in the support for `stream_data()`, as the chunks returned will not be syncronized.

# Future features
- Set up support for actual streaming of test data. 
- Consider mapping out which auxillary columns to actually use, for the different server and device versions. 