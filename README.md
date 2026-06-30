# Total rework. 

# Contributions needed
- We currently support BTS build 0760 (device type 24) and 0800 (device type 24 and 26) 
- Connectors to different server versions.
- Data transformations for different device types.
- Bug reports.
- Should main- and auxillary data merge be done server- or client sude? 
  
# Consideration
- We try to limit the table operations to `SELECT` and `UNION ALL` statements.
- We use the term `stream` for chunked queries. 
- Main and auxillary data are queried separately. 
  - Auxillary data table may be twice the size of the main data table when multiple auxillary channels are supported. This throws a wrench in the support for `stream_data()`, as the chunks returned will not be syncronized.

# Future features
- Set up support for actual streaming of test data. 
- Consider mapping out which auxillary columns to actually use, for the different server and device versions. 