# Introduction 
The neware module has a capability of downloading data from the SQL database used in the background by Neware. The module updates tests with the latest data on each run, i.e. no downloading of large files multiple times.

# Getting Started
```
import newaresql
db = newaresql.NewareDB(connection_string, file_format='parquet', file_name=None, file_path=None, download=None)
db.download(update=True, parallel=False) #parallel=True or int has the risk of Malloc Error
```
* User input on initiation
    * connection_string on the format dialect+driver://username:password@host:port/database
      * sqlite://user:password@your_server_ip:port/db_name
      * mysql+pymysql://user:password@your_server_ip:port/db_name
    * file_format: parquet, hdf, csv, excel, feather, html and latex currently supported. 
    * file_name: None (default) or callable with signature f(test). Must return a string. 
    * save_path: None (default) string, or callable with signature f(test). Must return a string or pathlib.Path object. Data is saved at this location
    * download: None (default) or callable with signature f(test). Return True or False to check if each specific test should be downloaded
* User input on download():
    * update: True or False. True will add only latest data. False will download the entire test
    * parallel: Run sequential or asyncrounus in parallel. 

# Do to
 - [ ] Test class should interact with test ad datafile both ways
 - [ ] Add plot methods to test class
 - [ ] Add backup method to NewareDB class. Download all tables into a single directory
 - [ ] Add summarize to step method
 - [ ] Make GUI interface
 - [ ] Refactor the sql connection to not initiate a new engine on every query

TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
