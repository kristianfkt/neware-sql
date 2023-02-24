# Introduction 
The neware module has a capability of downloading data from the SQL database used in the background by Neware. The module updates tests with the latest data on each run, i.e. no downloading of large files multiple times.

# Getting Started

```
import neware
repo = neware.sql.Repo(connection_string, save_format='parquet', filename=None, location=None, identify=None, parallel=False, download=None, n_rows=None)
repo.update()
```
* User input
    * connection_string on the format dialect+driver://username:password@host:port/database
    * save_format: parquet, hdf, csv, excel, feather and latex currently supported. 
    * fileame: None (default) or callable with signature func(test). Must return a string. 
    * location: None (default) string, or callable with signature func(test). Must return a string. Data is saved at this location
    * identify: None (default) or callable with signature func(test). Must return a string. Data is saved at /location/identify/ is identify is not None
    * download: None (default) or callable with signature func(test). Dowload all tests, or tests where download(test) returns True. Must return True or False
    * parallel: False (default), int or True. Use asyncronous parallel processing. Not sure if it offers any advantages, as bandwith is likely the limitation
    * n_rows: For testing only. pass an integer to download only the first n_rows of each test


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