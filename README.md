Updated 10/13/2023

# RamanRoot
Sharma Lab Python code for Raman spectra management, processing, visualization, and analysis.

----
----

# DB_Wrangler Module
Provides a simple interface for managing **PostgreSQL** databases using the psycopg2 library. It currently includes two classes: `DB_Manager` and `Table_Manager`.

## `DB_Manager` Class
### Initialization
```python
db_manager = DB_Manager(dbname='your_database', user='your_username', password='your_password', host='your_host', port='your_port')
```
### Methods
- `connect()` - Establishes a connection to the database.
- `disconnect()` - Closes connection.
- `exe_query (query, params=None, fetch_all=True)` - Executes a SQL query with optional parameters. Returns the fetched results optionally.
- `table_list()` - Returns a list of table names in the public schema.
- `table_stats(table_name)` - WIP to return information about the given table.

## Table_Manager Class ()
### Initialization
```python
table_manager = Table_Manager(table_name='your_table', dbname='your_database', user='your_username', password='your_password', host='your_host', port='your_port')
```
### Methods
- `col_list()` - Returns a list of column names in the table.
- `add_row(data_dict)` - Adds a new row to the table using a dictionary of column-value pairs.
- `remove_row(condition_column, condition_value)` - Removes row(s) from the table based on a specified condtion. Confirmational prompt is required.
- `grab_data(columns=None)` - Retrieves data from the table. Specific columns can be given as a list.
