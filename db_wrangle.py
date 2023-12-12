import psycopg2
import pandas as pd
import pw

class DB_Manager:
    def __init__(self, password, dbname, user='postgres', host='localhost', port='5432'):

        #Establish a connection to a PostgreSQL database and create a cursor object     
        self.db_params={
            'host':host,
            'port':port,
            'database':dbname,
            'user':user,
            'password':password
        }
        self._connect()

        self.tables=self._table_list()


    def _connect(self):
        self.conn = psycopg2.connect(**self.db_params)
        self.cur = self.conn.cursor()
    
    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

    def exe_query(self, query, params=None, fetch_all=True):
        try:
            self.cur.execute(query, params)
            self.conn.commit()
            return self.cur.fetchall() if fetch_all else self.cur
        except Exception as e:
            self.conn.rollback()
            print(f'Error at query execution: {e}')
            return []

    def _table_list(self): # this is an example of an API (but it is missing a route)
        stmt = 'SELECT table_name FROM information_schema.tables WHERE table_schema=\'public\' ORDER BY table_name;'
        
        # ALT: self.cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        return [table[0] for table in self.exe_query(stmt)]
    
    def table_info(self, table_name):
        columns_query = f"SELECT column_name FROM information_schema.columns WHERE table_name = %s;"
        columns = [column[0] for column in self.exe_query(columns_query, (table_name,))]

        rows_query = f"SELECT COUNT(*) FROM {table_name};"
        num_rows = self.exe_query(rows_query)[0][0]

        stats = {'columns': columns, 'num_rows': num_rows, 'column_stats': {}}
        #self.cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'")

class Table_Manager(DB_Manager):
    def __init__(self, table_name, password, dbname, 
                 user='postgres', host='localhost', port='5432',
                 condition=None,
                 collection_method_table=None, method_key='method_id'):
        super().__init__(password, dbname, user, host, port)
        
        self.table=table_name
        self.condition=condition
        self.collection_table=collection_method_table
        self.key=method_key


    def col_list(self):
        stmt = f"SELECT column_name FROM information_schema.columns WHERE table_name = %s;"
        res=self.exe_query(stmt, (self.table,))
        return [tup[0] for tup in res]
    
    def add_row(self, data_dict):
        try:
            columns=', '.join(data_dict.keys())
            value_placeholder=', '.join('%s' for _ in data_dict.values())
            stmt=f'INSERT INTO {self.table} ({columns}) VALUES ({value_placeholder})'
            self.exe_query(stmt, tuple(data_dict.values()), fetch_all=False)
        except Exception as e:
            print(f'Error adding row: {e}')
    
    def remove_row(self, condition_column, condition_value):
        #future, use dictionary to select multipel columns/values
        try:
            # Find rows that will be removed
            count_stmt=f'SELECT COUNT(*) FROM {self.table} WHERE {condition_column} = %s;'
            self.cur.execute(count_stmt, (condition_value,))

            # Display info and ask for confirmation
            print(f'Removing {self.cur.fetchone()[0]} row(s) where {condition_column} = {condition_value}')
            confirm = input("Are you sure you want to proceed? (yes/no): ").strip().lower()

            if confirm == 'yes': # Proceed to delete row(s)
                stmt=f'DELETE FROM {self.table} WHERE {condition_column} = %s;'
                self.exe_query(stmt, (condition_value,), fetch_all=False)
                self.conn.commit()
                print('Row(s) removed.')
            else:
                print('Deletion cancelled.')
        except Exception as e:
            print(f'Error removing row: {e}')
    
    def grab_data(self, columns=None):
        if columns is None: # If columns is not specified, select all columns
            column_names = '*'
        else: # If columns are specified, join them into a comma-separated string
            column_names=', '.join(columns)
        
        # Construct query statement
        stmt=f'SELECT {column_names} FROM {self.table};'
        return self.exe_query(stmt)
    
    def grab_data_advanced(self, time_var, where_stmt=None, columns=None):
        if columns is None: # If columns is not specified, select all columns
                column_names = '*'
        else: # If columns are specified, join them into a comma-separated string
            column_names=', '.join(columns)

        # Construct query statement
        stmt=f'SELECT {column_names} FROM {self.table} '
        stmt+=f'JOIN collection_method ON {self.table}.method_id=collection_method.method_id '
        stmt+=f'WHERE collection_method.time = %s' #\'{time_var}\'
        print(stmt)

        return self.exe_query(stmt, (time_var,))
    def grab_data_advanced(self, time_var, where_dict=None, columns=None):
        if not self.collection_table:
            print('Collection method table needs to be set.')
            return tuple()
        
        if columns is None: # If columns is not specified, select all columns
            column_names = '*'
        else: # If columns are specified, join them into a comma-separated string
            columns=[f'{self.table}.{col}' for col in columns] # add table to column names
            column_names=', '.join(columns)

        # Construct query statement
        stmt=f'SELECT {column_names} FROM {self.table} '
        stmt+=f'JOIN {self.collection_table} ON {self.table}.{self.key} = {self.collection_table}.{self.key} '
        stmt+=f'WHERE {self.collection_table}.time = %s' #\'{time_var}\' #TODO add functionality  
        if where_dict:
            for key,val in where_dict.items():
                stmt+=f' AND {self.table}.{key} {val}'
        
        return self.exe_query(stmt, (time_var,))
