import psycopg2

class DB_Manager:
    def __init__(self, password, dbname, user='postgres', host='localhost', port='5432'):
        
        self.db_params={
            'host':host,
            'port':port,
            'database':dbname,
            'user':user,
            'password':password
        }

        self.conn=None
        self.cur=None

        self.connect()

    def connect(self):
        self.conn = psycopg2.connect(**self.db_params)
        self.cur = self.conn.cursor()
    
    def disconnect(self):
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

    def table_list(self):
        stmt = 'SELECT table_name FROM information_schema.tables WHERE table_schema=\'public\' ORDER BY table_name;'
        return [table[0] for table in self.exe_query(stmt)]
    
    def table_stats(self, table_name):
        columns_query = f"SELECT column_name FROM information_schema.columns WHERE table_name = %s;"
        columns = [column[0] for column in self.exe_query(columns_query, (table_name,))]

        rows_query = f"SELECT COUNT(*) FROM {table_name};"
        num_rows = self.exe_query(rows_query)[0][0]

        stats = {'columns': columns, 'num_rows': num_rows, 'column_stats': {}}

    def table_dets(self, table):
        pass

class Table_Manager(DB_Manager):
    def __init__(self, table_name, password, dbname, user='postgres', host='localhost', port='5432'):
        super().__init__(password, dbname, user, host, port)
        self.table=table_name

    def col_list(self):
        stmt = f"SELECT column_name FROM information_schema.columns WHERE table_name = %s;"
        return self.exe_query(stmt, (self.table,))
    
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