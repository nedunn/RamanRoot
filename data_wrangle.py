import pandas as pd


class Data_DF:
    def __init__(self, dataframe, name_dict, column_name='index'):
        """
        Initialize the DataFrameSelector.

        Parameters:
        - dataframe: pd.DataFrame
            The DataFrame to be processed. Columns are Raman shifts, indexes are identifiers for each spectra, and are keys in the name_dict.
        - selection_dict: dict
            A dictionary where keys are index values, and values are the row names.
        - column_name: str, optional
            The column name to use for selection. Default is 'index'.
        """
        self.df = dataframe
        self.name_dict = name_dict
        self.col_name = column_name
   
    def select_rows(self, select_vals):
        """
        Select rows from the DataFrame where the specified column values match any value in `select_vals`.
        
        Parameters:
        - select_vals: Union[str, List[str]]
            The value or list of values to select. If a single value is provided, it will be converted to a list internally.

        Returns:
        - pd.DataFrame
            The sub-DataFrame containing selected rows.
        """
        selected_indices = [key for key, value in self.name_dict.items() if value in select_vals]

        if not selected_indices:
            print("No indices found with given values.")
            return pd.DataFrame()

        sub_dataframe = self.df[self.df[self.col_name].isin(selected_indices)]
        return sub_dataframe

if __name__ == '__main__':
    #Example usage when running the module directly
    
    # Create a DataFrame
    data = {'index': [1, 2, 3, 4, 5],
            'value': ['a', 'b', 'c', 'd', 'e']}
    df = pd.DataFrame(data)

    # Create a selection dictionary
    selection_dict = {1: 'select', 2: 'not_select', 3: 'select', 4: 'not_select', 5: 'select'}

    # Create an instance of DataFrameSelector
    selector = Data_DF(dataframe=df, name_dict=selection_dict, column_name='index')

    # Get the sub-DataFrame with selected rows
    selected_rows = selector.select_rows('select')

    # Display the result
    print(selected_rows)
