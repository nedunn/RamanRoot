import pandas as pd
from typing import Dict, List, Union


class Data_DF:
    def __init__(self, dataframe, name_dict):
        """
        Initialize the DataFrameSelector.

        Parameters:
        - dataframe: pd.DataFrame
            The DataFrame to be processed. Columns are Raman shifts, indexes are identifiers for each spectra, and are keys in the name_dict.
        - selection_dict: dict
            A dictionary where keys are index values, and values are the row names.
   
        """
        self.df = dataframe
        self.name_dict = name_dict

   
    def get_subdf(self, select_vals):
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
        print (selected_indices)

        if not selected_indices:
            print("No indices found with given values.")
            return pd.DataFrame()

        sub_dataframe = self.df[self.df.index.isin(selected_indices)]
        return sub_dataframe
    
    def get_name_counts(self, get_counts=False):
        """
            Get unique values and their corresponding counts from the name_dict.

            Parameters:
            - get_counts: bool, optional
                If True, return a dictionary with unique values as keys and their counts as values.
                If False (default), return a list of unique values.

            Returns:
            - Union[List[str], Dict[str, int]]
                If get_counts is False, returns a list of unique values.
                If get_counts is True, returns a dictionary with unique values as keys and their counts as values.
            """
        value_counts = {}
        for value in self.name_dict.values():
            value_counts[value] = value_counts.get(value, 0) + 1
        
        if get_counts:
            return value_counts
        else:
            return list(value_counts.keys())
    
    
if __name__ == '__main__':
    #Example usage when running the module directly
    
    # Create a DataFrame
    data = {'index': [1, 2, 3, 4, 5],
            'value': ['a', 'b', 'c', 'd', 'e']}
    df = pd.DataFrame(data)
    df.set_index('index', inplace=True, drop=True)

    # Create a selection dictionary
    selection_dict = {1: 'select', 2: 'not_select', 3: 'select', 4: 'not_select', 5: 'select'}

    # Create an instance of DataFrameSelector
    selector = Data_DF(dataframe=df, name_dict=selection_dict)

    # Get the sub-DataFrame with selected rows for multiple values
    selected_rows = selector.get_subdf(['select', 'not_select'])
    print(selected_rows)# Display the result

    # Get unique values from the name_dict
    unique_values = selector.get_name_counts(get_counts=False)
    print("Unique Values:", unique_values)