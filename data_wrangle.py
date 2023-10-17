import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Union
import plotly.express as px
from preprocess import PreproSpectra

class Data_DF:
    def __init__(self, dataframe, name_dict, group_dict=None, color_dict=None, apply_prepro=True):
        """
        Initialize the DataFrameSelector.

        Parameters:
        - dataframe: pd.DataFrame
            The DataFrame to be processed. Columns are Raman shifts, indexes are identifiers for each spectra, and are keys in the name_dict.
        - selection_dict: dict
            A dictionary where keys are index values, and values are the row names.
   
        """
        self.name_dict = name_dict

        if apply_prepro:
            self.df = self.preprocess_data(dataframe)
        else:
            self.df = dataframe

        if group_dict:
            self.group_dict=group_dict
        else:
            print('No group dictionary provided, creating one based on the name_dict.')
            self.group_dict=self.make_group_dict()
        
        if color_dict:
            self.color_dict=color_dict
        else:
            self.color_dict=self.assign_colors()
            print('No color dictionary provided, default color values will be used.')
   
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

        if not selected_indices:
            print("No indices found with given values.")
            return pd.DataFrame()

        sub_dataframe = self.df[self.df.index.isin(selected_indices)] #could use .contains instead of .isin
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
    
    def make_group_dict(self): #Invert dict function from DataBase/utils.py
        if self.name_dict is None:
            print('No name dictionary provided.')
            return None
        unique_values=set(self.name_dict.values())
        group_dict={value: [key for key, val in self.name_dict.items() if val == value] for value in unique_values}
        self.group_dict=group_dict
        return group_dict

    def assign_colors(self):
        color_list=px.colors.qualitative.Vivid[:len(self.group_dict)] # get a list of colors
        color_dict={key:color_list[i] for i, key in enumerate(self.group_dict.keys())}
        self.color_dict=color_dict
        return color_dict

    def preprocess_data(self, df):
        """
        Apply 'PreproSpectra(row_as_array).get()' for each row of the DataFrame.

        Returns:
        - pd.DataFrame
            The DataFrame with preprocessed data.
        """
        # Initialize new dataframe
        prepro_df=pd.DataFrame(index=df.index, columns=df.columns)

        # Apply Preprocessing
        for index, row in df.iterrows():
            prepro_df.loc[index]=PreproSpectra(row.values).get()
        return prepro_df

    def figure_format(self, input_fig):
        fig = go.Figure(input_fig)
        
        # Add axis labels
        fig.add_annotation(text='Intensity',
                           x=-0.1,y=0.5, xref='paper',yref='paper',
                           showarrow=False,font=dict(size=20, family='Arial'), textangle=-90)
        fig.add_annotation(text='Raman Shift (cm<sup>-1</sup>)',
                           x=0.5,y=-0.2, xref='paper',yref='paper',
                           showarrow=False,font=dict(size=20, family='Arial'))
        
        # Final formatting
        fig.update_layout(template='simple_white', font=dict(family='Arial', size=20), 
                          margin = dict(l=80, r=50, b=80, t=50)) #Set margins for consistency during export
        fig.show()

    def subplot_fig(self, group_order=None):
        # Check for required parameters
        if group_order is None:
            group_order=list(self.group_dict.keys())

        # Create Subplots
        fig=make_subplots(rows=len(group_order), cols=1, shared_xaxes=True, subplot_titles=group_order)
        
        # Create a subplot for each group
        i=1
        for group in group_order:
            if group in self.group_dict:
                indices=self.group_dict[group]
                # Make a subdf that contains the rows from the `indices` list
                sub=self.df.loc[self.df.index.isin(indices)]
                # Add traces to subplot
                for index, row in sub.iterrows():
                    fig.add_trace(go.Scatter(x=self.df.columns, y=row, name=index, line=dict(color=self.color_dict[group]), showlegend=False), row=i, col=1)
                i=i+1

        # Format final figure
        return self.figure_format(fig)

        #Add annotation
        #Optional subplot titles
        #Add title
        # what happens if there are more groups that colors in color_list?
    
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