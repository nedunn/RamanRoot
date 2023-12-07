import pandas as pd
from typing import Dict, List, Union
import numpy as np
import os
import glob

import pybaselines.spline
import scipy.signal as ss

from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class PreproSpectra:
    #FIX add some checks to prevent format errors (array vs series vs list)
    """
    A class for preprocessing a Raman spectra.

    Parameters
    ----------
    original_intensity : list or array-like
        The original intensity values of the spectral data.

    raman_shifts : list or array-like, optional
        The corresponding Raman shift values. If not provided, bogus Raman shift values will be generated.

    alerts : bool, optional
        Flag indicating whether to display alerts generated when initializing instance.

    name : str, optional
        Name of the PreproSpectra instance.

    no_neg : bool, optional
        Flag indicating whether to replace negative intensity values with 0. Default is True.
    """
    def __init__(self, original_intensity, raman_shifts=None,
                 name=None, alerts=False, no_neg=True,
                 snv=False,
                 smooth_window=9, smooth_poly=3, zap_window=2, zap_threshold=5,
                 **params):
        
        self.y=original_intensity
        if raman_shifts is None:
            alert=f'You did not provide Raman Shift values.\n'
            self.x=list(range(0,len(self.y))) #Generate bogus Raman Shift values for X axis
        elif (len(original_intensity) != len(raman_shifts)) == True:
            alert=f'Error: the given `original_intensity` and `raman_shifts` are not of equal length.\n'
            alert+='\tBogus values will be used instead.\n'
            alert+=f'\tLength intensity values: {len(original_intensity)}\n\tLength RS values: {len(raman_shifts)}\n'
            self.x=list(range(0,len(self.y)))
        else:
            alert=''
            self.x=raman_shifts
        self.name=name

        # Smoothing parameters
        self.smooth_window=smooth_window
        self.smooth_poly=smooth_poly
        
        # Zap parameters
        self.zap_window=zap_window
        self.zap_threshold=zap_threshold

        # Apply Zap
        zap, zap_text = self.zap(self.y)
        self.y_zap=zap
        alert+=zap_text
        
        # Apply Smooth
        self.y_zap_smooth=ss.savgol_filter(self.y_zap, self.smooth_window, self.smooth_poly)
        alert+=f'Smoothing has been applied: window = {self.smooth_window}, polynomial = {self.smooth_poly}.\n'

        # Baseline
        self.baseline=pybaselines.spline.pspline_asls(self.y_zap_smooth)[0]
        y_base=self.y_zap_smooth-self.baseline

        # SNV
        if snv:
            Y=self.snv(y_base)
            alert+='SNV normalization applied to spectra.'
        else:
            Y=y_base

        # Return preprocessed data
        if no_neg:
            self.Y=list(map(lambda num: num if num >= 0 else 0, Y))
            alert+='Intensity returned with all negative values replaced with `0`.\n'
        else:
            self.Y=Y
            alert+='Intensity returned with negative values.\n\t**Set `no_neg` to `True` to remove negatives.\n'

        # self.__dict__.update(params)

        # Display alerts
        if alerts:
            print(alert)
 
    def zscore(self,nums):
        """
        Calculate the Z-scores of the given numbers.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.

        Returns
        -------
        zscores : ndarray
            The Z-scores of the input numbers.
        """
        mean=np.mean(nums)
        std=np.std(nums)
        zscores1=(nums-mean)/std
        zscores=np.array(abs(zscores1))
        return(zscores)

    def mod_zscore(self, nums):
        """
        Calculate the modified Z-scores (MAD Z-scores) of the given numbers.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.

        Returns
        -------
        mod_z_scores : ndarray
            The modified Z-scores (MAD Z-scores) of the input numbers.
        """
        median_int = np.median(nums)
        mad_int = np.median([np.abs(nums - median_int)])
        if mad_int != 0:
            mod_z_scores1 = 0.6745 * (nums - median_int) / mad_int
        else: #avoid divide by 0 problem
            mod_z_scores1 = np.zeros_like(nums)
        mod_z_scores = np.array(abs(mod_z_scores1))
        return mod_z_scores
    
    def WhitakerHayes_zscore(self, nums, threshold):
        """
        Whitaker-Hayes Function using Intensity Modified Z-Scores.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.
        threshold : int or float
            The threshold value.

        Returns
        -------
        intensity_modified_zscores : ndarray
            The intensity modified Z-scores.
        """
        dist=0
        delta_intensity=[]
        for i in np.arange(len(nums)-1):
            dist=nums[i+1]-nums[i]
            delta_intensity.append(dist)
        delta_int=np.array(delta_intensity)
        
        #Run the delta_int through MAD Z-Score Function
        intensity_modified_zscores=np.array(np.abs(self.mod_zscore(delta_int)))
        return intensity_modified_zscores
    
    def detect_spikes(self,nums):
        """
        Detect spikes, or sudden, rapid changes in spectral intensity.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.

        Returns
        -------
        spikes : ndarray
            Boolean array indicating whether each value is a spike (True) or not (False).
        """
        spikes=abs(np.array(self.mod_zscore(np.diff(nums))))>self.zap_threshold
        return spikes
    
    def zap(self,nums):
        """
        Replace spike intensity values with the average values that are not spikes in the selected range.

        Parameters
        ----------
        nums : list or array-like
            The input numbers.
        window : int = selected range
            Selection of points around the detected spike.
            Default = 2.
        threshold : int
            Binarization threshold. Increase value will increase spike detection sensitivity. (*I think*)
            Default = 5.

        Returns
        -------
        y_out : list or array-like
            Window Average.
            Average values that are around spikes in the selected range.
        """
        y_out=nums.copy() #Prevents overeyeride of input y
        spikes=abs(np.array(self.mod_zscore(np.diff(nums))))>self.zap_threshold
        try:
            for i in np.arange(len(spikes)):
                if spikes[i] !=0: #If there is a spike detected in position i
                    w=np.arange(i-self.zap_window, i+1+self.zap_window) #Select 2m+1 points around the spike
                    w2=w[spikes[w]==0] #From the interval, choose the ones which are not spikes                
                    if not w2.any(): #Empty array
                        y_out[i]=np.mean(nums[w]) #Average the values that are not spikes in the selected range

                    if w2.any(): #Normal array
                        y_out[i]=np.mean(nums[w2]) #Average the values that are not spikes in the selected range
            return y_out, f'Zap has been applied with threshold = {self.zap_threshold}, window = {self.zap_window}.\n'
        except TypeError:
            return nums, 'Zap step has been skipped.\n'
    
    def remove_peak(self,data, location, window):
        """ 
        FIXME This function has not been integrated into the class
        TODO Add xaxis reference option for easier specification of the peak to be removed
        TODO Loop so that multple locations can be specified

        Remove a peak centered at the given location by averaging values within the specified window.

        Parameters
        ----------
        data : list or array-like
            The input data.
        location : int
            The location (x-axis value) of the peak to be removed.
        window : int
            Half-width of the window in x-axis units.

        Returns
        -------
        result : list or array-like
            Data with the peak removed by averaging surrounding values.

        Example Use
        -----------
        original_data = [10, 12, 15, 18, 22, 30, 25, 20, 17, 16]
        location_of_peak = 5  # Example: peak centered at x=5
        window_size = 3  # Example: window of +/-3 units
        new_data = remove_peak(original_data, location_of_peak, window_size)
        """
        result = data.copy()
        left = max(0, location - window)
        right = min(len(data), location + window + 1)
        if left < right:
            values_around_peak = result[left:right]
            # Calculate the average of values except the peak value itself
            average_value = np.mean(np.delete(values_around_peak, window))
            # Replace the peak value with the average
            result[location] = average_value
        return result

    def snv(self,nums):
        vals=np.array(nums)
        ave=vals.mean() #Calculate the mean for the sample
        centered=vals-ave #Subtract the mean from each intensity value
        std=np.std(centered) #Calculate STD
        res=centered/std#Divide each item in the centered list by the STD
        return res

    def show(self):
        """
        Display the data before and after preprocessing.

        Returns
        -------
        fig: plotly.graphs.Figure
            The figure object containing the plot.
        """
        # Initalize figure
        fig=make_subplots(rows=2, cols=1,
                        shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.05,
                        x_title='Raman Shift (cm-1)', y_title='Intensity')
        #add traces
        fig.append_trace(go.Scatter(x=self.x, y=self.y, name='raw'),row=1,col=1)
        fig.append_trace(go.Scatter(x=self.x, y=self.baseline, name='baseline'),row=1,col=1)
        fig.append_trace(go.Scatter(x=self.x, y=self.Y, name='output'),row=2,col=1)
        
        # Adjust layout
        fig.update_layout(title_text=self.name,title_font_size=15,plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='lightgrey')
        return fig
    
    def __str__(self):
        """
        Return a string representation of the PreproSpectra instance.

        Returns
        -------
        str
            A string representation of the PreproSpectra instance.
        """
        return f"PreproSpectra instance: {self.name if self.name else 'Unnamed'}"

    def __repr__(self) -> list:
        return repr(self.Y)
    
    def get(self):
        return np.array(self.Y)

#TODO 's
#Add annotation
#Optional subplot titles
#Add title
# what happens if there are more groups that colors in color_list?

class DataDir():
    '''A class to grab data from a directory and compile it into a simple dataframe and dictionary to use with other code.'''
    def __init__(self, path, name_select=None, nameselect_include_mode=False, name_drop='.'):
        #TODO edit to include or exclude part of a file name
        #TODO add preprocess step?
        self.path=path
        self.filepathlist=glob.glob(f'{path}/*.csv')
        
        if name_select:
            if nameselect_include_mode:
                self.filelist=[os.path.basename(file) for file in self.filepathlist if name_select in file]
            else:
                self.filelist=[os.path.basename(file) for file in self.filepathlist if name_select not in file]
        else:
                self.filelist=[os.path.basename(file) for file in self.filepathlist]
        
        self.xs=[]
        self.ys=[]
        self.name_drop=name_drop
        self.short_namedict={}
        self.long_namedict={}

        self.x=None
        self.df=None
        self.grab_data()
        self.deal_with_x()
        self.make_dataframe()

    def get(self):
        return self.df, self.short_namedict, self.long_namedict

    def grab_data(self):
        for i, file in enumerate(self.filelist):
            d=pd.read_csv(self.path+file, header=None)

            # X data
            ax=d.iloc[:,0]
            self.xs.append(np.array(ax))

            # Grab Y (+ Preprocess?)
            ay=list(d.iloc[:,1])
            y=pd.Series(ay, name=i)
            self.ys.append(y)

            # Naming dictionary
            self.long_namedict[i]=file.split('.csv')[0]
            self.short_namedict[i]=file.split(self.name_drop)[0]
            

        return self
    
    def deal_with_x(self): #FIX need to check that xs are compatable 1st
        # Stack x arrays (makes a 2D array)
        stack_x=np.stack(self.xs,axis=0)

        # Calculate average (long axis = 0)
        x=np.mean(stack_x, axis=0)
        self.x=np.round(x,2) # fix decimal places

        return self
    
    def make_dataframe(self):
        self.df=pd.DataFrame(self.ys)
        self.df.columns=self.x

    def _display_notice(self,message):
        print(f'Notice from DataDir: {message}')

class Data_DF:
    def __init__(self, dataframe, name_dict=None, xax=None, group_dict=None, color_dict=None, apply_prepro=False,
                 display_notice=False, display_examples=False,
                 ):

        """
        Initialize the DataFrameSelector.


        Parameters:
        - dataframe: pd.DataFrame
            The DataFrame to be processed. Columns are Raman shifts, indexes are identifiers for each spectra, and are keys in the name_dict.
        - name_dict: dict
            A dictionary where keys are index values, and values are the row names.
   
        """
        
        self.df=self._preprocess_data(dataframe) if apply_prepro else dataframe
        #TODO use params to allow for seeting changes during preprocessing

        self.display_notice=display_notice
        self.display_examples=display_examples

        self.name_dict=name_dict if name_dict else self._default_namedict(self.df.index.to_list())
        self.group_dict = group_dict if group_dict else self._default_groupdict(self.name_dict)   
        self.color_dict=color_dict if color_dict else self._default_assign_colordict(list(set(self.name_dict.values())))

        if xax:
            self.xax=xax
        else:
            self._display_notice('Dataframe column values will be used as Raman Shift values.')
            self.xax=np.array(self.df.columns)


        self._display_argument_examples('\'idn\' refers to a unique, numerical value assigned to a spectra.\n\tIn this code ***\'idn\' = dataframe index.***')
        self._display_argument_examples('DataFrames used as inputs should be formated where each ROW = SPECTRA and each COLUMN = RAMAN SHIFT.')

    def get_df(self):
        return self.df

    def get_subdf(self, select_vals): #TODO select included OR excluded option
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
    
    def _default_groupdict(self, dic): #Invert dict function from DataBase/utils.py
        unique_values=set(dic.values())
        group_dict={value: [key for key, val in dic.items() if val == value] for value in unique_values}
        # self.group_dict=group_dict
        self._display_notice('Group_dict not provided. Name_dict values will be used to form groups.')
        self._display_argument_examples('group_dict = {\'Group Name / Identifier\':\'[idns]\'}\n\tDefault made by clustering samples that have the same name_dict value')
        return group_dict
    
    def _default_namedict(self, lst):
        '''Makes a dictionary from a list where key=value. Useful for developing functions that will mainly expect dictionary-df input pairs where dictionary names df rows based on index.
        EX:
        lst = [1,2,3]
        d={ 1:'1', 2:'2', 3:'3'}
        '''
        d={} #initialize dictionary
        for item in lst:
            d[item]=str(item)
        self._display_notice('Name_dict not given. Default values will be assigned.')
        self._display_argument_examples('name_dict={idn:\'name / label / sample\'}')
        return d
    
    def _default_assign_colordict(self, lst):
        dic={}
        for i, key in enumerate(lst):
            if i< 11:
                dic[key]=px.colors.qualitative.Vivid[i]
            else:
                dic[key]=px.colors.qualitative.Alphabet[i]
            
        self._display_notice('No color dictionary provided, default color values will be used.')
        self._display_argument_examples('color_dict = {\'name\':\'color\'}\n\t\'name\' = name_dict key')
        return dic

    def _preprocess_data(self, df):
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
        
        # self._display_notice('Preprocessing has been applied to the data.')

        return prepro_df
    
    def _display_notice(self,message):
        if self.display_notice:
        # Display a message to the user
            print(f'Notice: {message}')
        else:
            return None

    def _display_argument_examples(self,message):
        if self.display_examples:
            print(f'{message}')
        else:
            pass

class Multivar(Data_DF): #TODO merge from df_analysis
    '''Input dataframe is expected to have Raman shift as columns and rows are samples.
    The index values for each row should be the sampleID.
    Dictionaries can be used to set the labels for hovertext, color, and symbol.''' 
    def __init__(self, dataframe, name_dict=None, xax=None, group_dict=None, color_dict=None, apply_prepro=False, 
                 display_notice=False, display_examples=False):
        super().__init__(dataframe, name_dict, xax, group_dict, color_dict, apply_prepro, display_notice, display_examples)

    def _label_gen(self,prefix,n):
        nums=list(range(1,n+1))
        res=['%s %s'%(prefix, num) for num in nums]
        return res

    def pca_run(self, ncomp=10):
        pca=decomposition.PCA(n_components=ncomp)
        
        # Apply PCA
        X=scale(self.df.T) # Rotate DF so that each column is a different sample and RS are indexs
        X=pca.fit_transform(X)
        
        # Get components
        pc_labels=self._label_gen('PC',ncomp)
        pcs=pca.components_.T*np.sqrt(pca.explained_variance_)
        pcs=pd.DataFrame(pcs, columns=pc_labels, index=self.df.index.to_list())

        # Get explained variance
        explained_list=[round(x*100,2) for x in pca.explained_variance_ratio_]
        explained={pc_labels[i]:explained_list[i] for i in range(ncomp)}

        # Get loadings
        loadings=pd.DataFrame(X,columns=pc_labels,index=self.xax)

        # Store the PCA results as an attribute
        self.pca=(pcs, loadings, explained, ncomp)

        # Return self to allow chaining methods
        return self

class SpecVis(Data_DF): #TODO check if the change of xax in Data_df (frome columns to np.array impacts functionality of the child class)
    #TODO streamline with Data_DF
    def __init__(self, dataframe, name_dict, xax=None, group_dict=None, color_dict=None,
                 apply_prepro=False,
                 display_notice=False, display_examples=False):
        super().__init__(dataframe, name_dict, xax, group_dict, color_dict, apply_prepro,display_notice=display_examples)

        self.vertical_spacing=0.05

    def _figure_format(self, input_fig, n_plots):
        fig = go.Figure(input_fig)
        
        # Add axis labels
        fig.add_annotation(text='Intensity',
                           x=-0.17,y=0.5, xref='paper',yref='paper',
                           showarrow=False,font=dict(size=20, family='Arial'), textangle=-90)
        fig.add_annotation(text='Raman Shift (cm<sup>-1</sup>)',
                           x=0.5,y=-0.1, xref='paper',yref='paper',
                           showarrow=False,font=dict(size=20, family='Arial'))
        
        # Dimensions
        height=80*n_plots
        width=1000
        inital_margin = dict(l=110, r=70, b=70, t=20)

        # Calculate proportional dimensions
        width_ratio=width/800
        height_ratio=height/600
        adjusted_row_spacing=0#self.vertical_spacing * height_ratio # Adjusted for vertical spacing
        # adjusted_row_spacing=(self.vertical_spacing * (n_plots+2))
        margin=dict(l=inital_margin['l']*width_ratio,r=inital_margin['r']*width_ratio,
                    t=inital_margin['t']*height_ratio+adjusted_row_spacing,b=inital_margin['b']*height_ratio+adjusted_row_spacing)

        # Final formatting
        fig.update_layout(template='simple_white', font=dict(family='Arial', size=20),
                          height=height, width=width,
                          margin = margin
                          ) #Set margins for consistency during export
        

        # return self.adjust_figure_size_with_spacing(1000,380*n_plots)
        return fig

    def _create_subplots(self,group_order):
        # Check for required parameters
        if group_order is None:
            group_order=list(self.group_dict.keys())

        # Create Subplots
        fig=make_subplots(rows=len(group_order), cols=1, shared_xaxes=True, subplot_titles=group_order,
                          vertical_spacing=self.vertical_spacing
                          )

        # Create a subplot for each group        
        for i,group in enumerate(group_order):
            if group in self.group_dict:
                indices=self.group_dict[group]
                # Make a subdf that contains the rows from the `indices` list
                sub=self.df.loc[self.df.index.isin(indices)]
                # Add traces to subplot
                for index, row in sub.iterrows():
                    fig.add_trace(go.Scatter(x=self.xax, y=row, name=index, line=dict(color=self.color_dict[group]), showlegend=False), row=i+1, col=1)

        return self._figure_format(fig,len(group_order))

    def subplot_fig(self, group_order=None, **params):      
        return self._create_subplots(group_order, **params)
    
    def grouped_subplot(self, main_groups):
        pass

    def ave_subplot_figure(self, group_order=None, show_std=False):
        # Check for required parameters
        if group_order is None:
            group_order=list(self.group_dict.keys())

        # Create Subplots
        fig=make_subplots(rows=len(group_order), cols=1, shared_xaxes=True, subplot_titles=group_order)

        # Create subplot for each group with averaged spectra
        for i, group in enumerate(group_order):
            if group in self.group_dict:
                indices=self.group_dict[group]
                # Get subdf
                sub=self.df.loc[self.df.index.isin(indices)]
                # Compute average + standard deviation
                ave_spec=sub.mean(axis=0)
                std=sub.std(axis=0)
                # Add spectra to subplot
                fig.add_trace(go.Scatter(x=self.xax, y=ave_spec, name=group, line=dict(color=self.color_dict[group]), showlegend=False), row=i+1, col=1)

                if show_std: # Add standard deviation shading
                    fig.add_trace(go.Scatter(x=self.xax,y=ave_spec+std,
                                             mode='lines', name='STD Upper Bound',
                                             line=dict(width=0), fillcolor='rgba(0,100,80,0.2)',
                                             fill='tonexty', showlegend=False),
                                             row=i+1, col=1)
                    fig.add_trace(go.Scatter(x=self.xax,y=ave_spec-std,
                                             mode='lines', name='STD Lower Bound',
                                             line=dict(width=0), fillcolor='rgba(0,100,80,0.2)',
                                             fill='tonexty', showlegend=False),
                                             row=i+1, col=1)
        return self._figure_format(fig,len(group_order))
    