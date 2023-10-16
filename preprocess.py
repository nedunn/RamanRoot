import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pybaselines.spline
import scipy.signal as ss

class PreproSpectra:
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
                 name=None, alerts=True, no_neg=True,
                 snv=True,
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
