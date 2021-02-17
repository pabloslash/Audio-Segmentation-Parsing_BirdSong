'''Helper functions for automatic audio segmentation'''

# Imports
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter, filtfilt, freqz
import numpy as np

'''
FILTER FUNCTIONS
'''

# Load Butterworth filter coefficients from a file
def load_filter_coefficients_matlab(filter_file_path):
    coefficients = loadmat(filter_file_path)
    a = coefficients['a'][0]
    b = coefficients['b'][0]
    return b, a  # The output is a double list after loading .mat file


# Filter non-causally (forward & backwards) given filter coefficients
def noncausal_filter(signal, b, a=1):
    y = filtfilt(b, a, signal)
    return y


def calculate_signal_rms(signal):
    """"
     Returns the root mean square {sqrt(mean(samples.^2))} of a 1D vector.
    """
    return np.sqrt(np.mean(np.square(signal)))


def find_start_end_idxs_POIs(binary_signal, samples_between_poi, min_samples_poi=1):
    """"
    Returns a list of tuples (start_idx, end_idx) for each period of interest found in the audio file.
    
     Input: Binary vector where ones indicate samples that are above a specified audio threshold, zeros indicate samples below the threshold.
     samples_between_poi = Number of samples needed to consider two POIs independent.
     min_samples_poi = Minimum number of samples that a POI must have to not be discarded.

     Output: list of [start_idx, end_idx] for each POI found.
    """
    start_end_idxs = []
    start_idx = None 
    end_idx = None 
    zero_counter = 0

    for i in range(len(binary_signal)):

        if binary_signal[i] == 1:
            if start_idx == None:
                start_idx = i
#                 print(start_idx)

            elif zero_counter != 0 or end_idx is not None:
                zero_counter = 0
                end_idx = None

        elif binary_signal[i] == 0:

            if start_idx != None:
                if zero_counter == 0: 
                    end_idx = i
                    zero_counter += 1

                elif zero_counter < samples_between_poi:
                    zero_counter += 1

                elif zero_counter >= samples_between_poi:
                    if end_idx - start_idx > min_samples_poi:  
                        start_end_idxs.append([start_idx, end_idx])
                    start_idx = None
                    end_idx = None
                    zero_counter = 0

        # If we are in a poi and the file ends
        if i == len(binary_signal)-1 and start_idx != None:
            start_end_idxs.append([start_idx, i])
            
    return start_end_idxs