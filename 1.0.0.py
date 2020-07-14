

# Import the following packages:
import numpy  as np  # for numerical computation
from glob import glob  # for reading files
from scipy.io import loadmat  # for loading .mat files from matlab (loading a,b for the butter filter)

from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt  # visualization  of audio files amplitude, It provides a MATLAB-like way of plotting.

from math import pi
import librosa as lr

from scipy import signal
import math
from scipy.signal import filtfilt
import numpy.ma as ma
import numpy as np

"""set the path of wav. files
call the file from the folder (set the path of the folder). use glob to call the wav file load the file 
[number of file]. Audio variable represent the data of the file, in terms of amplitude and [sfreq] 
variable represent the Nyquist frequency which is half 1/2 of sampling frequency.
"""
pathname = '/2019-09-23_findSong'
audio_files = glob('*.wav')
audio_array, nyquist_frequency = lr.load(audio_files[1],8000)

# we can create a for loop or a while loop to run all wav. files at the same time.

"""Preprocessing Data
use the butter high-pass filter to remove the noise and unecessary audioable data. variable a, and b are 
calculated from MATLAB and imported to python to be used as they are more accurate in matlab than 
python. butter.mat is the file name in MATLAB. zero was added to the vector to make ....."""

signaldata = loadmat('butterf.mat')
b = signaldata['b'][0]; a = signaldata['a'][0]
#y = lfilter(b, a, audio) # never use it for audio (for neural is not good to use it create a phase shift)
audio_filtered = filtfilt(b,a,audio_array)
# you should use filtfilt (it is in scipy.signal)

# plot the frequency response by showing the critical point. (make sure to import the signal package from scipy)
w,h = signal.freqs(b,a)
# b, a = signal.butter(5, 100, 'low', analog=True)



#Rectify the data:
rectify_data_unfiltered = np.absolute(audio_array)
rectify_data_filtered = np.absolute(audio_filtered)


## Plot the original unfiltered data and the new filtered data
# The Filtered Audio
plt.figure()
plt.plot(audio_filtered[0:10**5])
plt.ylabel('Amplitude dB/Sample')
plt.xlabel('Frequency rad/sample')
plt.title('frequency response/Filtered Data')

# the Unfiltered Audio
plt.figure()
plt.plot(audio_array[0:10**5])
plt.ylabel('Amplitude dB/Sample')
plt.xlabel('Frequency rad/sample')
plt.title('Frequency Response/Unfiltered Data')



# The recified Filtered Audio
plt.figure()
plt.plot(rectify_data_filtered[0:10**5])
plt.ylabel('Amplitude dB/Sample')
plt.xlabel('Frequency rad/sample')
plt.title('frequency response/rectify_data_filtered')

# the rectified Unfiltered Audio
plt.figure()
plt.plot(rectify_data_unfiltered[0:10**5])
plt.ylabel('Amplitude dB/Sample')
plt.xlabel('Frequency rad/sample')
plt.title('Frequency Response/rectify_data_unfiltered')




"""Graph The Frequency Response
The plotting for the frequency response, showing the critical points of the w,h 
using the variables a and b from matlab file import."""
#Note: The cut frequency is equal to 0.5
b = signaldata['b'][0]; a = signaldata['a'][0]
w,h = signal.freqs(b,a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth Filter Frequency Response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude[dB]')
plt.ylim((0,8))
plt.xlim((-10,10))
plt.margins(0,0.01)
plt.grid(which='both',axis='both')
plt.axvline(0.2,color ='green') #cutoff frequency
plt.show()


"""Root Mean Square and Threshold
Function that determine RMS and can return values:rms,mean,root,square. rms is used to 
dertmine the Threshold. Find the Threshold of the filtered data: find the RMS and multiply 
by j which is 3, or 4 or 5 depend. mean is equal to zero create a function that clacualte the RMS"""

def RMS_value(rectify_data_filtered):
    # intiate the coeffieciants for [square,mean,root]
    square = 0.0
    mean = 0.0
    root = 0.0
    for i in range (0,len(rectify_data_filtered)):
        square += (audio_filtered[i]**2)
    mean = square / (float)(len(rectify_data_filtered))
    rms = math.sqrt(mean)
    return rms

threshold = 2 * RMS_value(rectify_data_filtered)
print(threshold)


## Plot wav.audio including the threshold
plt.figure()
plt.plot(audio_array[0:10**5])
plt.plot(np.linspace(5,len(audio_array[0:10**5]),len(audio_array[0:10**5])),[threshold]*len(audio_array[0:10**5]))
plt.ylabel('Amplitude dB/Sample')
plt.xlabel('Frequency rad/sample')
plt.title('lowpass filter frequency response')

plt.figure()
plt.plot(audio_filtered[0:100000])
plt.plot(np.linspace(5,len(audio_filtered[0:100000]),len(audio_filtered[0:100000])),[threshold]*len(audio_filtered[0:100000]))
plt.ylabel('Amplitude dB/Sample')
plt.xlabel('Frequency rad/sample')
plt.title('filtered freq_response/zoomIN')


plt.figure()
plt.plot(rectify_data_unfiltered[0:10**5])
plt.plot(np.linspace(5,len(rectify_data_unfiltered[0:10**5]),len(rectify_data_unfiltered[0:10**5])),[threshold]*len(rectify_data_unfiltered[0:10**5]))
plt.ylabel('Amplitude dB/Sample')
plt.xlabel('Frequency rad/sample')
plt.title('lowpass filter frequency response')

plt.figure()
plt.plot(rectify_data_filtered[0:100000])
plt.plot(np.linspace(5,len(rectify_data_filtered[0:100000]),len(rectify_data_filtered[0:100000])),[threshold]*len(rectify_data_filtered[0:100000]))
plt.ylabel('Amplitude dB/Sample')
plt.xlabel('Frequency rad/sample')
plt.title('filtered freq_response/zoomIN')



"""Create an array of 0 and 1 where 1 represent the sound and 0 refer to silience. """

def index_abv_thr(x,th):
    idx_abv_th = np.array([0]*len(x))
    for i in range(len(x)):
        if x[i] >= th:
            idx_abv_th[i] = 1
    return idx_abv_th

bsing_array = index_abv_thr(rectify_data_filtered,threshold)
print(bsing_array)
plt.figure()
num_samples = 10000
plt.scatter(np.linspace(0, len(bsing_array[0:num_samples]), num=len(bsing_array[0:num_samples])), bsing_array[0:num_samples])



