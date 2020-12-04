

'''IMPORTS'''

import numpy  as np    
from glob import glob  
from scipy.io import loadmat  
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt  
from math import pi
import librosa as lr 
import soundfile as sf
import os
from pathlib import Path
from scipy import signal
import math 
from scipy.signal import filtfilt
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages  # Deal with PDFs
import random

from audio_autoSegmentation_helper import *



'''DATA PATHS'''
path_load = '/net/expData/speech_bci/raw_data/'

    
'''CODE'''
    
# Create a new file if it does not exist to write summary to:
f = open(path_load + "rawData_summary.txt", "w")
    
for x in os.walk(path_load):
    
    print('Working on ' + x[0])
    
    os.chdir(x[0])
    audio_files = glob('*.wav') # Retrieve all .wav files in folder
    
    print('{} ')
    
    f.write(x[0] + ": " + str(len(audio_files)) + ".wav files")
    if len(audio_files) < 24: f.write( "   Warning!")
    f.write("\n")

f.close()
    

   