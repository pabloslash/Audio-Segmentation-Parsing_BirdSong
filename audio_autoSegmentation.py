#!/home/finch/anaconda3/bin/python

"""IMPORTS"""

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


"""DATA PATHS"""
# path_load = '/mnt/sphere/speech_bci/raw_data/z_k17k18_21/2021-05-26/'
# path_load = '/net/expData/speech_bci/processed_data/audio_habituation/test_b1431/test2/'
path_load = '/mnt/sphere/speech_bci/raw_data/'
path_save = '/mnt/sphere/speech_bci/derived_data/'
# path_save = '/net/expData/speech_bci/processed_data/audio_habituation/test_b1431/'

'''VARIABLES set by user'''
th = 8  # Amplitude Detection Threshold -> # of standard devioations above rms for song detection
POIs2save = 200  # Number of sample POIs found to be saved (.wav sample, wave & spectrogram plots)
time_between_poi = 2  # Number of seconds needed to consider two POIs independent.
min_poi_time = 0.5  # Only save POI if it's at least min_poi seconds long (discard random noisy threshold crossings)
    
'''FILTER'''
b, a = load_filter_coefficients_matlab('/home/finch/scripts/Audio-Segmentation-Parsing_BirdSong/filters/butter_bp_250hz-8000hz_order4_sr48000.mat')
    
    
'''CODE'''
if not os.path.isdir(path_load): print('Data path does not exist.')
    
for x in os.walk(path_load):
    
    try:
    
        os.chdir(x[0])
        audio_files = glob('*.wav') # Retrieve all .wav files in folder

        # Get Bird & Session names to store results
        path = os.path.normpath(x[0])
        path_folders = path.split(os.sep)
        bird = path_folders[-2]
        session = path_folders[-1]

        print('Segmenting audio from bird:', bird, ', session:', session)

        if audio_files and not os.path.isdir(path_save + bird + '/' + session + '/'):  # If session has recorded files and has already not been analyzed.

            # Create folder where to store data if it does not exist already and change directory.
            Path(path_save + bird + '/' + session + '/').mkdir(parents=True, exist_ok=True)
            os.chdir(path_save + bird + '/' + session + '/')

            # RETRIEVE ALL POIs FROM ALL AUDIO FILES

            pois = []   # Store all found POIs
            audio_filename = [] # Save the file they belong to, to have an idea of the time at which song occurs
            for af in audio_files:

                print('Loading file: ', af)

                # To preserve the native sampling rate of the file, use sr=None
                audio_signal, sr = lr.load(x[0] + '/' + af, sr=None)  

                # Filter audio
                filt_audio_signal = noncausal_filter(audio_signal, b, a=a)

                # Rectify audio signal
                rf_filt_audio_signal = np.absolute(filt_audio_signal)

                # Calculate RMS of audio signal
                rms = calculate_signal_rms(rf_filt_audio_signal)

                # Create binary vector of indexes where the audio crosses the specified threshold
                idx_above_th = np.argwhere(rf_filt_audio_signal > th*rms)
                binary_signal = np.zeros(len(rf_filt_audio_signal))
                binary_signal[idx_above_th] = 1

                # Retrieve start / end sample index for each Period of Interest found.
                start_end_idxs = find_start_end_idxs_POIs(binary_signal, time_between_poi*sr, min_samples_poi=min_poi_time*sr)

                for poi in range(len(start_end_idxs)):
                    signal = audio_signal[start_end_idxs[poi][0]:start_end_idxs[poi][1]]
                    pois.append(signal)
                    audio_filename.append(af)

            # Save total number of POIs found to .txt file
            f = open("totalPOIs.txt","w+")
            f.write("{} total POIs found in bird {} session {}".format(len(pois), bird, session))
            f.close()
            print('Found {} POIs in session {}'.format(len(pois), session))

            ## SAVE SOME SAMPLE POIs

            # Create pdf to save snippets of POIs found (waveforms & spectrograms).
            pdf_wave = PdfPages('POIs_pressureWave_' + str(bird) + '_' + str(session) + '.pdf')
            pdf_spectrogram = PdfPages('POIs_spectrogram_' + str(bird) + '_' + str(session) + '.pdf')

            # Plot snippets of X POIs (if sufficient found) and save them:
            numPois2plot = np.min((POIs2save, len(pois)))
            ex2plot = random.sample(range(len(pois)), numPois2plot)  # generate 200 random integer values without duplicates
            for poi in range(len(ex2plot)):
                signal = pois[ex2plot[poi]]

                # wave
                plt.figure()
                plt.plot(np.linspace(0,len(signal)/sr,len(signal)), signal)
                plt.ylabel('Amplitude')
                plt.xlabel('Time (s)')
                plt.title('POI {} in session {}, found in file #{}'.format(poi, session, audio_filename[ex2plot[poi]]))
                # When no figure is specified the current figure is saved
                pdf_wave.savefig()
                plt.close()

                #spectrogram
                plt.figure()
                powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal, Fs=sr)
                plt.axis(ymin=0, ymax=10000)
                plt.xlabel('Time')
                plt.ylabel('Frequency')
                plt.title('POI {} in session {}, found in file #{}'.format(poi, session, audio_filename[ex2plot[poi]]))
                # When no figure is specified the current figure is saved
                pdf_spectrogram.savefig()
                plt.close()

                # Save .wav file of snippet
                sf.write(str(session) + '_' +'POI' + str(poi) + '.wav', signal, sr)


            print('Saved figures to PDF')
            pdf_wave.close()
            pdf_spectrogram.close()

        else: print('Bird {} session {} is empty or has already been segmented'.format(bird, session))

            
    except ValueError:
        print('There was an unexpected error investigating (potentially) bird {} session {}.'.format(bird, session))
