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
from scipy.io import wavfile # added by zk to read just the one channel of the file
import math 
from scipy.signal import filtfilt
import numpy as np 
from matplotlib.backends.backend_pdf import PdfPages  # Deal with PDFs
import random
import shutil

from audio_autoSegmentation_helper import *


"""DATA PATHS"""
#path_load = '/mnt/sphere/speech_bci/raw_data/z_b9m16_21/2021-12-08/'
path_load = '/mnt/sphere/speech_bci/raw_data/'
path_save = '/mnt/sphere/speech_bci/derived_data/'

'''VARIABLES set by user'''
th = 8  # Amplitude Detection Threshold -> # of standard devioations above rms for song detection
POIs2save = 200  # Number of sample POIs found to be saved (.wav sample, wave & spectrogram plots)
time_between_poi = 2  # Number of seconds needed to consider two POIs independent.
min_poi_time = 0.5  # Only save POI if it's at least min_poi seconds long (discard random noisy threshold crossings)
    
'''FILTER'''
b, a = load_filter_coefficients_matlab('/home/finch/scripts/Audio-Segmentation-Parsing_BirdSong/filters/butter_bp_250hz-8000hz_order4_sr48000.mat')
    
    
'''CODE'''
#if not os.path.isdir(path_load): print('Data path does not exist.')
    
for x in os.walk(path_load):
    
    try:
    
        os.chdir(x[0])
        audio_files = np.sort(glob('*.wav')) # Retrieve all .wav files in folder, ordered alphabetically

        # Get Bird & Session names to store results
        path = os.path.normpath(x[0])
        path_folders = path.split(os.sep)

        bird = path_folders[-3]
        session = path_folders[-2]
        alsa_folder = path_folders[-1]

        print('Searching directory: ', path)
        # if the bird is starling, announce it and raise value error to get to the next part of the loop
        # it technically is a value error, since it is a wrong bird

        if bird.split('_')[0] == 's':
            print('Bird is starling, will just skip it')
            raise ValueError

        if alsa_folder=='alsa' and audio_files.size != 0 and not os.path.isdir(path_save + bird + '/' + session + '/bout_detection_threshold/'):  # If session has recorded .wav files and has not already been analyzed.

            print('Segmenting audion from bird:', bird, ', session:', session)

            # Create folder where to store data if it does not exist already and change directory.
            # since we are using os.path anyway...
            dir_save = os.path.join(path_save, bird, session, 'bout_detection_threshold')
            os.makedirs(dir_save, exist_ok=True, mode=0o777)
            print('saving to {}'.format(dir_save))

            #Path(path_save + bird + '/' + session + '/bout_detection_threshold/').mkdir(parents=True, exist_ok=True, mode=0o775)

            os.chdir(path_save + bird + '/' + session + '/bout_detection_threshold/')

            # RETRIEVE ALL POIs FROM ALL AUDIO FILES

            pois = []   # Store all found POIs
            audio_filename = [] # Save the file they belong to, to have an idea of the time at which song occurs
            for af in audio_files[:]:
                
                # If .wav file is not empty (sometimes the recording/saving goes wrong)
                if os.path.getsize(x[0] + '/' + af) != 0:
                
                    print('Loading file: ', af)

                    # To preserve the native sampling rate of the file, use sr=None
                    # To load all channels (avoid averaging them), use mono=False and take the desired one [0]
                    sr, audio_signal = wavfile.read(x[0] + '/' + af, mmap=False)  
                    audio_signal = audio_signal[:, 0] # get the first channel 

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
            pdf_wave = PdfPages('POIs_pressureWave_' + str(bird) + '_' + str(session) + '_' + str(len(pois)) + 'POIs.pdf')
            pdf_spectrogram = PdfPages('POIs_spectrogram_' + str(bird) + '_' + str(session) + '_' + str(len(pois)) + 'POIs.pdf')

            # Plot snippets of X POIs (if sufficient found) and save them:
            numPois2plot = np.min((POIs2save, len(pois)))
            ex2plot = np.sort(random.sample(range(len(pois)), numPois2plot))  # generate 200 random integer values without duplicates, sorted so that they are saved in order of occurrence during the day

            for poi in range(len(ex2plot)):
                signal = pois[ex2plot[poi]]

                # wave
                plt.figure()
                plt.plot(np.linspace(0,len(signal)/sr,len(signal)), signal)
                plt.ylabel('Amplitude')
                plt.xlabel('Time (s)')
                plt.title('POI {} in session {}, found in file #{}'.format(ex2plot[poi], session, audio_filename[ex2plot[poi]]))
                # When no figure is specified the current figure is saved
                pdf_wave.savefig()
                plt.close()

                #spectrogram
                plt.figure()
                powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(signal, Fs=sr)
                plt.axis(ymin=0, ymax=10000)
                plt.xlabel('Time')
                plt.ylabel('Frequency')
                plt.title('POI {} in session {}, found in file #{}'.format(ex2plot[poi], session, audio_filename[ex2plot[poi]]))
                # When no figure is specified the current figure is saved
                pdf_spectrogram.savefig()
                plt.close()

            # Save all .wav files to train classifiers in the future:
            for poi in range(len(pois)):
                signal = pois[poi]

                # Save .wav file of snippet
                sf.write(str(session) + '_' +'POI' + str(poi) + '.wav', signal, sr)

            print('Saved figures to PDF')
            pdf_wave.close()
            pdf_spectrogram.close()
            
            if bird[0] == 'z': 
                print('Successfully extraced and saved .wav files. Deleting the following raw data path: ', path)
                shutil.rmtree(path) # Remove raw data to avoid cluttering the harddrive

        else: print('Bird {} session {} is empty or has already been segmented'.format(bird, session))

            
    except ValueError:
        print('There was an unexpected error investigating (potentially) bird {} session {}.'.format(bird, session))

print('DONE PARSING!')
