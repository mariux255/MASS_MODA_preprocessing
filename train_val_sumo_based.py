from scipy import signal
from scipy.fft import fftshift
import numpy as np

from scipy.signal import butter, sosfilt, sosfreqz, resample, sosfiltfilt
rng = np.random.default_rng()

from mne.time_frequency import tfr_array_multitaper

from os import listdir, mkdir, walk
from os.path import isfile, join, exists, isdir

import numpy as np
import matplotlib.pyplot as plt
import json
import random
import mne
import pandas as pd

OVERLAP = 0
WINDOW_SIZE = 30
MASS_sampling_freq = 256
val_proportion = 0.2
test_proportion = 0

def main():
    MASS_path = '/dtu-compute/macaroni/data/mass'
    MODA_path = '/home/s174411/code/MODA_GC/output/exp/annotFiles'
    MASS_MODA_proccessed_path = '/scratch/s174411/sumo_split_fix_30'

    if not exists(MASS_MODA_proccessed_path):
                mkdir(MASS_MODA_proccessed_path)

    PSG_recordings = {}
    for root, dirs, files in walk(MASS_path, topdown=False):
	    for name in files:
	        if name[-7:] == 'PSG.edf':
	            PSG_recordings[name[-18:-8]] = (join(root,name))



    MODA_path_dict = {}
    for root, dirs, files in walk(MODA_path, topdown=False):
	    for name in files:
	        if name[-6:] == 'GS.txt':
	            MODA_path_dict[name[-22:-12]] = join(root, name)

    segment_10 = {}
    segment_others = {}
    ss_no_10_total = {'1':0, '2':0, '3':0, '4':0, '5':0}
    ss_total = {'1':0, '2':0, '3':0, '4':0, '5':0}
    for k,v in MODA_path_dict.items():
	    data = pd.read_csv(MODA_path_dict[k], sep="\t")
	    data.drop(data.columns[-1], axis=1, inplace=True)
	    relevant_segments = data[data.eventName == 'segmentViewed']
	    
	    if len(relevant_segments) == 10:
	        segment_10[k] = v
	        ss_total[k[4]] += 1
	    else:
	        segment_others[k] = v
	        ss_no_10_total[k[4]] += 1

    val_size = {}
    test_size = {}
    for k, v in ss_no_10_total.items():
	    val_size[k] = int(v*val_proportion)
	    test_size[k] = int(v*test_proportion)


    train_dirs_list = list(segment_others.values())
    val_dirs_list = []
    test_dirs_list = []
    val_creation = True
    while val_creation:
	    seq_choice = random.choice(train_dirs_list)
	    while val_size[seq_choice[-18]] == 0:
	        seq_choice = random.choice(train_dirs_list)
	    val_size[seq_choice[-18]] -= 1
	    val_dirs_list.append(seq_choice)
	    train_dirs_list.remove(seq_choice)
	    loop_check = False
	    for k,v in val_size.items():
	        if v > 0:
	            loop_check = True
	    if not loop_check:
	        val_creation = False
	            
    if test_proportion > 0:
	    test_creation = True
	    while test_creation:
	        seq_choice = random.choice(train_dirs_list)
	        while test_size[seq_choice[-18]] == 0:
	            seq_choice = random.choice(train_dirs_list)
	        test_size[seq_choice[-18]] -= 1
	        test_dirs_list.append(seq_choice)
	        train_dirs_list.remove(seq_choice)
	        loop_check = False
	        for k,v in test_size.items():
	            if v > 0:
	                loop_check = True
	        if not loop_check:
	            test_creation = False


    train_dirs_list = train_dirs_list + list(segment_10.values())

    for path in train_dirs_list:
    	get_segment_viewed(PSG_recordings, path, MASS_MODA_proccessed_path + '/TRAIN')


    for path in val_dirs_list:
    	get_segment_viewed(PSG_recordings, path, MASS_MODA_proccessed_path + '/VAL')


    for path in test_dirs_list:
    	get_segment_viewed(PSG_recordings, path, MASS_MODA_proccessed_path + '/TEST')


def overlapping_windows(sequence, labels, master_start, master_stop, sampling_frequency, window_duration, overlap):
    window_len = sampling_frequency * window_duration
    step_size = (1-overlap) * window_len

    # Additinal seq length is given, cause it's hard to divide a seq into equal lengths. 
    if ((master_stop-master_start)*sampling_frequency) % step_size == 0:
        no_windows = int((master_stop-master_start)*sampling_frequency/step_size)
    else:
        no_windows = int((master_stop-master_start) * sampling_frequency // step_size)
        no_windows += 1

    sequence_windowed = []
    labels_windowed = []
    for i in range(0,no_windows):
        window_start = int((i)*step_size)
        if window_start >= ((master_stop-master_start)*sampling_frequency):
            continue
        sequence_windowed.append(sequence[window_start:(window_start+window_len)])

        current_window = []
        for j in range(0,len(labels)):
            # if the spindles' end is inside the window and if the spindle starts before current window end
            if ((labels[j][0] + labels[j][1]) > (master_start + (window_start/sampling_frequency))) and (labels[j][0] < (master_start + window_duration + (window_start/sampling_frequency))):
                
                if ((labels[j][0] + labels[j][1]) - (master_start + (window_start/sampling_frequency))) < 0.3:
                    continue
                # if spindle starts before window start use window start as it's starting point
                if labels[j][0] < (master_start + (window_start/sampling_frequency)):
                    x1 = 0
                    x2 = (labels[j][0] + labels[j][1] - (master_start + (window_start/sampling_frequency)))/window_duration
                    width = x2 - x1
                    center = x1 + width/2
                    current_window.append((center, width))
                    continue
                # if spindle ends after windows end, use windows end as ending coordinate
                if (labels[j][0] + labels[j][1]) > (master_start + window_duration + (window_start/sampling_frequency)):
                    x1 = (labels[j][0] - (master_start + (window_start/sampling_frequency)))/window_duration
                    x2 = 1
                    width = x2 - x1
                    center = x1 + width/2
                    current_window.append((center, width))
                    continue
                
                x1 = (labels[j][0] - (master_start + (window_start/sampling_frequency)))/window_duration
                x2 = (labels[j][0] + labels[j][1] - (master_start + (window_start/sampling_frequency)))/window_duration
                width = x2 - x1
                center = x1 + width/2
                current_window.append((center, width))

        labels_windowed.append(current_window)
    return sequence_windowed, labels_windowed


def get_segment_viewed(mass_recordings_dict, moda_annotation_path, processed_path):
    file_name = moda_annotation_path[-22:-12]

    data = pd.read_csv(moda_annotation_path, sep="\t")
    data.drop(data.columns[-1], axis=1, inplace=True)
    relevant_segments = data[data.eventName == 'segmentViewed']

    recording_path = mass_recordings_dict[file_name]
    recording = mne.io.read_raw_edf(recording_path)

    channels = recording.info['ch_names']
    index = 0
    for i, ch in enumerate(channels):
        if 'C3' in ch:
            index = i

    raw_data = recording.get_data()
    c3_channel = raw_data[index,:]

    counter = 0
    for start, duration in zip(relevant_segments.startSec, relevant_segments.durationSec):
        ADDITIONAL_END = WINDOW_SIZE
        sequence = c3_channel[int(start*MASS_sampling_freq) : int((start + duration + ADDITIONAL_END)*MASS_sampling_freq)]


        labels = data[data.eventName == 'spindle']
        labels.drop(labels.columns[-1],axis=1, inplace = True)

        sequence_windowed, labels_windowed = overlapping_windows(sequence, labels.to_numpy(), start, start + duration, MASS_sampling_freq, WINDOW_SIZE, OVERLAP)

        for j,window in enumerate(sequence_windowed):
           
            if labels_windowed[j] == []:
                continue
            #fig, Sxx = plot_spectrogram(window, MASS_sampling_freq, win_sec = 2, fmin = 0.3, fmax = 20,train=True, arrays=True)

            if not exists(processed_path):
                mkdir(processed_path)

            if not exists(processed_path + '/input/'):
                mkdir(processed_path + '/input/')

            if not exists(processed_path + '/input/' + file_name):
                mkdir(processed_path + '/input/' + file_name)

            if not exists(processed_path + '/labels/'):
                mkdir(processed_path + '/labels/')

            if not exists(processed_path + '/labels/' + file_name):
                mkdir(processed_path + '/labels/' + file_name)

            window_np = np.asarray(window)
            #filtered_array = butter_bandpass_filter(window_np, 0.3, 30, 256, 10)
            filtered_array = window_np
            np.save(processed_path + '/input/' + file_name + "/" + str(counter) + '.npy', filtered_array)
            #np.save(Dreams_path + '/windowed' + '/labels/' + str(i) + "/" + str(j) + '.npy', label_windows)

            with open(processed_path + '/labels/' + file_name + "/" + str(counter) + '.json', 'w') as fp:
                json.dump({'boxes':labels_windowed[j], 'labels':[1]*len(labels_windowed[j])}, fp)

            counter += 1

            plt.close('all')


main()