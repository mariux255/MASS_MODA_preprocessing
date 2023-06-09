
OVERLAP = 0.5
WINDOW_SIZE = 23
MASS_sampling_freq = 256


MASS_path = '/home/marius/Documents/THESIS/data/mass'
MODA_path = '/home/marius/Documents/THESIS/data/MODA_GC/output/exp/annotFiles'
MASS_MODA_proccessed_path = '/home/marius/Documents/THESIS/data'

from scipy import signal
from scipy.fft import fftshift
import numpy as np

from scipy.signal import butter, sosfiltfilt, sosfreqz
rng = np.random.default_rng()

from mne.time_frequency import tfr_array_multitaper

from os import listdir, mkdir, walk
from os.path import isfile, join, exists

"""
Plotting functions of YASA.
"""
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize, ListedColormap
import json
import mne


def plot_spectrogram(
    data,
    sf,
    win_sec=30,
    fmin=0.5,
    fmax=25,
    trimperc=2.5,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    train=False,
    arrays = False,
    **kwargs,
):
   
    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 18})

    # Safety checks
    assert isinstance(data, np.ndarray), "Data must be a 1D NumPy array."
    assert isinstance(sf, (int, float)), "sf must be int or float."
    assert data.ndim == 1, "Data must be a 1D (single-channel) NumPy array."
    assert isinstance(win_sec, (int, float)), "win_sec must be int or float."
    assert isinstance(fmin, (int, float)), "fmin must be int or float."
    assert isinstance(fmax, (int, float)), "fmax must be int or float."
    assert fmin < fmax, "fmin must be strictly inferior to fmax."
    assert fmax < sf / 2, "fmax must be less than Nyquist (sf / 2)."
    assert isinstance(vmin, (int, float, type(None))), "vmin must be int, float, or None."
    assert isinstance(vmax, (int, float, type(None))), "vmax must be int, float, or None."
    if vmin is not None:
        assert isinstance(vmax, (int, float)), "vmax must be int or float if vmin is provided"
    if vmax is not None:
        assert isinstance(vmin, (int, float)), "vmin must be int or float if vmax is provided"


    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    assert data.size > 2 * nperseg, "Data length must be at least 2 * win_sec."
    
    
    noverlap = int(0.9 * nperseg)
    
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=noverlap)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    # if raw is fed, use the Sxx below
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    #t /= 3600  # Convert t to hours

    # Normalization
    if vmin is None:
        vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
    norm = Normalize(vmin=vmin, vmax=vmax)


    fig, ax1 = plt.subplots(nrows=1, figsize=(16, 8))


    # Draw Spectrogram
    im = ax1.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading="auto")
    #ax1.set_xlim(0, t.max())
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [seconds]")


    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1, shrink=0.95, fraction=0.1, aspect=25)
    cbar.ax.set_ylabel("Log Power (dB / Hz)", rotation=270, labelpad=20)

    # Revert font-size
    plt.rcParams.update({"font.size": old_fontsize})

    if(train and arrays):
        ax1.set_axis_off()
        cbar.remove()

        return fig,Sxx
    elif(arrays):
        return Sxx
    else:
        ...
        #ax1.set_xticks([0,1/12, 2/12, 3/12, 4/12, 5/12, 6/12], labels = np.arange(0, 35, 5))
        


    return fig

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y

def overlapping_windows(sequence, labels, master_start, master_stop, sampling_frequency, window_duration, overlap):
    window_len = sampling_frequency * window_duration
    step_size = (1-overlap) * window_len
    no_windows = int(sequence.shape[0]/step_size)

    sequence_windowed = []
    labels_windowed = []
    for i in range(0,no_windows):
        window_start = int((i)*step_size)
        if window_start > (master_stop*sampling_frequency - window_len):
            continue
        sequence_windowed.append(sequence[window_start:(window_start+window_len)])
        current_window = []
        for j in range(0,len(labels)):
            # NEEDS TO BE CHANGED. If spindle more than 0.5s then keep
            if (labels[j][0] - master_start < (window_start+window_len)/sampling_frequency) and (labels[j][0] - master_start > window_start/sampling_frequency):
                x1 = (labels[j][0]-(window_start/sampling_frequency) - master_start)/window_duration
                y1 = 0
                x2 = (labels[j][0]-(window_start/sampling_frequency) - master_start + labels[j][1])/window_duration
                y2 = 1
                current_window.append((x1, y1, x2, y2))
        labels_windowed.append(current_window)
    return sequence_windowed, labels_windowed

# Load MODA txt files. Load MASS data but only look at 'SegmentViewed' sequences.
# Put 'SegmentViewed' sequences into overlapping_window function and find spindles for windows.

def get_segment_viewed(mass_recordings_dict, moda_annotation_path, processed_path):
    file_name = moda_annotation_path[-22:-12]

    data = pd.read_csv(moda_annotation_path, sep="\t")
    data.drop(data.columns[-1], axis=1, inplace=True)
    relevant_segments = data[data.eventName == 'segmentViewed']

    recording_path = mass_recordings_dict[file_name]
    recording = mne.io.read_raw_edf(recording_path)
    raw_data = recording.get_data()
    c3_channel = raw_data[6,:]
    
    for start, duration in zip(relevant_segments.startSec, relevant_segments.durationSec):
        ADDITIONAL_END = 15

        sequence = c3_channel[int(start*MASS_sampling_freq) : int((start + duration + ADDITIONAL_END)*MASS_sampling_freq)]


        labels = data[data.eventName == 'spindle']
        labels.drop(labels.columns[-1],axis=1, inplace = True)

        sequence_windowed, labels_windowed = overlapping_windows(sequence, labels.to_numpy(), start, start + duration, MASS_sampling_freq, WINDOW_SIZE, OVERLAP)


        counter = 0
        for j,window in enumerate(sequence_windowed):
            if labels_windowed[j] == []:
                continue
            fig, Sxx = plot_spectrogram(window, MASS_sampling_freq, win_sec = 2, fmin = 0.3, fmax = 20,train=True, arrays=True)

            if not exists(processed_path + '/MASS_MODA_processed'):
                mkdir(processed_path + '/MASS_MODA_processed')

            if not exists(processed_path + '/MASS_MODA_processed' + '/images'):
                mkdir(processed_path + '/MASS_MODA_processed' + '/images')

            if not exists(processed_path + '/MASS_MODA_processed' + '/images/' + file_name):
                mkdir(processed_path + '/MASS_MODA_processed' + '/images/' + file_name)

            if not exists(processed_path + '/MASS_MODA_processed' + '/real/'):
                mkdir(processed_path + '/MASS_MODA_processed' + '/real/')

            if not exists(processed_path + '/MASS_MODA_processed' + '/real/' + file_name):
                mkdir(processed_path + '/MASS_MODA_processed' + '/real/' + file_name)

            if not exists(processed_path + '/MASS_MODA_processed' + '/labels/'):
                mkdir(processed_path + '/MASS_MODA_processed' + '/labels/')

            if not exists(processed_path + '/MASS_MODA_processed' + '/labels/' + file_name):
                mkdir(processed_path + '/MASS_MODA_processed' + '/labels/' + file_name)


            fig.savefig(processed_path + '/MASS_MODA_processed' + '/images/' + file_name + "/" + str(counter) + '.png', bbox_inches='tight')
            #print(processed_path + '/MASS_MODA_processed' + '/real/' + file_name + "/" + str(counter))
            np.save(processed_path + '/MASS_MODA_processed' + '/real/' + file_name + "/" + str(counter) + '.npy', Sxx)
            #np.save(Dreams_path + '/windowed' + '/labels/' + str(i) + "/" + str(j) + '.npy', label_windows)
            with open(processed_path + '/MASS_MODA_processed' + '/labels/' + file_name + "/" + str(counter) + '.json', 'w') as fp:
                json.dump({'boxes':labels_windowed[j], 'labels':[0]*len(labels_windowed[j])}, fp)

            counter += 1

            plt.close('all')

    

PSG_recordings = {}
for root, dirs, files in walk(MASS_path, topdown=False):
    for name in files:
        if name[-7:] == 'PSG.edf':
            PSG_recordings[name[-18:-8]] = (join(root,name))


for root, dirs, files in walk(MODA_path, topdown=False):
    for name in files:
        if name[-6:] == 'GS.txt':
            print(name)
            try:
                get_segment_viewed(PSG_recordings, join(root,name), MASS_MODA_proccessed_path)
            except:
                print(name, ' did not load')
   
