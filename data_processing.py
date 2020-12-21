#  Sound files are all sampled at 16kHz, linear PCM quantization, .wav format #
from scipy.io import wavfile
import pandas as pd
import glob
import scipy.signal as signal
import numpy as np
import pywt

VOICE_MIN_ENERGY = 0.02

def get_energy(x, win_size=100):
    """
    get energy of signal at each index from a given window size
    :param x: signal
    :param win_size: length of window
    :return: signal energy at each index
    """
    temp = (x / np.max(np.abs(x))) ** 2
    energy = signal.oaconvolve(temp, np.ones(win_size), mode='same') / win_size
    return energy


def crop_signal(x, energy_cutoff=VOICE_MIN_ENERGY):
    """
    crops silence at beginning of voice signal
    :param x:
    :param energy_cutoff:
    :return:
    """
    energy = get_energy(x)
    start_index = np.where(energy >= energy_cutoff)[0][0]
    return x[start_index:]


def get_dataframes(crop_silence=False):
    """
    Gets data from txt files and .wav files and formats into DataFrames and Series
    Need to have 'kids' 'men' and 'women' directories in current working directory
    Need to have 'timedata.txt' 'vowdata.txt' and 'vowdata_stats.txt' in current working directory
    :param crop_silence: If True, attempts to remove silence at beginning of each voice sample
    :return: timedata, vowdata, vowdata_stats, full_data
    """

    timedata = pd.read_csv('timedata.txt', delim_whitespace=True, skiprows=5, index_col=0).sort_index()

    vowdata = pd.read_csv('vowdata.txt', delim_whitespace=True, header=None, names=['File', 'f0', 'F1', 'F2', 'F3', 'F4'],
                          skiprows=30, usecols=[0, 2, 3, 4, 5, 6], index_col=0).sort_index()

    vowdata_stats = pd.read_csv('vowdata_stats.txt', delim_whitespace=True, header=0, skiprows=9,
                                index_col=[1, 0, 8]).sort_index()


    all_data = []
    all_tags = []
    for person in ['men', 'women', 'kids']:
        all_data += [wavfile.read(fn)[1] for fn in glob.glob(person+'/*.wav')]
        all_tags += [fn[-9:-4] for fn in glob.glob(person+'/*.wav')]

    if crop_silence:
        all_data = [crop_signal(x) for x in all_data]

    full_data = pd.Series(data=all_data, index=all_tags, name='Signal').sort_index()

    return timedata, vowdata, vowdata_stats, full_data


def get_formatted_data(data_dataframe, group='all'):
    """
    after loading dataframes with get_dataframes, use this to get data and labels formatted for training/testing
    for specific groups (men, women, adults, etc.)
    :param data_dataframe: pandas dataframe "full_data" generated with get_dataframes function
    :param group: string, choose from ['m', 'w', 'b', 'g', 'adults', 'children', 'all'] (first 4 options correspond
    to men, women, boys, and girls)
    :return: (data_crop, labels), data is (num_samples, num_features) array, YL is (num_samples, ) array with integers from
    0 to 11 corresponding to the 12 different available vowels
    """
    vow_classes = {'ae': 0, 'ah': 1, 'aw': 2, 'eh': 3, 'er': 4, 'ei': 5,
                   'ih': 6, 'iy': 7, 'oa': 8, 'oo': 9, 'uh': 10, 'uw': 11}
    group_defs = {'adults': ['m', 'w'], 'children': ['b', 'g'], 'all': ['m', 'w', 'b', 'g']}
    assert(group in ['m', 'w', 'b', 'g', 'adults', 'children', 'all']), 'group param must be one of specified options'
    if group in ['m', 'w', 'b', 'g']:
        data = data_dataframe[data_dataframe.index.str[0] == group]
        data = data.loc[data.str.len() >= 5000]
        data_crop = data.apply(lambda x: x[0:5000])
        data_crop = np.stack(data_crop.values)
    else:
        data = data_dataframe.loc[np.isin(data_dataframe.index.str[0], group_defs[group])]
        data = data.loc[data.str.len() >= 5000]  # optional
        data_crop = data.apply(lambda x: x[0:5000])  # was 3500 w/out previous line
        data_crop = np.stack(data_crop.values)

    labels = data.index.str[3:5]
    labels = np.asarray([vow_classes[labels.values[x]] for x in range(len(labels))])

    return [data_crop, labels]


def get_WPD_coeffs(data, levels, wavelet='db10', extension_mode='zero'):
    """
    gets WPD data from each of the 5 frequency bands splitting up 0-5kHz and formats it for training.
    :param data: output from 'get_formatted_data' function. Must be (num_samples, num_features=5000) numpy array
    :param levels: list of packet tree levels (e.g. [6, 5, 4, 4, 4]). Must have 7 <= level <= 3
    :param wavelet: name of wavelet used for decomposition (check wavelet_dct_options in main.py for options)
    :param extension_mode: signal extension mode used by pywt library. I recommend either 'zero' or 'periodization'
    :return: vectorized coefficients prepared for training
    """

    coeffs = []
    for n in range(data.shape[0]):  # Results better with mixed level components !
        tree = pywt.WaveletPacket(data=data[n, :], wavelet=wavelet, mode=extension_mode)
        coeffs.append(np.split(np.concatenate([x.data for x in tree.get_level(level=levels[0], order='freq')]), 8)[0])
        coeffs.append(np.split(np.concatenate([x.data for x in tree.get_level(level=levels[1], order='freq')]), 8)[1])
        coeffs.append(np.split(np.concatenate([x.data for x in tree.get_level(level=levels[2], order='freq')]), 8)[2])
        coeffs.append(np.split(np.concatenate([x.data for x in tree.get_level(level=levels[3], order='freq')]), 8)[3])
        coeffs.append(np.split(np.concatenate([x.data for x in tree.get_level(level=levels[4], order='freq')]), 8)[4])
    coeffs = np.concatenate(coeffs)
    coeffs_vec = np.reshape(coeffs, (data.shape[0], int(len(coeffs)/data.shape[0])), order='C')

    return coeffs_vec



