#  Sound files are all sampled at 16kHz, linear PCM quantization, .wav format #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.style.use('seaborn-deep')
import seaborn as sns
import sklearn as skl
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.utils import np_utils
from keras.utils import normalize
import scipy
import cv2
import scipy.signal as signal
from scipy.io import wavfile
import pandas as pd
import glob
from extra_functions import *
import scipy.fft as fft
from data_processing import *
import pywt


Fs = 16000
Ts = 1/Fs
window_options = ['boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'parzen', 'blackmanharris', 'barthann']

wavelet_cts_options = ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1',
                       'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']

wavelet_dct_options = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1',
                       'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2',
                       'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12',
                       'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6',
                       'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18',
                       'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30',
                       'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1',
                       'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3',
                       'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4',
                       'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14',
                       'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']

vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']
vow_classes = {'ae': 0, 'ah': 1, 'aw': 2, 'eh': 3, 'er': 4, 'ei': 5,
               'ih': 6, 'iy': 7, 'oa': 8, 'oo': 9, 'uh': 10, 'uw': 11}
characters = ['m', 'w', 'b', 'g']  # man woman boy girl

NUM_VOWELS = 12


# Processing and loading data #
time_df, vow_df, vow_stats_df, data_df = get_dataframes(crop_silence=True)

svm_data, YL = get_formatted_data(data_df, 'm')


# general method for getting STFT coefficients #
win_len = 128
f, t, Zxx = signal.stft(svm_data, Fs, window='bartlett', nperseg=win_len, noverlap=win_len // 2,
                        nfft=win_len, return_onesided=True, boundary='zeros', padded=True, axis=1)
Z_stft = np.abs(Zxx) ** 2
Z_stft = np.reshape(Z_stft, (Z_stft.shape[0], Z_stft.shape[1] * Z_stft.shape[2]), order='C')


#  some random useful functions for CWT stuff  #
wvlt = 'cmor1-1.5'
center_freq = pywt.scale2frequency(wvlt, 1)/Ts
# scales = np.arange(1, 128)
scales = center_freq/np.linspace(2000, 1, 50)
coeff, freq = pywt.cwt(svm_data, scales=scales, wavelet=wvlt, sampling_period=Ts, method='auto', axis=1)
power = np.abs(coeff) ** 2
power = np.reshape(power, (power.shape[0], power.shape[1] * power.shape[2]), order='C')


#  some random useful functions for DWT stuff  #
wvlt = 'sym10'
wvlt_data = pywt.Wavelet(wvlt)
father, mother, coords = wvlt_data.wavefun(level=1)  # father=scaling=LPF
lpf_coeffs, hpf_coeffs = wvlt_data.filter_bank[0:2]
center_freq = pywt.scale2frequency(wvlt, 1)/Ts
coeffs = pywt.wavedecn(svm_data, wavelet=wvlt, mode='periodization', level=6, axes=1)
coeffs_vec, coeffs_slices = pywt.coeffs_to_array(coeffs, axes=[1])
coeffs_vec = np.abs(coeffs_vec) ** 2


#  first attempt at training SVM with STFT coefficients (note: no normalization of coefficients or k fold CV)  #
svm_data, YL = get_formatted_data(data_df, group='w')

win_len = 256 // 2
f, t, Zxx = signal.stft(svm_data, Fs, window='bartlett', nperseg=win_len, noverlap=win_len // 2,
                        nfft=win_len, return_onesided=True, boundary='zeros', padded=True, axis=1)

Z_stft = np.abs(Zxx) ** 2
Z_stft = np.reshape(Z_stft, (Z_stft.shape[0], Z_stft.shape[1] * Z_stft.shape[2]), order='C')

x_train, x_validation, y_train, y_validation = train_test_split(Z_stft, YL, train_size=0.8, stratify=YL)

clf = SVC(gamma='scale', kernel='rbf', degree=1, C=3.0)
clf.fit(x_train, y_train)
score = clf.score(x_validation, y_validation)
print(score)

test = clf.decision_function(x_validation)
predictions = np.argmax(test, axis=1)


svm_data, YL = get_formatted_data(data_df, group='adults')

win_len = 128
f, t, Zxx = signal.stft(svm_data, Fs, window='bartlett', nperseg=win_len, noverlap=win_len // 2,
                        nfft=win_len, return_onesided=True, boundary='zeros', padded=True, axis=1)
# plt.figure(dpi=300)
# plt.pcolormesh(t, f, np.abs(Zxx[0, :, :]) ** 2, cmap='viridis', shading='nearest')

Z_stft = np.abs(Zxx[:, :1 + win_len//4 + win_len//16, :]) ** 2
Z_stft = np.reshape(Z_stft, (Z_stft.shape[0], Z_stft.shape[1] * Z_stft.shape[2]), order='C')
Z_stft = Z_stft / np.max(Z_stft, axis=0)
Z_stft = Z_stft - np.mean(Z_stft, axis=0)
coeffs_vec = Z_stft


for constant in [1]:  # placeholder if you want to loop some variable within the loop (e.g. on of the WPD levels)
    coeffs_vec = get_WPD_coeffs(svm_data, levels=[6, 5, 4, 4, 4], wavelet='db10', extension_mode='zero')
    coeffs_vec = np.abs(coeffs_vec) ** 2
    coeffs_vec = coeffs_vec/np.max(coeffs_vec, axis=0)
    coeffs_vec = coeffs_vec - np.mean(coeffs_vec, axis=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    skf.get_n_splits(coeffs_vec, YL)

    scores = [[] for x in range(5)]
    for p, indices in enumerate(skf.split(coeffs_vec, YL)):
        x_train, y_train, x_validation, y_validation = get_kfolds(coeffs_vec, YL, indices)
        U, mu = PCA(x_train)  # Using 100-300 PCs is good (For both DWT and STFT) (210 for sym10)

        for n in range(10, 500, 10):
            Z_train, Ul_train = apply_PCA_from_Eig(x_train, U, n, mu)
            Z_validation, Ul_validation = apply_PCA_from_Eig(x_validation, U, n, mu)
            clf = SVC(gamma='scale', kernel='poly', degree=1, C=3.0)  # C=3.0 for poly 1, C=3.0 for rbf, sigmoid bad
            clf.fit(Z_train, y_train)
            score = clf.score(Z_validation, y_validation)
            scores[p].append(score)
        print('Finished fold %d' % (p+1))
    scores_mean = np.mean(np.asarray(scores), axis=0)
    print(scores_mean)

test = clf.decision_function(x_validation)
predictions = np.argmax(test, axis=1)


y_predict = clf.predict(x_validation)  # this and one below are for finding confusion matrix

plt.figure(dpi=300)
confmat = confusion_matrix(y_validation, y_predict)
sns.heatmap(confmat, annot=True, annot_kws={'size': 6}, xticklabels=list(vow_classes.keys()), yticklabels=list(vow_classes.keys()), cbar=False, square=True)
plt.title('Confusion Matrix for All Segment Classifier')
plt.xlabel('Predicted Character')
plt.ylabel('True Character')
plt.xticks(rotation=0)
plt.yticks(rotation=0)


boy_data, YL = get_formatted_data(data_df, group='all')

svm_data, YL = get_formatted_data(data_df, group='all')


win_len = 32  # Interestingly, 128 performs better than 256
f, t, Zxx = signal.stft(svm_data, Fs, window='bartlett', nperseg=win_len, noverlap=win_len // 2,
                        nfft=win_len, return_onesided=True, boundary='zeros', padded=True, axis=1)

Z_stft = np.abs(Zxx[:, :1 + win_len//4 + win_len//16, :]) ** 2
Z_stft = np.reshape(Z_stft, (Z_stft.shape[0], Z_stft.shape[1] * Z_stft.shape[2]), order='C')
Z_stft = Z_stft / np.max(Z_stft, axis=0)
Z_stft = Z_stft - np.mean(Z_stft, axis=0)
coeffs_vec = Z_stft

# coeffs = pywt.wavedecn(svm_data, wavelet='db10', mode='zero', level=6, axes=1)
# coeffs_vec, coeffs_slices = pywt.coeffs_to_array(coeffs, axes=[1])

def baseline_model(INPUT_LENGTH, NUM_CLASSES, optimizer=1, layer1=256, layer2=128):
    # Create model
    model = Sequential()
    model.add(Dense(layer1, input_dim=INPUT_LENGTH, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(layer2, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(layer3, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    # Compile model
    SGD = optimizers.SGD()
    Adadelta = optimizers.Adadelta()
    Adagrad = optimizers.Adagrad()
    options = [SGD, Adadelta, Adagrad]
    model.compile(loss='categorical_crossentropy', optimizer=options[optimizer], metrics=['accuracy'])
    return model


for constant in [1]:
    coeffs_vec = get_WPD_coeffs(svm_data, levels=[6, 5, 4, 4, 4], wavelet='db10', extension_mode='zero')
    coeffs_vec = np.abs(coeffs_vec) ** 2
    coeffs_vec = coeffs_vec/np.max(coeffs_vec, axis=0)
    coeffs_vec = coeffs_vec - np.mean(coeffs_vec, axis=0)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    skf.get_n_splits(coeffs_vec, YL)

    scores = [[] for x in range(5)]
    train_acc = [[] for x in range(5)]
    test_acc = [[] for x in range(5)]
    for p, indices in enumerate(skf.split(coeffs_vec, YL)):
        x_train, y_train, x_validation, y_validation = get_kfolds(coeffs_vec, YL, indices)
        y_train = np_utils.to_categorical(y_train, num_classes=NUM_VOWELS)
        y_validation = np_utils.to_categorical(y_validation, num_classes=NUM_VOWELS)
        U, mu = PCA(x_train)  # Using 100-300 PCs is good (For both DWT and STFT) (210 for sym10)

        for n in range(1, 10):  # 70 or 90 for wavelet coeff training ?

            Z_train, Ul_train = apply_PCA_from_Eig(x_train, U, n, mu)
            Z_validation, Ul_validation = apply_PCA_from_Eig(x_validation, U, n, mu)
            model = baseline_model(Z_train.shape[1], NUM_VOWELS, optimizer=1, layer1=64, layer2=32)  # 1 best
            history = model.fit(x=Z_train, y=y_train, validation_data=(Z_validation, y_validation),
                                shuffle=True, batch_size=256, epochs=50, verbose=0)
            train_acc[p] = history.history['accuracy']
            test_acc[p] = history.history['val_accuracy']
            score = np.max(test_acc[p][-5:])
            scores[p].append(score)
        print('Finished fold %d' % (p+1))
    scores_mean = np.mean(np.asarray(scores), axis=0)
    print(scores_mean)













##############################################################################################################
##############################################################################################################
##############################################################################################################
##################### Everything below these lines is just result recording and plotting #####################
##############################################################################################################
##############################################################################################################
##############################################################################################################


















dwt_pca_data = np.zeros((4, 49))  # 6544, 66666, 44444, 55555
dwt_pca_data[0, :] = np.asarray(str.split('0.72188636 0.7977299  0.80594777 0.80341413 0.80973925 0.80594378 0.80656471 0.80909835 0.81289183 0.81100707 0.8078485  0.80342611 0.80531885 0.80657868 0.80847942 0.81101106 0.80848341 0.80848141 0.81038414 0.81038214 0.8078505  0.80279719 0.8027912  0.80026954 0.80089446 0.80152737 0.80090245 0.79710698 0.79836881 0.79521024 0.79773989 0.79520824 0.79520425 0.79710298 0.79077387 0.78761929 0.79330551 0.79394042 0.79204169 0.79077786 0.79014295 0.79267061 0.79267061 0.79393044 0.79519227 0.79076588 0.7913968  0.79202971 0.79265863'), dtype='float')
dwt_pca_data[1, :] = np.asarray(str.split('0.71871581 0.76929082 0.78193308 0.78129617 0.78381983 0.78824222 0.79140478 0.79330352 0.79330152 0.79456535 0.79582119 0.78886515 0.78824222 0.79204169 0.79203969 0.79267061 0.79393044 0.79583117 0.79455936 0.797091   0.79582917 0.79268259 0.79647007 0.79393443 0.79141477 0.78825021 0.7882542  0.79330951 0.78698838 0.78698439 0.78446073 0.78509364 0.7831989  0.78635946 0.78509564 0.78572056 0.78256399 0.78383181 0.78508565 0.78698039 0.78318892 0.78761929 0.78508565 0.78634748 0.78382183 0.78065927 0.78002436 0.78192709 0.78382382'), dtype='float')
dwt_pca_data[2, :] = np.asarray(str.split('0.72503494 0.74526614 0.74400232 0.74589905 0.74399832 0.7427305 0.73514954 0.73704229 0.72629877 0.72439804 0.72693168 0.72567384 0.72314819 0.72187637 0.71934473 0.71745198 0.7187198  0.72124745 0.71998962 0.72820748 0.72693168 0.72125145 0.71493232 0.71303758 0.7143014  0.71619614 0.71176576 0.71176776 0.70860919 0.70481172 0.70733139 0.70545062 0.70987302 0.70544863 0.70544464 0.70734137 0.70544863 0.70671645 0.7048237  0.70355389 0.70671844 0.70735136 0.70735335 0.70229405 0.70103023 0.70292697 0.70483169 0.70357385 0.70672843'), dtype='float')
dwt_pca_data[3, :] = np.asarray(str.split('0.72377511 0.78698838 0.78951803 0.79521024 0.79836282 0.80531885 0.80214631 0.80467596 0.8008705  0.80151739 0.7977339  0.79519626 0.7920377  0.7876153  0.78571457 0.78887713 0.78635547 0.77939943 0.78698039 0.78382183 0.78382183 0.7819171  0.77939145 0.77496706 0.77812762 0.76864992 0.76864393 0.76548936 0.76801302 0.76675318 0.76169189 0.76674919 0.76674719 0.76484846 0.76358863 0.76675119 0.7673861  0.76612027 0.76801102 0.76674919 0.7667432  0.7673821 0.76422154 0.75979715 0.75852933 0.76106098 0.76106098 0.75916623 0.76168989'), dtype='float')
dwt_pca_no_wpt = [0.70229206, 0.72504692, 0.71366649, 0.71303758, 0.70607156,
       0.69595296, 0.69911153, 0.69911352, 0.69911752, 0.69279839,
       0.68837799, 0.69154055, 0.68774508, 0.68333267, 0.67763647,
       0.68143194, 0.67827137, 0.67763048, 0.67763247, 0.67446792,
       0.6782574 , 0.68141996, 0.67320409, 0.67131134, 0.6694186 ,
       0.67384099, 0.66499221, 0.66940862, 0.66877571, 0.67004353,
       0.66751787, 0.66182566, 0.67004552, 0.67130735, 0.66878369,
       0.66499022, 0.66434333, 0.66877571, 0.66751587, 0.66561514,
       0.65929601, 0.65803218, 0.66246256, 0.66309548, 0.66372839,
       0.66626003, 0.66563111, 0.66056982, 0.6599409]


dwt_pca_short = np.zeros((4, 9))
dwt_pca_short[0, :] = np.asarray(str.split('0.30594977 0.36978597 0.46775346 0.5455217  0.63524538 0.67948928 0.69719283 0.702905   0.70544863'), dtype='float')
dwt_pca_short[1, :] = np.asarray(str.split('0.26736613 0.34702911 0.42794993 0.4841872  0.58784491 0.63651919 0.65989897 0.69469113 0.71048596'), dtype='float')
dwt_pca_short[2, :] = np.asarray(str.split('0.25853332 0.32554606 0.44565148 0.54235914 0.63148984 0.64474903 0.64982231 0.67698359 0.71175578'), dtype='float')
dwt_pca_short[3, :] = np.asarray(str.split('0.31160604 0.37484527 0.44184802 0.51137643 0.59985026 0.66242463 0.69530408 0.70860121 0.72188436'), dtype='float')

stft_pca_data = np.zeros((3, 49))  # winlen 128, 256, 64
stft_pca_data[0, :] = np.asarray(str.split('0.7541289  0.80027353 0.81544344 0.81859202 0.82743282 0.82490916 0.82553608 0.81669928 0.8223815  0.81984387 0.82362936 0.82363535 0.82236952 0.82679591 0.82805774 0.82426626 0.82869664 0.82173861 0.8267999  0.82363934 0.82679791 0.82996047 0.82806573 0.82932756 0.82932556 0.826161   0.82552809 0.82173462 0.82300244 0.82426427 0.82552809 0.82553009 0.82743282 0.82553208 0.82489518 0.82489318 0.82173462 0.82109971 0.82236353 0.82236353 0.82300044 0.82173262 0.82109971 0.81794513 0.81983788 0.81983788 0.81920696 0.81794513 0.81920497'), dtype='float')
stft_pca_data[1, :] = np.asarray(str.split('0.71176576 0.76992972 0.78320688 0.79963064 0.7977339  0.80783852 0.81352474 0.81731422 0.8109911  0.80784451 0.80594777 0.80341213 0.80719962 0.80276724 0.81035818 0.81225692 0.81352074 0.8065707 0.80846744 0.80972927 0.81225891 0.81415166 0.81478856 0.81288983 0.811628   0.80846943 0.80720561 0.80846943 0.80657469 0.80593779 0.80593779 0.80783452 0.80846744 0.80593779 0.80593779 0.80846744 0.80783452 0.80783253 0.80846145 0.80719962 0.80719962 0.80846344 0.80783253 0.8065687  0.80783452 0.8065707  0.8065687  0.80846344 0.80783053'), dtype='float')
stft_pca_data[2, :] = np.asarray(str.split('0.76423152 0.79582917 0.80783253 0.80025157 0.80150741 0.79708901 0.79456135 0.78949806 0.7920337  0.78887913 0.78698239 0.78761331 0.78888112 0.782558   0.782556   0.782556   0.79077587 0.78824422 0.78571657 0.7869784  0.78065727 0.78192509 0.78128619 0.78571657 0.78508765 0.7819211  0.77939145 0.78002436 0.78318692 0.78192309 0.78128818 0.78128818 0.78065727 0.78318892 0.78255201 0.782556 0.78192709 0.78192309 0.78192309 0.78066126 0.78192709 0.78065927 0.78382183 0.78318892 0.78318692 0.78255401 0.78444675 0.78318093 0.78507767'), dtype='float')

stft_pca_short = np.zeros((4, 9))
stft_pca_short[0, :] = np.asarray(str.split('0.29518029 0.40897656 0.51644971 0.55565228 0.67002556 0.69026275 0.70480573 0.71682306 0.73579843'), dtype='float')
stft_pca_short[1, :] = np.asarray(str.split('0.30149543 0.33626363 0.43995328 0.44122709 0.53288544 0.5682586 0.62578365 0.68963183 0.69849459'), dtype='float')
stft_pca_short[2, :] = np.asarray(str.split('0.2509304  0.37106577 0.44627241 0.58977758 0.67953121 0.69659186 0.72188835 0.74211157 0.73958591'), dtype='float')
stft_pca_short[3, :] = np.asarray([0.25472787, 0.31037615, 0.47473146, 0.50129777, 0.62576568,
       0.66180569, 0.67697361, 0.70922413, 0.73324682])

window_pca = np.asarray([0.81036417, 0.82363734, 0.80592181, 0.82679391, 0.82237352, 0.82363934, 0.79834085, 0.79391846, 0.82742483])
wavelet_pca = np.asarray([0.73577247, 0.78128818, 0.81038214, 0.80217426, 0.73577247, 0.76801102, 0.78889111, 0.78823823, 0.62326598])
wavelet_pca_names = ['db3', 'db5', 'db10', 'db20', 'sym3', 'sym5', 'sym10', 'sym20', 'haar']

WPT_PCA_NN = np.asarray([0.7383181 , 0.79076989, 0.80532883, 0.81289184, 0.81985185,
       0.82048875, 0.82805973, 0.82236754, 0.81858804, 0.82238951,
       0.83123627, 0.82554407, 0.82806773, 0.82047678, 0.81415565,
       0.81099907, 0.81478858, 0.8166853 , 0.81226691, 0.81605041,
       0.81352276, 0.81353871, 0.81289383, 0.81101305, 0.80404704,
       0.81225892, 0.80467397, 0.79266661, 0.80276724, 0.79836082,
       0.79265664, 0.80530887, 0.79202172, 0.78886516, 0.78697042,
       0.7920397 , 0.7857006 , 0.78569661, 0.79772191, 0.78507168,
       0.79076189, 0.78064131, 0.77685581, 0.79140279, 0.77749271,
       0.78696842, 0.78065927, 0.77557402, 0.78570659])
WPT_PCA_NN_short = np.asarray([0.32047278, 0.39379268, 0.48102065, 0.56891745, 0.65611948,
       0.70226011, 0.71238469, 0.71302561, 0.72313421])
STFT_PCA_NN = np.asarray([0.77497903, 0.80469593, 0.82996646, 0.84197179, 0.840698  ,
       0.85018367, 0.84641216, 0.84576328, 0.84008505, 0.8463982 ,
       0.85398313, 0.84322764, 0.83691452, 0.83248613, 0.83754342,
       0.83060735, 0.82490717, 0.83121831, 0.83058339, 0.83501378,
       0.82616102, 0.83184522, 0.83056942, 0.82490118, 0.82490116,
       0.82554607, 0.82299844, 0.82995447, 0.82554208, 0.81792916,
       0.82300444, 0.81921097, 0.81161803, 0.80971928, 0.82109771,
       0.81223495, 0.8166873 , 0.81225692, 0.82110769, 0.81542747,
       0.81288384, 0.82235955, 0.81667334, 0.81352874, 0.81983988,
       0.81920698, 0.8179651 , 0.81162602, 0.81604841])
STFT_PCA_NN_short = np.asarray([0.29896978, 0.41720641, 0.52971289, 0.57839117, 0.68960788,
       0.71048996, 0.72945133, 0.74401031, 0.76487042])


wpt_twohun_train = np.asarray([0.10729358, 0.16054536, 0.22075455, 0.284447  , 0.35050884,
       0.39808005, 0.44738302, 0.4900485 , 0.5371362 , 0.5837495 ,
       0.63479215, 0.6738279 , 0.7114411 , 0.7465238 , 0.76675355,
       0.78729546, 0.80041295, 0.8147936 , 0.8317007 , 0.8399188 ,
       0.85193026, 0.86077803, 0.87168294, 0.87879306, 0.8824288 ,
       0.88543147, 0.89412296, 0.89965343, 0.9066064 , 0.914192  ,
       0.9175097 , 0.92240936, 0.92730826, 0.92873013, 0.9366318 ,
       0.9380534 , 0.9404236 , 0.94358444, 0.94800884, 0.9511695 ,
       0.9541723 , 0.9552776 , 0.95954543, 0.9612838 , 0.96191597,
       0.96507645, 0.96823674, 0.9691846 , 0.9706069 , 0.9737676 ])
wpt_twohun_test = np.asarray([0.13846783, 0.19408617, 0.24844667, 0.29330751, 0.34197181,
       0.39380266, 0.43553887, 0.46777343, 0.51014256, 0.54869226,
       0.58661103, 0.62199616, 0.6655872 , 0.68520346, 0.70350996,
       0.71109891, 0.72058858, 0.73006029, 0.74018089, 0.7471449 ,
       0.75536277, 0.76040809, 0.77305435, 0.77559597, 0.78064328,
       0.78570259, 0.7819151 , 0.78696642, 0.78886914, 0.79898175,
       0.8015074 , 0.79707105, 0.79898375, 0.80594777, 0.80466998,
       0.80656472, 0.80593381, 0.80973327, 0.81162601, 0.81099708,
       0.81162202, 0.81794914, 0.81099111, 0.81225692, 0.81541948,
       0.81668332, 0.81668332, 0.81353074, 0.8211097 , 0.8160524 ])
stft_twohun_train = np.asarray([0.08865194, 0.15501964, 0.2435131 , 0.33833233, 0.4089728 ,
       0.4710802 , 0.5172249 , 0.560682  , 0.60445493, 0.64269865,
       0.68552244, 0.7204489 , 0.75505805, 0.77797383, 0.79709405,
       0.81084216, 0.8250653 , 0.8372332 , 0.84608364, 0.8569873 ,
       0.8664688 , 0.8748434 , 0.8805334 , 0.8879608 , 0.89680994,
       0.90076095, 0.90613395, 0.91119033, 0.9167205 , 0.921937  ,
       0.92430604, 0.9288891 , 0.9303113 , 0.9402658 , 0.9415307 ,
       0.94706184, 0.9476932 , 0.9508532 , 0.9525916 , 0.9551209 ,
       0.95828086, 0.96128356, 0.96270627, 0.9647597 , 0.9657081 ,
       0.9676043 , 0.96965873, 0.97218704, 0.97566354, 0.97645366])
stft_twohun_test = np.asarray([0.12012539, 0.20165915, 0.28447071, 0.34072595, 0.39888592,
       0.45326438, 0.50444835, 0.54425987, 0.57396877, 0.62768638,
       0.65991495, 0.69026674, 0.70733538, 0.72313422, 0.73260992,
       0.74209359, 0.75536876, 0.76106695, 0.76611627, 0.77118356,
       0.77434013, 0.78570659, 0.78381783, 0.78318892, 0.79646409,
       0.79392645, 0.80026354, 0.80025556, 0.80152138, 0.8021503 ,
       0.80531286, 0.80277722, 0.80278322, 0.80404505, 0.80910035,
       0.81479656, 0.81542947, 0.81794713, 0.81605839, 0.82300444,
       0.81731622, 0.82047879, 0.81668531, 0.82111768, 0.81921295,
       0.82174261, 0.82174659, 0.82426826, 0.82553407, 0.82489518])

wpt_full_train = np.asarray([0.1856872 , 0.37199733, 0.42983946, 0.48815003, 0.5567357 ,
       0.6256324 , 0.68947566, 0.7413067 , 0.7879243 , 0.8211088 ,
       0.8381777 , 0.8584057 , 0.8795807 , 0.89064294, 0.9073923 ,
       0.9198781 , 0.9282543 , 0.9415282 , 0.9487974 , 0.9587532 ,
       0.96633804, 0.97250223, 0.9758204 , 0.9807197 , 0.9832486 ,
       0.98672503, 0.9898856 , 0.9916242 , 0.9930464 , 0.99336255,
       0.99494267, 0.99573296, 0.9966814 , 0.99668103, 0.9973135 ,
       0.9974715 , 0.99826163, 0.9984196 , 0.9985777 , 0.99826163,
       0.9985777 , 0.9987357 , 0.99936783, 0.99920976, 0.99952585,
       0.9995259 , 0.999684  , 0.99984187, 1.        , 1.        ])
wpt_full_test = np.asarray([0.31286986, 0.37799385, 0.42350358, 0.48669887, 0.56004273,
       0.61251248, 0.65801421, 0.68899493, 0.70921214, 0.72691171,
       0.73765124, 0.74460528, 0.7553448 , 0.76293974, 0.76039612,
       0.77052668, 0.78000041, 0.77937348, 0.77682985, 0.78821428,
       0.79327558, 0.79390649, 0.8002336 , 0.79390649, 0.79770195,
       0.80844748, 0.80275726, 0.80338619, 0.80591184, 0.80718764,
       0.80276326, 0.80654675, 0.80023559, 0.80592581, 0.80591782,
       0.80592183, 0.80718964, 0.80718164, 0.80591981, 0.804656  ,
       0.80908438, 0.80718764, 0.80782056, 0.80655673, 0.80529889,
       0.80719165, 0.80592382, 0.80719163, 0.80592781, 0.80213633])
stft_full_train = np.asarray([0.24067362, 0.4364788 , 0.5069585 , 0.57032454, 0.6387435 ,
       0.69468296, 0.7409865 , 0.77227944, 0.79898906, 0.8179518 ,
       0.8367583 , 0.8500317 , 0.86488706, 0.8712076 , 0.88606197,
       0.89348996, 0.9077109 , 0.91371644, 0.92051184, 0.9260435 ,
       0.9325217 , 0.94058067, 0.94627094, 0.9510118 , 0.95701635,
       0.96049196, 0.9623885 , 0.96823615, 0.9728187 , 0.97503126,
       0.9804041 , 0.9804047 , 0.9838809 , 0.9860934 , 0.98546183,
       0.98877966, 0.9898863 , 0.9911504 , 0.99146634, 0.9932045 ,
       0.99431103, 0.9946269 , 0.99446887, 0.9957331 , 0.9963652 ,
       0.99683875, 0.9969975 , 0.99747145, 0.9971552 , 0.99762946])
stft_full_test = np.asarray([0.39508246, 0.46585472, 0.53727988, 0.59293015, 0.6454059 ,
       0.68015813, 0.71365451, 0.73703829, 0.74335942, 0.75409895,
       0.76483449, 0.77685181, 0.78951005, 0.79835882, 0.80277324,
       0.8109911 , 0.81226292, 0.81984386, 0.81921495, 0.81920696,
       0.82363335, 0.82237751, 0.82047679, 0.83059738, 0.83565068,
       0.83375993, 0.83184922, 0.83945214, 0.84070798, 0.84007506,
       0.84071198, 0.83375195, 0.83565468, 0.84070399, 0.83564467,
       0.83944014, 0.8419678 , 0.83564267, 0.83816634, 0.84006706,
       0.84449546, 0.84196582, 0.8413309 , 0.84196181, 0.84386654,
       0.84260671, 0.84196581, 0.84259872, 0.84765601, 0.84259472])

wpt_full_six_train = np.asarray([0.17903617, 0.3778329 , 0.45637664, 0.50188816, 0.5485093 ,
       0.59750146, 0.6600843 , 0.72534716, 0.7675396 , 0.8030942 ,
       0.8305933 , 0.8519276 , 0.87199575, 0.88859034, 0.8996499 ,
       0.9154536 , 0.9230407 , 0.93394506, 0.94437456, 0.95227575,
       0.9606513 , 0.9655508 , 0.97250366, 0.97519004, 0.9807213 ,
       0.98498774, 0.9890965 , 0.9898863 , 0.99194086, 0.994627  ,
       0.9957331 , 0.99699754, 0.9968394 , 0.9973135 , 0.9981035 ,
       0.99794567, 0.9982618 , 0.9979456 , 0.99889374, 0.9988936 ,
       0.99920976, 0.9993679 , 0.99905175, 0.9993679 , 0.999684  ,
       0.99952585, 0.99952585, 1.        , 1.        , 1.        ])
wpt_full_six_test = np.asarray([0.29774987, 0.39888192, 0.45198058, 0.50382141, 0.54363893,
       0.58281555, 0.63211476, 0.69090165, 0.71744998, 0.72312222,
       0.7433694 , 0.74778579, 0.74654193, 0.76549735, 0.76801699,
       0.76865392, 0.77750468, 0.78381783, 0.78571856, 0.7869844 ,
       0.78761531, 0.79329952, 0.79519826, 0.79583317, 0.80152538,
       0.80025556, 0.80405502, 0.80342011, 0.80404904, 0.80467795,
       0.80783852, 0.80594778, 0.79961666, 0.8072116 , 0.80973924,
       0.81290182, 0.81163998, 0.80973526, 0.80404705, 0.8065787 ,
       0.81352673, 0.80657669, 0.80594177, 0.80658268, 0.80531485,
       0.80911034, 0.80657469, 0.80784451, 0.80847543, 0.80910634])
wpt_full_four_train = np.asarray([0.15566042, 0.32458198, 0.40516883, 0.4626864 , 0.5282694 ,
       0.57946664, 0.63540864, 0.697673  , 0.73528636, 0.7803295 ,
       0.80909187, 0.83848846, 0.86440766, 0.8881136 , 0.90897405,
       0.9246187 , 0.9386836 , 0.94974613, 0.95717394, 0.96697176,
       0.9740822 , 0.98119307, 0.9834054 , 0.98672485, 0.98956907,
       0.99225605, 0.99320424, 0.9951007 , 0.99620694, 0.9963649 ,
       0.9973134 , 0.99810374, 0.9985777 , 0.9985777 , 0.9988939 ,
       0.99921   , 0.9995259 , 0.99968404, 0.99968404, 0.99984205,
       0.99984205, 0.99984205, 0.99984205, 0.99984205, 0.99984205,
       0.99984205, 0.99984205, 1.        , 1.        , 1.        ])
wpt_full_four_test = np.asarray([0.26294774, 0.33378589, 0.39574931, 0.45452022, 0.49753024,
       0.54810326, 0.58350637, 0.61699278, 0.64921135, 0.66500421,
       0.68775307, 0.68015214, 0.69786167, 0.70038334, 0.71176775,
       0.70986303, 0.71427743, 0.72187238, 0.72187437, 0.72439604,
       0.7275586 , 0.72629677, 0.72566186, 0.72630275, 0.72061255,
       0.72565588, 0.72186639, 0.72313421, 0.72376912, 0.71997565,
       0.72377112, 0.72313421, 0.72692968, 0.7231362 , 0.72629877,
       0.72376912, 0.72503694, 0.72566785, 0.72061255, 0.72503693,
       0.72756059, 0.72503494, 0.72440202, 0.72439204, 0.72376512,
       0.72819149, 0.72439604, 0.72376311, 0.72503096, 0.7224993 ])
wpt_full_five_train = np.asarray([0.18773828, 0.36441275, 0.45101362, 0.52244407, 0.57396257,
       0.63464844, 0.68964183, 0.741314  , 0.7803482 , 0.81084377,
       0.83359754, 0.85540515, 0.87515813, 0.8898547 , 0.90786904,
       0.92367375, 0.9315735 , 0.9459546 , 0.9506947 , 0.95764846,
       0.96902645, 0.9728192 , 0.9783497 , 0.98135215, 0.98482955,
       0.98814833, 0.99051875, 0.99162483, 0.9935215 , 0.9949436 ,
       0.99573344, 0.99636567, 0.9966816 , 0.9971555 , 0.9973135 ,
       0.9977876 , 0.99841976, 0.99841976, 0.9987356 , 0.9987358 ,
       0.99905175, 0.9988939 , 0.9992099 , 0.9993679 , 0.9995259 ,
       0.99984187, 0.99984187, 0.99984187, 0.99984187, 1.        ])
wpt_full_five_test = np.asarray([0.3015094 , 0.39948289, 0.47406461, 0.52335981, 0.57202212,
       0.62067444, 0.66175778, 0.68200895, 0.70351596, 0.72312022,
       0.73638542, 0.74904363, 0.75031544, 0.7528411 , 0.7616879 ,
       0.76295571, 0.77180849, 0.7718025 , 0.76990577, 0.77243741,
       0.77812761, 0.77623088, 0.77495908, 0.77053667, 0.77179251,
       0.77369924, 0.77307032, 0.77306434, 0.7717985 , 0.76928083,
       0.76990976, 0.76990576, 0.76990178, 0.76610829, 0.77495108,
       0.77306432, 0.76800305, 0.76736414, 0.76673921, 0.76926686,
       0.77116359, 0.76926886, 0.76863196, 0.76231282, 0.76358064,
       0.7667432 , 0.7616819 , 0.76674122, 0.76484646, 0.76484847])
wpt_full_best_train = np.asarray([0.17967618, 0.36884126, 0.44232732, 0.5072726 , 0.57443607,
       0.64491487, 0.6984878 , 0.746845  , 0.78824204, 0.8195324 ,
       0.84007615, 0.86156845, 0.88147974, 0.8969674 , 0.9138776 ,
       0.92240983, 0.9363162 , 0.9459553 , 0.9559104 , 0.9622323 ,
       0.96760464, 0.9748745 , 0.9772445 , 0.9826171 , 0.98641014,
       0.988622  , 0.9900449 , 0.99099237, 0.9928889 , 0.9952594 ,
       0.99557525, 0.9962074 , 0.99636537, 0.99699724, 0.9973136 ,
       0.9973138 , 0.9979458 , 0.9981038 , 0.99841994, 0.9985779 ,
       0.9985779 , 0.99905187, 0.99921   , 0.9993679 , 0.9993679 ,
       0.9993679 , 0.999526  , 0.99984205, 1.        , 0.99968404])
wpt_full_best_test = np.asarray([0.30025555, 0.3899992 , 0.43740367, 0.49682145, 0.55371761,
       0.60236791, 0.65229805, 0.67632474, 0.70414287, 0.7155273 ,
       0.73068922, 0.74334943, 0.75977519, 0.76103501, 0.77368326,
       0.77685182, 0.77496307, 0.78696641, 0.79201374, 0.79329153,
       0.78506171, 0.79454938, 0.79771394, 0.79834087, 0.80340414,
       0.80783453, 0.80340813, 0.80719763, 0.80150743, 0.80782654,
       0.80466598, 0.80909436, 0.81035819, 0.80719563, 0.80909238,
       0.80846545, 0.8103582 , 0.80656271, 0.80782455, 0.80782855,
       0.81035019, 0.80719364, 0.80845946, 0.80782655, 0.80719763,
       0.80845946, 0.80782255, 0.80719563, 0.80782454, 0.80909238])


plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(10, 500, 10), dwt_pca_data[0, :], label='65444')
plt.plot(np.arange(10, 500, 10), dwt_pca_data[1, :], label='66666')
plt.plot(np.arange(10, 500, 10), dwt_pca_data[2, :], label='44444')
plt.plot(np.arange(10, 500, 10), dwt_pca_data[3, :], label='55555')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('SVM Performance vs WPT Nodes Used', fontsize=18)
plt.savefig('SVM_PCA_WPT.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(10, 500, 10), dwt_pca_data[0, :], label='WPT')
plt.plot(np.arange(10, 500, 10), dwt_pca_no_wpt, label='DWT')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('SVM Performance, DWT vs WPT', fontsize=18)
plt.savefig('SVM_PCA_WPT_DWT.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(1, 10), dwt_pca_short[0, :], label='65444')
plt.plot(np.arange(1, 10), dwt_pca_short[1, :], label='66666')
plt.plot(np.arange(1, 10), dwt_pca_short[2, :], label='44444')
plt.plot(np.arange(1, 10), dwt_pca_short[3, :], label='55555')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('SVM Performance vs WPT Nodes Used', fontsize=18)
plt.savefig('SVM_PCA_WPT_short.png', dpi=600)


plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(10, 500, 10), stft_pca_data[0, :], label='128')
plt.plot(np.arange(10, 500, 10), stft_pca_data[1, :], label='256')
plt.plot(np.arange(10, 500, 10), stft_pca_data[2, :], label='64')
plt.legend(title='win_len', loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('SVM Performance vs STFT Window Length', fontsize=18)
plt.savefig('SVM_PCA_STFT.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(1, 10), stft_pca_short[0, :], label='128')
# plt.plot(np.arange(1, 10), stft_pca_short[1, :], label='256')
plt.plot(np.arange(1, 10), stft_pca_short[2, :], label='64')
plt.plot(np.arange(1, 10), stft_pca_short[3, :], label='32')
plt.legend(title='win_len', loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('SVM Performance vs STFT Window Length', fontsize=18)
plt.savefig('SVM_PCA_STFT_short.png', dpi=600)


plt.figure(dpi=300, figsize=(8, 5))
plt.bar(x=np.arange(len(window_options)), height=window_pca, tick_label=window_options)
plt.ylim((0.5, 1))
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.title('SVM Performance vs STFT Window', fontsize=18)
plt.tight_layout()
plt.savefig('STFT_PCA_WINDOWS.png', dpi=600)

plt.figure(dpi=300, figsize=(8, 5))
plt.bar(x=np.arange(len(wavelet_pca_names)), height=wavelet_pca, tick_label=wavelet_pca_names)
plt.ylim((0.5, 1))
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.title('SVM Performance vs WPT Wavelet Fn', fontsize=18)
plt.tight_layout()
plt.savefig('WPT_PCA_WINDOWS.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(10, 500, 10), dwt_pca_data[0, :], label='WPT')
plt.plot(np.arange(10, 500, 10), stft_pca_data[0, :], label='STFT')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('WPT vs STFT, SVM Performance', fontsize=18)
plt.savefig('SVM_STFT_PCA_WPT.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(1, 10), dwt_pca_short[0, :], label='WPT')
plt.plot(np.arange(1, 10), stft_pca_short[0, :], label='STFT')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('WPT vs STFT, SVM Performance', fontsize=18)
plt.savefig('SVM_STFT_PCA_WPT_short.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(10, 500, 10), WPT_PCA_NN, label='WPT')
plt.plot(np.arange(10, 500, 10), STFT_PCA_NN, label='STFT')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('WPT vs STFT, NN Performance', fontsize=18)
plt.savefig('NN_STFT_PCA_WPT.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(1, 10), WPT_PCA_NN_short, label='WPT')
plt.plot(np.arange(1, 10), STFT_PCA_NN_short, label='STFT')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('# of Principal Components', fontsize=16)
plt.title('WPT vs STFT, NN Performance', fontsize=18)
plt.savefig('NN_STFT_PCA_WPT_short.png', dpi=600)


plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(0, 50), wpt_full_six_train, label='train')
plt.plot(np.arange(0, 50), wpt_full_six_test, label='test')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim((0, 1))
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.title('Training Curves for Layer 6 WPT Data', fontsize=18)
plt.savefig('nn_curves_layer6.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(0, 50), wpt_full_five_train, label='train')
plt.plot(np.arange(0, 50), wpt_full_five_test, label='test')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim((0, 1))
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.title('Training Curves for Layer 5 WPT Data', fontsize=18)
plt.savefig('nn_curves_layer5.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(0, 50), wpt_twohun_train, label='train')
plt.plot(np.arange(0, 50), wpt_twohun_test, label='test')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim((0, 1))
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.title('Training Curves for Layer 4 WPT Data', fontsize=18)
# plt.savefig('nn_curves_layer4.png', dpi=600)

plt.figure(dpi=300, figsize=(10, 6))
plt.plot(np.arange(0, 50), stft_twohun_train, label='train')
plt.plot(np.arange(0, 50), stft_twohun_test, label='test')
plt.legend(loc='lower left', fancybox=True, shadow=True, fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim((0, 1))
plt.ylabel('Classification Accuracy', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.title('Training Curves for Mixed Layer WPT Data', fontsize=18)
# plt.savefig('nn_curves_layer_all.png', dpi=600)
