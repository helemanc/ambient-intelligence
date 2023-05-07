import math
import librosa
import numpy as np
import pickle
import os

LENGTH_CHOSEN = 80000
#SCALERS_FOLDER = "speech_emotion_recognition/scalers/data_scaler"
#SCALERS_FOLDER = "speech_emotion_recognition/scalers/data_scaler_ravdess"
SCALERS_FOLDER = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scalers")))

def read_file(audio_file):
    """

    :param audio_file: a string representing the full-path of the input audio file
    :type audio_file: string
    :return: an array representing the sampled audio file, sample rate
    :rtype: np.array, int
    """
    samples, sr = librosa.load(audio_file, res_type='kaiser_fast', sr=16000)
    return samples, sr


def cut_pad(samples):
    """

![](../../../Library/Group Containers/group.com.apple.notes/Accounts/LocalAccount/Media/45B5498C-5174-4A3E-8F62-C5D85AB24B47/Pasted Graphic.png)    :param samples: an array representing the sampled audio
    :type samples: np.array, float64
    :return: an array representing the cut or padded audio file
    :rtype: np.array, float64
    """
    # cut
    if samples.shape[0] > LENGTH_CHOSEN:
        new_samples = samples[:LENGTH_CHOSEN]
    # pad
    elif samples.shape[0] < LENGTH_CHOSEN:
        new_samples = np.pad(samples, math.ceil((LENGTH_CHOSEN - samples.shape[0]) / 2), mode='median')
    # nothing
    else:
        new_samples = samples
    if len(new_samples) == 80001:
        new_samples = new_samples[:-1]
    return new_samples


def compute_energy(samples):
    """

    :param samples: an array representing the sampled audio
    :type samples: float64
    :return: a value representing the rms on the input audio
    :rtype: float
    """
    energy = librosa.feature.rms(samples)
    energy = energy.T
    energy = np.array(energy)
    return energy


def compute_energy_mean(samples):
    """


    :param samples: an array representing the sampled audio
    :type samples: float64
    :return: a value representing the average rms on the input audio
    :rtype: float
    """
    energy = librosa.feature.rms(samples)
    energy = energy.T
    energy = np.array(energy)
    energy = np.mean(energy, axis=0)
    return energy


def compute_mfccs(samples, n_mfcc, scaler, feature_energy):
    """

    :param samples: an array representing the sampled audio
    :type samples: np.array, float 64
    :param n_mfcc: the desired number of mfcc
    :type n_mfcc: int
    :param scaler: the full-path of the scaler pickle object
    :type scaler: string
    :param feature_energy: a value to indicate whether the energy feature should be concatenated to MFCCs
    :type feature_energy: boolean
    :return: an of audio features
    :rtype: np.array, float64
    """
    # Compute MFCCS
    mfccs = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    mfccs = np.array(mfccs)
    mfccs = mfccs[:, 1:]  # get rid of the first component

    # Compute energy, if required
    if feature_energy == True:
        energy = compute_energy(samples)
        features = []
        conc = np.column_stack((mfccs, energy))
        features.append(conc)
        mfccs = np.array(features)

    # Reshape features in (1, 157, n_mfcc)
    if feature_energy == True:
        mfccs = mfccs.reshape(1, 157, n_mfcc)
    else:
        mfccs = mfccs.reshape(1, 157, n_mfcc-1)

    # Load scaler
    SCALER_PATH = os.path.join(SCALERS_FOLDER, scaler)
    # print("SCALER PATH", SCALER_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # print("Shape MFCCS", mfccs.shape)
    # Scale data
    mfccs = scaler.transform(mfccs.reshape(-1, mfccs.shape[-1])).reshape(mfccs.shape)

    return mfccs


def compute_mfccs_mean(samples, n_mfcc, scaler, feature_energy):
    """

    :param samples: an array representing the sampled audio
    :type samples: np.array, float 64
    :param n_mfcc: the desired number of mfcc
    :type n_mfcc: int
    :param scaler: the full-path of the scaler pickle object
    :type scaler: string
    :param feature_energy: a value to indicate whether the energy feature should be concatenated to MFCCs
    :type feature_energy: boolean
    :return: an of audio features
    :rtype: np.array, float64
    """
    mfccs = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    mfccs = np.array(mfccs)
    mfccs = np.mean(mfccs[:, 1:], axis=0)
    # print("Shape MFCCS", mfccs.shape)

    # Compute energy, if required
    if feature_energy == True:
        energy = compute_energy_mean(samples)
        features = []
        conc = np.concatenate((mfccs, energy), axis=None)
        features.append(conc)
        mfccs = np.array(features)

    # Reshape features in (1, 157, n_mfcc)
    if feature_energy == True:
        mfccs = mfccs.reshape(1, n_mfcc)
    else:
        mfccs = mfccs.reshape(1, n_mfcc - 1)
    # Load scaler
    SCALER_PATH = os.path.join(SCALERS_FOLDER, scaler)
    # print("SCALER PATH", SCALER_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # Scale data
    mfccs = scaler.transform(mfccs.reshape(-1, mfccs.shape[-1])).reshape(mfccs.shape)

    return mfccs


def mfccs_scaled(samples, scaler, id_exp):
    """

    :param samples: an array representing the sampled audio
    :type samples: np.array, float 64
    :param scaler: the full-path of the scaler pickle object
    :type scaler: string
    :param id_exp: a string that identifies the experiment
    :type id_exp: string
    :return: an array of features
    :rtype: np.array, float64
    """
    parts = id_exp.split('_')
    num_exp = parts[0]
    # print("Computing features for Experiment: ", num_exp)
    if num_exp == '1':
        mfccs = compute_mfccs(samples, 13, scaler, feature_energy=False)
        return mfccs
    elif num_exp == '2':
        mfccs = compute_mfccs(samples, 13, scaler, feature_energy=True)
        return mfccs
    elif num_exp == '3':
        mfccs = compute_mfccs(samples, 26, scaler, feature_energy=False)
        return mfccs
    elif num_exp == '4':
        mfccs = compute_mfccs(samples, 26, scaler, feature_energy=True)
        return mfccs
    elif num_exp == '5':
        mfccs = compute_mfccs_mean(samples, 13, scaler, feature_energy=False)
        return mfccs
    elif num_exp == '6':
        mfccs = compute_mfccs_mean(samples, 13, scaler, feature_energy=True)
        return mfccs
    elif num_exp == '7':
        mfccs = compute_mfccs_mean(samples, 26, scaler, feature_energy=False)
        return mfccs
    elif num_exp == '8':
        mfccs = compute_mfccs_mean(samples, 26, scaler, feature_energy=True)
        return mfccs
