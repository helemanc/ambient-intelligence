import math
import librosa
import numpy as np
import pickle
import os

LENGTH_CHOSEN = 80000
SCALERS_FOLDER = "speech_emotion_recognition/data_scaler"

def read_file(audio_file):
    """
    :param audio_file:
    :type audio_file:
    :return:
    :rtype:
    """
    samples, sr = librosa.load(audio_file, res_type='kaiser_fast', sr=16000)
    return samples, sr

def cut_pad(samples):
    """

    :param samples:
    :type samples:
    :return:
    :rtype:
    """
    #cut
    if samples.shape[0] > LENGTH_CHOSEN:
        new_samples = samples[:LENGTH_CHOSEN]
    #pad
    elif samples.shape[0] < LENGTH_CHOSEN:
        new_samples = np.pad(samples, math.ceil((LENGTH_CHOSEN - samples.shape[0]) / 2), mode='median')
    #nothing
    else:
        new_samples = samples
    if len(new_samples) == 80001:
        new_samples = new_samples[:-1]
    return new_samples


def compute_energy(samples):
    """

    :param samples:
    :type samples:
    :return:
    :rtype:
    """
    energy = librosa.feature.rms(samples)
    energy = energy.T
    energy = np.array(energy)
    return energy

def compute_energy_mean(samples):
    """

    :param samples:
    :type samples:
    :return:
    :rtype:
    """
    energy = librosa.feature.rms(samples)
    energy = energy.T
    energy = np.array(energy)
    energy = np.mean(energy, axis=0)
    return energy



def compute_mfccs(samples, n_mfcc, scaler, feature_energy):
    """

    :param samples:
    :type samples:
    :param n_mfcc:
    :type n_mfcc:
    :param scaler:
    :type scaler:
    :param feature_energy:
    :type feature_energy:
    :return:
    :rtype:
    """
    # Compute MFCCS
    mfccs = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    mfccs = np.array(mfccs)
    mfccs = mfccs[:, 1:] # get rid of the first component

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
    #print("SCALER PATH", SCALER_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    #print("Shape MFCCS", mfccs.shape)
    # Scale data
    mfccs = scaler.transform(mfccs.reshape(-1, mfccs.shape[-1])).reshape(mfccs.shape)

    return mfccs

def compute_mfccs_mean(samples, n_mfcc, scaler, feature_energy):
    mfccs = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=n_mfcc)
    mfccs = mfccs.T
    mfccs = np.array(mfccs)
    mfccs = np.mean(mfccs[:, 1:], axis=0)
    #print("Shape MFCCS", mfccs.shape)

    # Compute energy, if required
    if feature_energy == True:
        energy = compute_energy_mean(samples)
        features = []
        conc = np.concatenate((mfccs, energy), axis = None)
        features.append(conc)
        mfccs = np.array(features)

    # Reshape features in (1, 157, n_mfcc)
    if feature_energy == True:
        mfccs = mfccs.reshape(1, n_mfcc)
    else:
        mfccs = mfccs.reshape(1, n_mfcc - 1)
    # Load scaler
    SCALER_PATH = os.path.join(SCALERS_FOLDER, scaler)
    #print("SCALER PATH", SCALER_PATH)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    # Scale data
    mfccs = scaler.transform(mfccs.reshape(-1, mfccs.shape[-1])).reshape(mfccs.shape)

    return mfccs


def mfccs_scaled(samples, scaler, id_exp):
    """

    :param samples:
    :type samples:
    :param scaler:
    :type scaler:
    :param id_exp:
    :type id_exp:
    :return:
    :rtype:
    """
    parts = id_exp.split('_')
    num_exp = parts[0]
    #print("Computing features for Experiment: ", num_exp)
    if num_exp == '1':
        mfccs = compute_mfccs(samples, 13, scaler, feature_energy = False )
        return mfccs
    elif num_exp == '2':
        mfccs = compute_mfccs(samples, 13, scaler, feature_energy = True)
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






'''

    mfccs = librosa.feature.mfcc(y=samples, sr=16000, n_mfcc=40)
    mfccs = mfccs.T
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    mfccs = scaler.transform(mfccs.reshape(-1, mfccs.shape[-1])).reshape(mfccs.shape)
    mfccs = np.array(mfccs)
    mfccs = mfccs.reshape(1, 236, 40)
    return mfccs
'''