import math
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

LENGTH_CHOSEN = 120378
SCALER_PATH = "speech_emotion_recognition/data_scaler/scaler.pickle"

def read_file(audio_file):
    samples, sr = librosa.load(audio_file, res_type='kaiser_fast', sr=44000)
    return samples, sr

def cut_pad(samples):
    #cut
    if samples.shape[0] > LENGTH_CHOSEN:
        new_samples = samples[:LENGTH_CHOSEN]
    #pad
    elif samples.shape[0] < LENGTH_CHOSEN:
        new_samples = np.pad(samples, math.ceil((LENGTH_CHOSEN - samples.shape[0]) / 2), mode='median')
    #nothing
    else:
        new_samples = samples
    return new_samples

def mfccs_scaled(samples):
    mfccs = librosa.feature.mfcc(y=samples, sr=44000, n_mfcc=40)
    mfccs = mfccs.T
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    mfccs = scaler.transform(mfccs.reshape(-1, mfccs.shape[-1])).reshape(mfccs.shape)
    mfccs = np.array(mfccs)
    mfccs = mfccs.reshape(1, 236, 40)
    return mfccs