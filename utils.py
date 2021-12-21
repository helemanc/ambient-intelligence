from voice_activity_detection import vad
from speech_emotion_recognition import feature_extraction as fe, ensemble
import scipy
import numpy as np
from scipy import signal
from scipy.io.wavfile import write
import datetime

def denoise(samples):
    """
    :param samples: an array representing the sampled audio file
    :type samples: float
    :return: an array representing the clean audio file
    :rtype: float
    """
    samples_wiener = scipy.signal.wiener(samples)
    return samples_wiener


def execute_vad_ser(segmenter, filepath, prediction_scheme):
    """
    :param segmenter: an instance of the Segmenter object of Ina Speech Segementer
    :type segmenter: Segmenter object
    :param filepath: a string representing the full-path of the input audio file
    :type filepath: string
    :param prediction_scheme: a string representing the aggregation strategy to be used in the ensemble
    :type prediction_scheme: string

    TO DO
    :return: an integer flag representing the action to be sent to the control center
    :rtype: int
    """

    samples, sample_rate = fe.read_file(filepath)
    samples, sample_rate = resample(samples, sample_rate)
    new_samples = fe.cut_pad(samples)
    samples_wiener = denoise(new_samples)
    new_filepath = "voice_activity_detection/tmp.wav"
    write(new_filepath, sample_rate, samples_wiener)

    seg = vad.check_speech(segmenter, new_filepath)
    if seg == 'speech':
        print("#########################################\n")
        print("The audio contains speech. \nStarting Speech Emotion Recognition process.\n")
        print("#########################################\n")

        #start_time = datetime.datetime.now()

        final_prediction = ensemble.ensemble(new_samples, prediction_scheme)

        #elapsed = datetime.datetime.now() - start_time
        #print("Time elapsed for SER", elapsed)

        if final_prediction == 1:
            print("Speech contains disruptive emotion.")
        else:
            print("Speech does not contain disruptive emotion.")

def resample(input_data, sample_rate, required_sample_rate=16000, amplify=False):
    """
    Resampling function. Takes an audio with sample_rate and resamples
    it to required_sample_rate. Optionally, amplifies volume.
    ​
    :param input_data: Input audio data
    :type input_data: numpy.ndarray
    :param sample_rate: Sample rate of input audio data
    :type sample_rate: int
    :param required_sample_rate: Sample rate to resample original audio, defaults to 16000
    :type required_sample_rate: int, optional
    :param amplify: Wether to amplify audio volume, defaults to False
    :type amplify: bool, optional
    :return: Resampled audio and new sample rate
    :rtype: numpy.ndarray, int
    """
    if sample_rate < required_sample_rate:
        resampling_factor = int(round(required_sample_rate/sample_rate,0))
        new_rate = sample_rate * resampling_factor
        samples = len(input_data) * resampling_factor
        resampled = signal.resample(input_data, samples)
    elif sample_rate > required_sample_rate:
        resampling_factor = int(round(sample_rate/required_sample_rate,0))
        new_rate = int(sample_rate / resampling_factor)
        resampled = signal.decimate(input_data, resampling_factor)
    else:
        resampling_factor = 1
        new_rate = sample_rate
        resampled = input_data
    if amplify and input_data.size > 0:
        absolute_values = np.absolute(resampled)
        max_value = np.amax(absolute_values)
        max_range = np.iinfo(np.int16).max
        amplify_factor = max_range/max_value
        resampled = resampled * amplify_factor
        resampled = resampled.round()
    return resampled.astype(np.float32), new_rate

