import pickle
import os
import tensorflow as tf
from speech_emotion_recognition import feature_extraction as fe, emotion_predictions as ep

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

LENGTH_CHOSEN = 80000
MODELS_FOLDER = "speech_emotion_recognition/models/ensemble_models"
SCALERS_FOLDER = "speech_emotion_recognition/data_scaler"

def ensemble_prediction_voting(predictions):
    # voting without weighting the model
    count_ones = 0
    count_zeros = 0
    for p in predictions:
        if p == 1:
            count_ones +=1
        else:
            count_zeros +=1
    if count_ones >= count_zeros:
        return 1
    else:
        return 0

def ensemble(samples):
    predictions = []
    for dirpath, dirnames, filenames in os.walk(MODELS_FOLDER):
        print('ciao')
        # iterate over the list of dirnames to load the convmodel
        # iterate over the list of filenames to get the svm model
        for model in dirnames:
            print("Loading model: ", model)
            parts = model.split('_')
            num_exp = parts[1]
            num_data = parts[2].split('.')[0]
            id_exp = num_exp + '_' + num_data
            model_type = 'conv'
            model_path = os.path.join(dirpath, model)
            conv_model = tf.keras.models.load_model(model_path)
            for scaler in os.listdir(SCALERS_FOLDER):
                parts = scaler.split('_')
                num_exp = parts[1]
                num_data = parts[2].split('.')[0]
                id_exp_scaler = num_exp + '_' + num_data
                if id_exp == id_exp_scaler:
                    print("Loading scaler: ", scaler)
                    scaler_conv_model = scaler
                    mfccs = fe.mfccs_scaled(samples, scaler_conv_model, id_exp)
                    predictions.append(ep.make_predictions(conv_model, model_type, mfccs))
        for model in filenames:
            print("Loading model: ", model)
            parts = model.split('_')
            num_exp = parts[1]
            num_data = parts[2].split('.')[0]
            id_exp = num_exp + '_' + num_data
            model_type = 'svm'
            model_path = os.path.join(dirpath, model)
            with open(model_path, 'rb') as file:
                svm_model = pickle.load(file)
            for scaler in os.listdir(SCALERS_FOLDER):
                parts = scaler.split('_')
                num_exp = parts[1]
                num_data = parts[2].split('.')[0]
                id_exp_scaler = num_exp + '_' + num_data
                if id_exp == id_exp_scaler:
                    print("Loading scaler: ", scaler)
                    scaler_svm_model = scaler
                    mfccs = fe.mfccs_scaled(samples, scaler_svm_model, id_exp)
                    predictions.append(ep.make_predictions(svm_model, model_type, mfccs))
        break
    print(predictions)
    final_prediction = ensemble_prediction_voting(predictions)

    return final_prediction
'''
#ensemble("media/03-01-01-01-01-01-01.wav")

samples, sample_rate = fe.read_file("03-01-01-01-01-01-01.wav")
samples, sample_rate = fe.resample(samples, sample_rate)
samples = fe.cut_pad(samples)
print(len(samples))
'''


'''
for dirpath, dirnames, filenames in os.walk(MODELS_FOLDER):
        #iterate over the list of dirnames to load the convmodel
        #iterate over the list of filenames to get the svm model
        predictions = []
        for model in dirnames:
            print("Loading model: ", model)
            parts = model.split('_')
            num_exp = parts[1]
            num_data = parts[2].split('.')[0]
            id_exp = num_exp + '_' + num_data
            model_type = 'conv'
            model_path = os.path.join(dirpath, model)
            conv_model = tf.keras.models.load_model(model_path)
            for scaler in os.listdir(SCALERS_FOLDER):
                parts = scaler.split('_')
                num_exp = parts[1]
                num_data = parts[2].split('.')[0]
                id_exp_scaler = num_exp + '_' + num_data
                if id_exp == id_exp_scaler:
                    print("Loading scaler: ", scaler)
                    scaler_conv_model = scaler
                    mfccs = fe.mfccs_scaled(samples, scaler_conv_model, id_exp)
                    predictions.append(ep.make_predictions(conv_model, model_type, mfccs))
        for model in filenames:
            print("Loading model: ", model)
            parts = model.split('_')
            num_exp = parts[1]
            num_data = parts[2].split('.')[0]
            id_exp = num_exp + '_' + num_data
            model_type = 'svm'
            model_path = os.path.join(dirpath, model)
            with open(model_path, 'rb') as file:
                svm_model = pickle.load(file)
            for scaler in os.listdir(SCALERS_FOLDER):
                parts = scaler.split('_')
                num_exp = parts[1]
                num_data = parts[2].split('.')[0]
                id_exp_scaler = num_exp + '_' + num_data
                if id_exp == id_exp_scaler:
                    print("Loading scaler: ", scaler)
                    scaler_svm_model = scaler
                    mfccs = fe.mfccs_scaled(samples, scaler_svm_model, id_exp)
                    predictions.append(ep.make_predictions(svm_model, model_type, mfccs))
        print(predictions)


        #print(dirnames)
        #print(filenames)
        break

'''











