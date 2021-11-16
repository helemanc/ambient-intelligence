import pickle
import os
import tensorflow as tf
from speech_emotion_recognition import feature_extraction as fe, emotion_predictions as ep

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

LENGTH_CHOSEN = 80000
MODELS_FOLDER = "speech_emotion_recognition/models/ensemble_models"
SCALERS_FOLDER = "speech_emotion_recognition/data_scaler"

def ensemble_prediction_voting(predictions):
    """

    :param predictions:
    :type predictions:
    :return:
    :rtype:
    """
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
def ensemble_prediction_avg_1(predictions):
    """

    :param predictions:
    :type predictions:
    :return:
    :rtype:
    """
    threshold = 0.5
    pred_conv = [p for p in predictions if p != 0 and p!= 1 ]
    pred_svm = [p for p in predictions if p == 0 or p== 1 ]
    avg_prediction_conv = sum(pred_conv) / len(pred_conv)
    pred_svm.append(avg_prediction_conv)
    avg_prediction = sum(pred_svm) / len(pred_svm)
    return (1 * (avg_prediction >= threshold))

def ensemble_prediction_avg_2(predictions):
    """

    :param predictions:
    :type predictions:
    :return:
    :rtype:
    """
    threshold = 0.7
    avg_prediction = sum(predictions) / len(predictions)
    return (1 * (avg_prediction >= threshold))


def ensemble(samples, prediction_scheme, return_model_predictions = False):
    """
    prediction_scheme could be: majority, prob, w_prob
    :param samples: 
    :type samples: 
    :param return_model_predictions: 
    :type return_model_predictions: 
    :param prediction_scheme: 
    :type prediction_scheme: 
    :return: 
    :rtype: 
    """
    predictions = []
    model_predictions = {}
    for dirpath, dirnames, filenames in os.walk(MODELS_FOLDER):
        # iterate over the list of dirnames to load the convmodel
        # iterate over the list of filenames to get the svm model
        for model in dirnames:
            #print("Loading model: ", model)
            model_type = 'conv'
            parts = model.split('_')
            num_exp = parts[1]
            num_data = parts[2].split('.')[0]
            id_exp = num_exp + '_' + num_data
            model_path = os.path.join(dirpath, model)
            conv_model = tf.keras.models.load_model(model_path)
            for scaler in os.listdir(SCALERS_FOLDER):
                parts = scaler.split('_')
                num_exp = parts[1]
                num_data = parts[2].split('.')[0]
                id_exp_scaler = num_exp + '_' + num_data
                if id_exp == id_exp_scaler:
                    #print("Loading scaler: ", scaler)
                    scaler_conv_model = scaler
                    mfccs = fe.mfccs_scaled(samples, scaler_conv_model, id_exp)
                    pred = ep.make_predictions(conv_model, model_type, mfccs, prediction_scheme)
                    # add result to list and dictionary
                    predictions.append(pred)
                    model_predictions[id_exp] = pred
        for model in filenames:
            #print("Loading model: ", model)
            model_type = 'svm'
            parts = model.split('_')
            num_exp = parts[1]
            num_data = parts[2].split('.')[0]
            id_exp = num_exp + '_' + num_data
            model_path = os.path.join(dirpath, model)
            with open(model_path, 'rb') as file:
                svm_model = pickle.load(file)
            for scaler in os.listdir(SCALERS_FOLDER):
                parts = scaler.split('_')
                num_exp = parts[1]
                num_data = parts[2].split('.')[0]
                id_exp_scaler = num_exp + '_' + num_data
                if id_exp == id_exp_scaler:
                    #print("Loading scaler: ", scaler)
                    scaler_svm_model = scaler
                    mfccs = fe.mfccs_scaled(samples, scaler_svm_model, id_exp)
                    pred = ep.make_predictions(svm_model, model_type, mfccs, prediction_scheme)
                    # add result to list and dictionary
                    predictions.append(pred)
                    model_predictions[id_exp] = pred
        break
    #print(predictions)
    if prediction_scheme == 'majority':  
        final_prediction = ensemble_prediction_voting(predictions)
    elif prediction_scheme == 'avg_1':
        final_prediction = ensemble_prediction_avg_1(predictions)
    elif prediction_scheme == 'avg_2':
        final_prediction = ensemble_prediction_avg_2(predictions)


    if return_model_predictions == False:
        return final_prediction
    else:
        return final_prediction, model_predictions
