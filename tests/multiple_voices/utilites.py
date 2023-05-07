import os
from tqdm.notebook import tqdm
from tqdm import tqdm
import pandas as pd 
from sklearn.metrics import classification_report
import feature_extraction as fe
import emotion_predictions as ep
import scipy
import numpy as np
from scipy import signal
from scipy.io.wavfile import write
import sklearn
import random
import tensorflow as tf
import pickle
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# print('The scikit-learn version is {}.'.format(sklearn.__version__))

MODELS_FOLDER = "models"
SCALERS_FOLDER = "scalers"
project_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
main_path = os.path.join(project_dir, "speech_emotion_recognition", "datasets")
TESS = os.path.join(main_path, "tess/TESS Toronto emotional speech set data/") 
RAV = os.path.join(main_path, "ravdess-emotional-speech-audio/audio_speech_actors_01-24")
SAVEE = os.path.join(main_path, "savee/ALL/")
CREMA = os.path.join(main_path, "creamd/AudioWAV/")


def extract_ravdess(gender=False):

    # READ RAVDESS 
    lst = []
    emotion = []
    voc_channel = []
    full_path = []
    modality = []
    intensity = []
    actors = []
    phrase =[]
    for root, dirs, files in tqdm(os.walk(RAV)):
        for file in files:
            
            try:
                #Load librosa array, obtain mfcss, store the file and the mfcss information in a new array
                # X, sample_rate = librosa.load(os.path.join(root,file), res_type='kaiser_fast')
                # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
                # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
                # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
            
                modal = int(file[1:2])
                vchan = int(file[4:5])
                lab = int(file[7:8])
                ints = int(file[10:11])
                phr = int(file[13:14])
                act = int(file[18:20])
                # arr = mfccs, lab
                # lst.append(arr)
                
                modality.append(modal)
                voc_channel.append(vchan)
                emotion.append(lab) #only labels
                intensity.append(ints)
                phrase.append(phr)
                actors.append(act)
                
                full_path.append((root, file)) # only files
            # If the file is not valid, skip it
            except ValueError:
                continue

    # EMOTIONS MAPPING 
    # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    # merge neutral and calm
    emotions_list = ['neutral', 'neutral', 'happy', 'sadness', 'angry', 'fear', 'disgust', 'surprise']
    emotion_dict = {em[0]+1:em[1] for em in enumerate(emotions_list)}

    df = pd.DataFrame([emotion, voc_channel, modality, intensity, actors, actors,phrase, full_path]).T
    df.columns = ['emotion', 'voc_channel', 'modality', 'intensity', 'actors', 'gender', 'phrase', 'path']
    df['emotion'] = df['emotion'].map(emotion_dict)
    df['voc_channel'] = df['voc_channel'].map({1: 'speech', 2:'song'})
    df['modality'] = df['modality'].map({1: 'full AV', 2:'video only', 3:'audio only'})
    df['intensity'] = df['intensity'].map({1: 'normal', 2:'strong'})
    df['actors'] = df['actors']
    df['gender'] = df['actors'].apply(lambda x: 'female' if x%2 == 0 else 'male')
    df['phrase'] = df['phrase'].map({1: 'Kids are talking by the door', 2:'Dogs are sitting by the door'})
    df['path'] = df['path'].apply(lambda x: x[0] + '/' + x[1])

    # remove files with noise to apply the same noise to all files for data augmentation 
    df = df[~df.path.str.contains('noise')]

    # only speech
    RAV_df = df
    RAV_df = RAV_df.loc[RAV_df.voc_channel == 'speech']
    RAV_df.insert(0, "emotion_label", RAV_df.emotion, True)
    RAV_df = RAV_df.drop(['emotion', 'voc_channel', 'modality', 'intensity', 'phrase'], 1)
    
    RAV_train = []
    RAV_val = []
    RAV_test = []
    for index, row in RAV_df.iterrows():
        if row['actors'] in range(1,21): 
            RAV_train.append(row) 
        elif row['actors'] in range(21,23): 
            RAV_val.append(row)
        elif row['actors'] in range(23,25): 
            RAV_test.append(row)
    #len(RAV_train), len(RAV_val), len(RAV_test)

    RAV_train = pd.DataFrame(RAV_train)
    RAV_val = pd.DataFrame(RAV_val)
    RAV_test = pd.DataFrame(RAV_test)

    # define gender-based test sets
    RAV_test_f = []
    RAV_test_m = []
    if gender: 
        for index, row in RAV_test.iterrows():
            if row['actors']%2 == 0: # even = females
                RAV_test_f.append(row)
            else: # odd = males 
                RAV_test_m.append(row)
        RAV_test_f = pd.DataFrame(RAV_test_f)
        RAV_test_m = pd.DataFrame(RAV_test_m)
        RAV_test_f.drop(['actors'], 1)
        RAV_test_f.reset_index(drop=True, inplace = True)
        RAV_test_m.drop(['actors'], 1)
        RAV_test_m.reset_index(drop=True, inplace = True)

        

    RAV_train = RAV_train.drop(['actors'], 1)
    RAV_val = RAV_val.drop(['actors'], 1)
    RAV_test = RAV_test.drop(['actors'], 1)

    RAV_train.reset_index(drop=True, inplace = True) 
    RAV_val.reset_index(drop=True, inplace = True) 
    RAV_test.reset_index(drop=True, inplace = True)

    if gender:
        return RAV_train, RAV_val, RAV_test_m, RAV_test_f
    else:
        return RAV_train, RAV_val, RAV_test

def extract_savee():
    # Get the data location for SAVEE
    dir_list = os.listdir(SAVEE)

    # parse the filename to get the emotions
    emotion=[]
    path = []
    actors = []
    gender = []
    for i in dir_list:
        actors.append(i[:2])
        if i[-8:-6]=='_a':
            emotion.append('angry')
            gender.append('male')
        elif i[-8:-6]=='_d':
            emotion.append('disgust')
            gender.append('male')
        elif i[-8:-6]=='_f':
            emotion.append('fear')
            gender.append('male')
        elif i[-8:-6]=='_h':
            emotion.append('happy')
            gender.append('male')
        elif i[-8:-6]=='_n':
            emotion.append('neutral')
            gender.append('male')
        elif i[-8:-6]=='sa':
            emotion.append('sadness')
            gender.append('male')
        elif i[-8:-6]=='su':
            emotion.append('surprise')
            gender.append('male') 
        else:
            emotion.append('Unknown') 
        path.append(SAVEE + i)
        
    # Now check out the label count distribution 
    SAVEE_df = pd.DataFrame(emotion, columns = ['emotion_label'])
                        
    SAVEE_df = pd.concat([SAVEE_df,
                        pd.DataFrame(actors, columns = ['actors']),
                        pd.DataFrame(gender, columns = ['gender']), 
                        pd.DataFrame(path, columns = ['path'])], axis = 1)
    SAVEE_train = []
    SAVEE_val = []
    SAVEE_test = []

    for index, row in SAVEE_df.iterrows(): 
        if row['actors'] == 'DC' or row ['actors'] == 'JE':
            SAVEE_train.append(row)
        elif row['actors'] == 'JK': 
            SAVEE_val.append(row)
        else: 
            SAVEE_test.append(row)

    SAVEE_train = pd.DataFrame(SAVEE_train)
    SAVEE_val = pd.DataFrame(SAVEE_val)
    SAVEE_test = pd.DataFrame(SAVEE_test)

    SAVEE_train = SAVEE_train.drop(['actors'], 1)
    SAVEE_val = SAVEE_val.drop(['actors'], 1)
    SAVEE_test = SAVEE_test.drop(['actors'], 1)

    SAVEE_train = SAVEE_train.reset_index(drop=True) 
    SAVEE_val = SAVEE_val.reset_index(drop=True) 
    SAVEE_test = SAVEE_test.reset_index(drop=True) 

    return SAVEE_train, SAVEE_val, SAVEE_test

def extract_tess():
    dir_list = os.listdir(TESS)
    dir_list.sort()
    dir_list

    path = []
    emotion = []
    gender = []
    actors = []

    for i in dir_list:
        fname = os.listdir(TESS + i)
        for f in fname:
            if i == 'OAF_angry':
                emotion.append('angry')
                gender.append('female')
                actors.append('OAF')
            elif i == 'YAF_angry': 
                emotion.append('angry')
                gender.append('female')
                actors.append('YAF')
                
                
            elif i == 'OAF_disgust' :
                emotion.append('disgust')
                gender.append('female')
                actors.append('OAF')
            elif i == 'YAF_disgust': 
                emotion.append('disgust')
                gender.append('female')
                actors.append('YAF')
                
                
            elif i == 'OAF_Fear':
                emotion.append('fear')
                gender.append('female')
                actors.append('OAF')
            elif i == 'YAF_fear': 
                emotion.append('fear')
                gender.append('female')
                actors.append('YAF') 
                
                
            elif i == 'OAF_happy' :
                emotion.append('happy')
                gender.append('female')
                actors.append('OAF')
            elif i == 'YAF_happy': 
                emotion.append('angry')
                gender.append('female')
                actors.append('YAF')            
                
            elif i == 'OAF_neutral':
                emotion.append('neutral')
                gender.append('female')
                actors.append('OAF')   
            elif i == 'YAF_neutral': 
                emotion.append('neutral')
                gender.append('female')
                actors.append('YAF')      
                
                    
            elif i == 'OAF_Pleasant_surprise':
                emotion.append('surprise')
                gender.append('female')
                actors.append('OAF')
            
            elif i == 'YAF_pleasant_surprised': 
                emotion.append('surprise')
                gender.append('female')
                actors.append('YAF')            
                
            elif i == 'OAF_Sad':
                emotion.append('sadness')
                gender.append('female')
                actors.append('OAF')
            elif i == 'YAF_sad': 
                emotion.append('sadness')
                gender.append('female')
                actors.append('YAF')            
            else:
                emotion.append('Unknown')
            path.append(TESS + i + "/" + f)

    TESS_df = pd.DataFrame(emotion, columns = ['emotion_label'])
    TESS_df = pd.concat([TESS_df, pd.DataFrame(gender, columns = ['gender']), 
                        pd.DataFrame(actors, columns= ['actors']),
                        pd.DataFrame(path, columns = ['path'])],axis=1)
    TESS_df= TESS_df[~TESS_df.path.str.contains('noise')]
    TESS_train = []
    TESS_test = []

    for index, row in TESS_df.iterrows(): 
        if row['actors'] == 'YAF': 
            TESS_train.append(row)
        else: 
            TESS_test.append(row)

    TESS_train = pd.DataFrame(TESS_train)
    TESS_test = pd.DataFrame(TESS_test)
    TESS_train = TESS_train.drop(['actors'], 1)
    TESS_test = TESS_test.drop(['actors'], 1)
    TESS_train = TESS_train.reset_index(drop=True) 
    TESS_test  = TESS_test.reset_index(drop=True) 

    return TESS_train, TESS_test

def extract_crema(gender=False):
    males = [1,
            5,
            11,
            14,
            15,
            16,
            17,
            19,
            22,
            23,
            26,
            27,
            31,
            32,
            33,
            34,
            35,
            36,
            38,
            39,
            41,
            42,
            44,
            45,
            48,
            50,
            51,
            57,
            59, 
            62, 
            64,
            65, 
            66,
            67,
            68,
            69,
            70,
            71,
            77, 
            80, 
            81, 
            83, 
            85, 
            86, 
            87,
            88, 
            90] 
    females = [ 2,
                3,
                4,
                6,
                7,
                8,
                9,
                10,
                12,
                13,
                18,
                20,
                21,
                24,
                25,
                28,
                29,
                30,
                37,
                40,
                43,
                46,
                47,
                49,
                52,
                53,
                54,
                55,
                56, 
                58, 
                60,
                61,
                63,
                72, 
                73, 
                74, 
                75, 
                76, 
                78, 
                79, 
                82, 
                84, 
                89, 
                91]
    crema_directory_list = os.listdir(CREMA)

    file_emotion = []
    file_path = []
    actors = []
    gender = []




    for file in crema_directory_list:

        # storing file emotions
        part=file.split('_')
        
        # use only high intensity files
        if "HI" in part[3] :
            actor = part[0][2:]
            actors.append(actor)
            if int(actor) in males:
                gender.append('male')
            else: 
                gender.append('female')
        
            # storing file paths
            file_path.append(CREMA + file)
            if part[2] == 'SAD':
                file_emotion.append('sadness')
            elif part[2] == 'ANG':
                file_emotion.append('angry')
            elif part[2] == 'DIS':
                file_emotion.append('disgust')
            elif part[2] == 'FEA':
                file_emotion.append('fear')
            elif part[2] == 'HAP':
                file_emotion.append('happy')
            elif part[2] == 'NEU':
                file_emotion.append('neutral')
            else:
                file_emotion.append('Unknown')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['emotion_label'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['path'])
    actors_df = pd.DataFrame(actors, columns=['actors'])
    gender_df = pd.DataFrame(gender, columns=['gender'])                      
    Crema_df = pd.concat([emotion_df, actors_df, gender_df, path_df], axis=1)

    actor_files = {}

    for index, row in Crema_df.iterrows():
        actor = row['actors']
        if actor not in actor_files.keys(): 
            actor_files[actor] = 1
        else: 
            actor_files[actor]+=1
    
    males_to_remove = ['17', '80', '88']
    new_df = []
    for index, row in Crema_df.iterrows(): 
        if row['actors'] not in males_to_remove: 
            new_df.append(row)
    CREMA_df = pd.DataFrame(new_df)

    new_df = []
    for index, row in Crema_df.iterrows(): 
        if row['actors'] not in males_to_remove: 
            new_df.append(row)
    count_males = 0 
    count_females = 0 
    male_list = []
    female_list = []
    for index, row in CREMA_df.iterrows(): 
        gender = row['gender']
        actor = row['actors']
        if gender == 'male':
            count_males +=1
            if actor not in male_list: 
                male_list.append(actor)
        else: 
            count_females +=1
            if actor not in female_list: 
                female_list.append(actor)
    CREMA_train = []
    CREMA_val = []
    CREMA_test = []
    CREMA_test_f = []
    CREMA_test_m = [] 

    females_train = random.sample(female_list, 32)
    males_train = random.sample(male_list, 32)

    # remove the elements assigned to train 
    for element in females_train:
        if element in female_list:
            female_list.remove(element)
            
    for element in males_train:
        if element in male_list:
            male_list.remove(element)

            
    females_val = random.sample(female_list, 6) 
    males_val = random.sample(male_list, 6) 

    # remove the elements assigned to val
    for element in females_val:
        if element in female_list:
            female_list.remove(element)
            
    for element in males_val:
        if element in male_list:
            male_list.remove(element)
            
    females_test = random.sample(female_list, 6) 
    males_test = random.sample(male_list, 6)  

    train = females_train + males_train 
    val = females_val + males_val 
    test = females_test + males_test   

    for index, row in CREMA_df.iterrows(): 
        gender = row['gender']
        actor = row['actors']

        if gender: 
            if actor in train: 
                CREMA_train.append(row)
            elif actor in val: 
                CREMA_val.append(row)
            elif actor in females_test:
                CREMA_test_f.append(row)
            elif actor in males_test:
                CREMA_test_m.append(row)
        else:
            if actor in train: 
                CREMA_train.append(row)
            elif actor in val: 
                CREMA_val.append(row)
            else:
                CREMA_test.append(row)

    CREMA_train = pd.DataFrame(CREMA_train) 
    CREMA_val = pd.DataFrame(CREMA_val) 
    if gender: 
        CREMA_test_f = pd.DataFrame(CREMA_test_f)
        CREMA_test_m = pd.DataFrame(CREMA_test_m)
    else: 
        CREMA_test = pd.DataFrame(CREMA_test)

    CREMA_train = CREMA_train.drop(['actors'], 1)
    CREMA_val = CREMA_val.drop(['actors'], 1)
    if gender: 
        CREMA_test_f = CREMA_test_f.drop(['actors'], 1)
        CREMA_test_m = CREMA_test_m.drop(['actors'], 1)
    else:
        CREMA_test = CREMA_test.drop(['actors'], 1)
    
    CREMA_train = CREMA_train.reset_index(drop=True) 
    CREMA_val = CREMA_val.reset_index(drop = True) 
    if gender: 
        CREMA_test_f = CREMA_test_f.reset_index(drop = True)
        CREMA_test_m = CREMA_test_m.reset_index(drop = True)
    else: 
        CREMA_test = CREMA_test.reset_index(drop = True)

    if gender: 
        return CREMA_train, CREMA_val, CREMA_test_f, CREMA_test_m
    else:
        return CREMA_train, CREMA_val, CREMA_test

def denoise(samples):
    """
    :param samples: an array representing the sampled audio file
    :type samples: float
    :return: an array representing the clean audio file
    :rtype: float
    """
    samples_wiener = scipy.signal.wiener(samples)
    return samples_wiener

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
    :param amplify: Whether to amplify audio volume, defaults to False
    :type amplify: bool, optional
    :return: Resampled audio and new sample rate
    :rtype: numpy.ndarray, int
    """
    if sample_rate < required_sample_rate:
        resampling_factor = int(round(required_sample_rate/sample_rate, 0))
        new_rate = sample_rate * resampling_factor
        samples = len(input_data) * resampling_factor
        resampled = signal.resample(input_data, samples)
    elif sample_rate > required_sample_rate:
        resampling_factor = int(round(sample_rate/required_sample_rate, 0))
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

def make_predictions(dataset, labels, model_name): 
    predictions = []
    model_predictions_list = []
    counter = 0
    for filepath in tqdm(dataset['path']):
        samples, sample_rate = fe.read_file(filepath)
        samples, sample_rate = resample(samples, sample_rate)
        new_samples = fe.cut_pad(samples)
        #new_filepath = "tmp.wav"
        for dirpath, dirnames, filenames in os.walk(MODELS_FOLDER):
            # iterate over the list of dirnames to load the convmodel
            # iterate over the list of filenames to get the svm model
            for model in dirnames:
                model_type = 'conv'
                parts = model.split('_')
                num_exp = parts[1]
                num_data = parts[2].split('.')[0]
                id_exp = num_exp + '_' + num_data
                if id_exp == model_name: 
                    model_path = os.path.join(dirpath, model)
                    conv_model = tf.keras.models.load_model(model_path)
                    for scaler in os.listdir(SCALERS_FOLDER):
                        parts = scaler.split('_')
                        num_exp = parts[1]
                        num_data = parts[2].split('.')[0]
                        id_exp_scaler = num_exp + '_' + num_data
                        if id_exp == id_exp_scaler:
                            # print("Loading scaler: ", scaler)
                            scaler_conv_model = scaler
                            mfccs = fe.mfccs_scaled(new_samples, scaler_conv_model, id_exp)
                            pred = ep.make_predictions(conv_model, model_type, mfccs, 'majority')
                            # add result to list and dictionary
                            predictions.append(pred)
                            #model_predictions[id_exp] = pred
            for model in filenames:
                # print("Loading model: ", model)
                model_type = 'svm'
                parts = model.split('_')
                num_exp = parts[1]
                num_data = parts[2].split('.')[0]
                id_exp = num_exp + '_' + num_data
                model_path = os.path.join(dirpath, model)
                
                if id_exp == model_name: 
                    with open(model_path, 'rb') as file:
                        svm_model = pickle.load(file)
                    for scaler in os.listdir(SCALERS_FOLDER):
                        
                        parts = scaler.split('_')
                        num_exp = parts[1]
                        num_data = parts[2].split('.')[0]
                    

                        id_exp_scaler = num_exp + '_' + num_data
                        print(id_exp, id_exp_scaler)
                        if id_exp == id_exp_scaler:
                            # print("Loading scaler: ", scaler)
                            scaler_svm_model = scaler
                            mfccs = fe.mfccs_scaled(samples, scaler_svm_model, id_exp)
                            pred = ep.make_predictions(svm_model, model_type, mfccs, 'majority')
                            predictions.append(pred)
                            #model_predictions[id_exp] = pred
            break
        #final_prediction, model_predictions = ensemble.ensemble(new_samples, prediction_scheme, return_model_predictions = True)

        #predictions.append(final_prediction)
        #model_predictions_list.append(model_predictions) 
        print(counter)
        print("True label", labels[counter], "Predicted label", predictions[counter])
        counter+=1
    return predictions

def create_dataframe_predictions(prediction_list):
    df_predictions = pd.DataFrame(prediction_list)
    return df_predictions

def create_dataframe_res(labels, df_predictions, dataset): 
    df_res = pd.concat([labels, 
                    df_predictions, 
                    dataset.path], axis = 1, ignore_index=True, sort=False)
    new_header = []
    new_header.append('true_label')
    new_header.append('pred_label')
    new_header = new_header 
    new_header.append('path')
    df_res.columns = new_header
    return df_res

def make_test(df_test, model_list, type): 
    for model in model_list: 
        emotion_enc = {'fear':1, 'disgust':1, 'neutral':0, 'calm':0,  'happy':0, 'sadness':1, 'surprise':0, 'angry':1}
        labels= pd.Series(list(df_test.emotion_label)).replace(emotion_enc)
        predictions = make_predictions(df_test, labels, model)
        df_predictions = create_dataframe_predictions(predictions)
        df_res = create_dataframe_res(labels, df_predictions, df_test)
        print("RESSS", df_res)
        print("###### ", model, " ######")
        report = classification_report(df_res.true_label, df_res.pred_label, output_dict=True)
        df = pd.DataFrame(report).transpose()
        pathname = 'results/' + type + '/' + model + '.csv'
        df.to_csv(pathname)

            






