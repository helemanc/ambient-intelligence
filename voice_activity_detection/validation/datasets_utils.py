import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# TO DO: fare esclusione .DS_Store da tutte le funzione di extract e risalvare il csv
# ELIMINARE _noise files

def extract_ravdess(dataset_path=None):
    if dataset_path is None:
        RAV = "datasets/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
    else:
        RAV = dataset_path

    dir_list = os.listdir(RAV)
    dir_list.sort()

    emotion = []
    gender = []
    path = []
    for i in dir_list:
        print(i)
        if not i.startswith('.'): #exclude .DS_Store files
            fname = os.listdir(RAV + i)

            for f in fname:
                if not f.startswith('.'): #exclude .DS_Store files
                    part = f.split('.')[0].split('-')
                    emotion.append(int(part[2]))
                    temp = int(part[6])
                    if temp % 2 == 0:
                        temp = "female"
                    else:
                        temp = "male"
                    gender.append(temp)
                    path.append(RAV + i + '/' + f)

    RAV_df = pd.DataFrame(emotion)
    RAV_df = RAV_df.replace(
        {1: 'speech', 2: 'speech', 3: 'speech', 4: 'speech', 5: 'speech', 6: 'speech', 7: 'speech', 8: 'speech'})

    RAV_df = pd.concat([pd.DataFrame(gender), RAV_df], axis=1)
    RAV_df.columns = ['gender', 'emotion']
    RAV_df['labels'] = RAV_df.emotion
    RAV_df['source'] = 'RAVDESS'
    RAV_df = pd.concat([RAV_df, pd.DataFrame(path, columns=['path'])], axis=1)
    RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
    RAV_df.labels.value_counts()

    return RAV_df

def extract_tess(dataset_path=None):
    if dataset_path is None:
        TESS = "datasets/tess/TESS Toronto emotional speech set data/"
    else:
        TESS = dataset_path


    tess_directory_list = os.listdir(TESS)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(TESS + dir)
        for file in directories:
            file_emotion.append('speech')
            print(TESS + dir + '/' + file)
            file_path.append(TESS + dir + '/' + file)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['labels'])
    emotion_df['source'] = 'TESS'
    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['path'])
    TESS_df = pd.concat([emotion_df, path_df], axis=1)
    TESS_df.head()

    TESS_df['labels'].unique()

    return TESS_df


def extract_emodb(dataset_path=None):
    if dataset_path is None:
        EMODB = "datasets/emo-db/wav/"
    else:
        EMODB = dataset_path

    #emodb_directory_list = os.listdir(EMODB)
    emotion = []
    path = []

    for root, dirs, files in os.walk(EMODB):
        for name in files:
            print("NAME ", name)
            if name[0:2] in '0310111215':  # MALE
                if name[5] == 'W':  # Ärger (Wut) -> Angry
                    emotion.append('speech')
                elif name[5] == 'L':  # Langeweile -> Boredom
                    emotion.append('speech')
                elif name[5] == 'E':  # Ekel -> Disgusted
                    emotion.append('speech')
                elif name[5] == 'A':  # Angst -> Angry
                    emotion.append('speech')
                elif name[5] == 'F':  # Freude -> Happiness
                    emotion.append('speech')
                elif name[5] == 'T':  # Trauer -> Sadness
                    emotion.append('speech')
                else:
                    emotion.append('unknown')
            else:
                if name[5] == 'W':  # Ärger (Wut) -> Angry
                    emotion.append('speech')
                elif name[5] == 'L':  # Langeweile -> Boredom
                    emotion.append('speech')
                elif name[5] == 'E':  # Ekel -> Disgusted
                    emotion.append('speech')
                elif name[5] == 'A':  # Angst -> Angry
                    emotion.append('speech')
                elif name[5] == 'F':  # Freude -> Happiness
                    emotion.append('speech')
                elif name[5] == 'T':  # Trauer -> Sadness
                    emotion.append('speech')
                else:
                    emotion.append('unknown')

            path.append(os.path.join(EMODB, name))

    emodb_df = pd.DataFrame(emotion, columns=['labels'])
    emodb_df['source'] = 'EMODB'
    emodb_df = pd.concat([emodb_df, pd.DataFrame(path, columns=['path'])], axis=1)

    return emodb_df

def extract_extra_sounds_speech():
    SPEECH_PATH = "datasets/recorded_sounds/speech/"
    labels = []
    path = []

    for root, dirs, files in os.walk(SPEECH_PATH):
        for name in files:
            labels.append('speech')
            path.append(os.path.join(SPEECH_PATH, name))

    speech_df = pd.DataFrame(labels, columns=['labels'])
    speech_df['source'] = 'recorded_speech'
    speech_df = pd.concat([speech_df, pd.DataFrame(path, columns=['path'])], axis=1)
    return speech_df

def extract_extra_sounds_noise():
    NOISE_PATH = "datasets/recorded_sounds/other/"
    labels = []
    path = []

    for root, dirs, files in os.walk(NOISE_PATH):
      for name in files:
          labels.append('noise')
          path.append(os.path.join(NOISE_PATH, name))

    noise_df = pd.DataFrame(labels, columns=['labels'])
    noise_df['source'] = 'recorded_speech'
    noise_df = pd.concat([noise_df, pd.DataFrame(path, columns=['path'])], axis=1)
    return noise_df

def datasets_union():
    rav_df = extract_ravdess()
    tess_df = extract_tess()
    emo_df = extract_emodb()
    speech_df = extract_extra_sounds_speech()
    noise_df = extract_extra_sounds_noise()
    df = pd.concat([rav_df, tess_df, emo_df, speech_df, noise_df], axis=0)
    #df.to_csv("datasets/validation_datasets.csv", index=False)
    return df

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def add_random_noise(df=None):
    '''
    Write copies of the audio files by adding random noise
    :return:
    '''
    if df is None:
        df = datasets_union()
    else:
        df = pd.read_csv('datasets/validation_datasets.csv')

    counter = 0
    rows_list = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        filepath = row['path']
        label = row['labels']
        source = row['source']
        print(filepath)
        if label == 'speech':
            provv_filename = filepath.replace(".wav", "")
            new_file_name = provv_filename + "_noise.wav"
            data, sample_rate = librosa.load(filepath)
            # noise injection
            x = noise(data)
            sf.write(new_file_name, x, sample_rate)
            dict_row = {"labels": label, "source": source, "path": filepath}
            rows_list.append(dict_row)
            counter += 1
    print('Done!')
"""
def add_ambient_noise():
    '''
    Write copies of the audio files by adding train and crowd noise
    :return:
    '''
"""

add_random_noise()
