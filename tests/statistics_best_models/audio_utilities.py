from pydub import AudioSegment
import librosa
import os 
from tqdm import tqdm
import pandas as pd
import numpy as npù

def create_overlapped_files(dataset_name, n, df_test): 
    project_dir = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
    emotion = df_test['emotion_label'][0]
    emotions = []
    paths = []
    # cut the len(df_test)%n samples to have a multiple of n
    if len(df_test)%n != 0:
        df_test = df_test.iloc[:-(len(df_test)%n)]
    
    file_index = 1 
    # create overlapped files
    for i in tqdm(range(0, len(df_test), n)):

        # overlap n files 
        for k in range(n):
            path = df_test.iloc[i+k]['path']
            if k == 0: 
                sound1 = AudioSegment.from_file(path)
                samples, sr = librosa.load(path, res_type='kaiser_fast', sr=16000)
                duration = librosa.get_duration(y = samples, sr = sr)
                sound2 = AudioSegment.from_file(df_test.iloc[i+k+1]['path'])
                combined = sound1.overlay(sound2)
            else:
                sound = AudioSegment.from_file(path)
                combined = combined.overlay(sound)
       
        n_overlapped_files = n

        # create a folder named "dataset_name" and a subfolder named "overlapped_" + n_overlapped_files if not exist
        path_overlapped = os.path.join(project_dir, "overlapped_audio_files", dataset_name, "overlapped_" + str(n_overlapped_files))
        if not os.path.exists(path_overlapped):
            os.makedirs(path_overlapped)
    
        # save audio file at new_path 
        new_path = os.path.join(path_overlapped, "overlapped_audio_" + str(file_index) + '_' + str(emotion) + ".wav")
        combined.export(new_path, format= 'wav')
        emotions.append(emotion)
        paths.append(new_path)
        file_index += 1

    df_test = pd.DataFrame([emotions, paths]).T
    df_test.columns = ['emotion_label', 'path']

    
    return df_test