from utilities import *
import os 
from audio_utilities import *
from sklearn.utils import shuffle

N = [2, 5, 10]
RAV_train, RAV_val, RAV_test = extract_ravdess(gender=False)
SAVEE_train, SAVEE_val, SAVEE_test = extract_savee() # test male 
TESS_train, TESS_test = extract_tess() # test female 
CREMA_train, CREMA_val, CREMA_test = extract_crema(gender=False)

# only best models 
RAV_model_list = ['3_2'] #, '3_1', '4_1', '7_1', '7_2', '8_2']
TESS_model_list = ['3_3'] #, '3_4', '4_3', '5_4', '8_4', '5_3']
SAVEE_model_list = ['1_6']# , '3_5', '4_6', '8_5', '5_6', '6_5']
CREMA_model_list = ['2_8'] #, '1_8', '3_7', '7_8', '8_8', '6_8']

best_models_list = RAV_model_list + SAVEE_model_list + TESS_model_list + CREMA_model_list
test_sets_list = [RAV_test, SAVEE_test, TESS_test, CREMA_test]
test_sets_name = ['RAVDESS', 'SAVEE', 'TESS', 'CREMA']
for i, df_test in enumerate(test_sets_list): 
    for n in N: 
        emotion_enc = {'fear':1, 'disgust':1, 'neutral':0, 'calm':0,  'happy':0, 'sadness':1, 'surprise':0, 'angry':1}
        labels= pd.Series(list(df_test.emotion_label)).replace(emotion_enc)
        # substitute labels of df_test with labels
        df_test['emotion_label'] = labels

        # random pick 10 samples marked as 1 and 10 samples marked as 0
        # extract list of samples marked as 1 and 0
        df_test_neg = df_test[df_test['emotion_label'] == 1]
        df_test_neg = df_test_neg.reset_index(drop=True)
        df_test_pos = df_test[df_test['emotion_label'] == 0]
        df_test_pos = df_test_pos.reset_index(drop=True)

        # random pick 10 samples marked as 1 and 10 samples marked as 0
        # shuffle df_test_neg and df_test_pos
        df_test_neg = shuffle(df_test_neg, random_state=42)
        df_test_pos = shuffle(df_test_pos, random_state=42)

        #print(test_sets_name[i], n, len(df_test_neg), len(df_test_neg)%n)
        #print(test_sets_name[i], n, len(df_test_pos), len(df_test_pos)%n)

        #df_test_neg = df_test_neg.sample(n=n, random_state=42).reset_index(drop=True)
        #df_test_pos = df_test_pos.sample(n=n, random_state=42).reset_index(drop=True)



        df_test_neg = create_overlapped_files(test_sets_name[i], n,  df_test_neg)
        df_test_pos = create_overlapped_files(test_sets_name[i], n,  df_test_pos)
        # overlap audio file of df_test_neg and df_test_pos located at path feature
        if test_sets_name[i] == 'RAVDESS':
            model_list = RAV_model_list
        elif test_sets_name[i] == 'SAVEE':
            model_list = SAVEE_model_list
        elif test_sets_name[i] == 'TESS':
            model_list = TESS_model_list
        elif test_sets_name[i] == 'CREMA':
            model_list = CREMA_model_list

        df = pd.concat([df_test_neg, df_test_pos]).reset_index(drop=True)
        make_test(df, model_list, test_sets_name[i] + '_GENERAL_' + str(n))
        #make_test(df_test_neg, model_list, test_sets_name[i] + '_NEG_' + str(n))
        #make_test(df_test_pos, model_list, test_sets_name[i] + '_POS_' + str(n))

# make_test(RAV_test_m, RAV_model_list, 'ravdess_m')
# make_test(SAVEE_test, SAVEE_model_list, 'savee_m')
# make_test(TESS_test, TESS_model_list, 'tess_f')
# make_test(CREMA_test_f, CREMA_model_list, 'crema_f')
# make_test(CREMA_test_m, CREMA_model_list, 'crema_m')