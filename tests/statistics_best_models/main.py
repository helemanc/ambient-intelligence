from utilities import *
import os 
from audio_utilities import *
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#from tensorflow import ConfigProto
#from tensorflow import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
os.environ["CUDA_VISIBLE_DEVICES"]="1"


RAV_train, RAV_val, RAV_test = extract_ravdess(gender=False)
SAVEE_train, SAVEE_val, SAVEE_test = extract_savee() # test male 
TESS_train, TESS_test = extract_tess() # test female 
CREMA_train, CREMA_val, CREMA_test = extract_crema(gender=False)

# only best models 
RAV_model_list = ['3_2', '3_1', '4_1', '7_1', '7_2', '8_2']
TESS_model_list = ['3_3', '3_4', '4_3', '5_4', '8_4', '5_3']
SAVEE_model_list = ['1_6' , '3_5', '4_6', '8_5', '5_6', '6_5']
CREMA_model_list = ['2_8', '1_8', '3_7', '7_8', '8_8', '6_8']

best_models_list = RAV_model_list + SAVEE_model_list + TESS_model_list + CREMA_model_list
test_sets_list = [RAV_test, SAVEE_test, TESS_test, CREMA_test]
test_sets_name = ['RAVDESS', 'SAVEE', 'TESS', 'CREMA']


# make_test(RAV_test, RAV_model_list, 'ravdess')
# make_test(SAVEE_test, SAVEE_model_list, 'savee')
# make_test(TESS_test, TESS_model_list, 'tess')
make_test(CREMA_test, CREMA_model_list, 'crema')

