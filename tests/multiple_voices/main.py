#from utilites import *
import os 


# RAV_train, RAV_val, RAV_test_m, RAV_test_f = extract_ravdess(gender=True)
# SAVEE_train, SAVEE_val, SAVEE_test = extract_savee() # test male 
# TESS_train, TESS_test = extract_tess() # test female 
# CREMA_train, CREMA_val, CREMA_test_f, CREMA_test_m = extract_crema(gender=True)

# RAV_model_list = ['3_2', '3_1', '4_1', '7_1', '7_2', '8_2']
# TESS_model_list = ['3_3', '3_4', '4_3', '5_4', '8_4', '5_3']
# SAVEE_model_list = ['1_6', '3_5', '4_6', '8_5', '5_6', '6_5']
# CREMA_model_list = ['2_8', '1_8', '3_7', '7_8', '8_8', '6_8']

# best_models_list = RAV_model_list + SAVEE_model_list + TESS_model_list + CREMA_model_list

# make_test(RAV_test_f, RAV_model_list, 'ravdess_f')
# make_test(RAV_test_m, RAV_model_list, 'ravdess_m')
# make_test(SAVEE_test, SAVEE_model_list, 'savee_m')
# make_test(TESS_test, TESS_model_list, 'tess_f')
# make_test(CREMA_test_f, CREMA_model_list, 'crema_f')
# make_test(CREMA_test_m, CREMA_model_list, 'crema_m')