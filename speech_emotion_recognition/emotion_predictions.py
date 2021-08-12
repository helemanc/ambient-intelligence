import tensorflow as tf

def load_model(model_type):
    if model_type == 'bin':
        model = tf.keras.models.load_model("speech_emotion_recognition/models/binary_model")
    else:
        model = tf.keras.models.load_model("speech_emotion_recognition/models/multiclass_model")
    return model

def make_predictions(model, model_type, audio_samples):
    pred = model.predict(audio_samples)
    if model_type == 'bin':
        final_prediction = [1 * (x[0]>=0.52) for x in pred]
        if final_prediction == 1:
            return 1 # disruptive
        else:
            return 0 # non-disruptive
    else:
        disruptive_emotions = [0,1,4,6]
        final_prediction = pred.argmax()
        if final_prediction in disruptive_emotions:
            return 1
        else:
            return 0


