from voice_activity_detection import vad
from speech_emotion_recognition import emotion_predictions as ep, feature_extraction as fe


def execute_vad_ser(segmenter, filepath, model_type):
    seg = vad.check_speech(segmenter, filepath)
    if seg == 'speech':
        print("#########################################\n")
        print("The audio contains speech. \nStarting Speech Emotion Recognition process.\n")
        print("#########################################\n")

        # Part 1.2: Speech Emotion Recognition
        # Prepare data
        samples, sample_rate = fe.read_file(filepath)
        new_samples = fe.cut_pad(samples)
        mfccs = fe.mfccs_scaled(new_samples)

        # Make prediction
        model = ep.load_model(model_type)
        pred = ep.make_predictions(model, model_type, mfccs)

        if pred == 1:
            print("Speech contains disruptive emotion.")
        else:
            print("Speech does contain disruptive emotion.")


