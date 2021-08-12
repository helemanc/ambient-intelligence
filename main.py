# Load the API for VAD
from inaSpeechSegmenter import Segmenter
from voice_activity_detection import vad
from argparse import ArgumentParser
from speech_emotion_recognition import emotion_predictions as ep, feature_extraction as fe

parser = ArgumentParser()
parser.add_argument("-m", "--mode",
                    dest="mode",
                    required=True,
                    help="Methods of execution. Possible values: 'mic' for real time execution or 'file' for offline execution.",
                    metavar="XX"
                    )
parser.add_argument("-f", "--file_audio",
                    dest="file_audio",
                    required=False,
                    help="Audio file to analyze. The audio file must be provided with the complete path.",
                    metavar="XX")

parser.add_argument("-t", "--type",
                    dest="model_type",
                    required=True,
                    help="Type of emotion classification. Possible values: 'bin' for binary classification, 'mul' for multi-class classification ",
                    metavar="XX")


args = vars(parser.parse_args())
if args['mode'] == 'file' and (args['file_audio']) == None:
    print("ERROR: To use the offline mode you need to specify a full-path audio file.")
    exit()




# Part 1: Voice Activity Detection
if args['mode']=='file':
    filepath = args['file_audio']
    seg = vad.check_speech(Segmenter(vad_engine='smn', detect_gender = False), filepath)
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
        model = ep.load_model(args['model_type'])
        pred = ep.make_predictions(model,args['model_type'],mfccs)

        if pred == 1:
            print("Speech contains disruptive emotion.")
        else:
            print("Speech does contain disruptive emotion.")


    else:
        print("#########################################\n")
        print("The audio does not contain speech.\n")
        print("#########################################\n")
