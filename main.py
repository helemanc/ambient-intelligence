from inaSpeechSegmenter import Segmenter
from argparse import ArgumentParser
import speech_recognition as sr
import utils
import warnings
import datetime
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# usage disruptive: python3 main.py -m file -f 'media/03-01-05-02-02-02-03.wav'
# usage noise: python3 main.py -m file -f 'media/03-01-01-01-01-01-01_noise.wav'
parser = ArgumentParser()
parser.add_argument("-m", "--mode",
                    dest="mode",
                    required=True,
                    help="Methods of execution. Possible values: 'mic' for real time execution or 'file' for offline "
                         "execution or 'real_mic' for real time execution in real environment.",
                    metavar="XX"
                    )
parser.add_argument("-f", "--file_audio",
                    dest="file_audio",
                    required=False,
                    help="Audio file to analyze. The audio file must be provided with the complete path.",
                    metavar="XX")

parser.add_argument("-p", "--prediction_scheme",
                    dest="prediction_scheme",
                    required=False,
                    help="Could be 'voting', 'avg_1','avg_2",
                    metavar="XX")

args = vars(parser.parse_args())
if args['mode'] == 'file' and (args['file_audio']) is None:
    print("ERROR: To use the offline mode you need to specify a full-path audio file.")
    exit()

prediction_scheme = args['prediction_scheme']
# Part 1: Voice Activity Detection
# Instantiate Segmenter once
#start_time = datetime.datetime.now()
segmenter = Segmenter(vad_engine='smn', detect_gender=False)
#elapsed = datetime.datetime.now() - start_time
#print("Time elapsed for Segmenter instatiation", elapsed)

# Read audio from wav file
if args['mode'] == 'file':
    filepath = args['file_audio']
    utils.execute_vad_ser(segmenter, filepath, prediction_scheme)

# Read audio from local microphone
elif args['mode'] == 'mic':
    try:
        while True:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Speak:")  # REMOVE: this is used only for the purpose of demonstration
                audio = r.listen(source, phrase_time_limit=5) # the timout set here is equal to length chosen
                                                              # to cut/pad training
                filepath = "voice_activity_detection/tmp.wav"
                with open(filepath, "wb") as f:
                    f.write(audio.get_wav_data(convert_rate=16000))
                utils.execute_vad_ser(segmenter, filepath, prediction_scheme)
    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")

# Read audio from real microphone
elif args['mode'] == 'real_mic':
    """
    TO DO
    """



