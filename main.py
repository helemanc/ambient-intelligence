# Load the API for VAD
from inaSpeechSegmenter import Segmenter
from argparse import ArgumentParser
import speech_recognition as sr
import utils
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# usage:  python3 main.py -m file -f 'media/03-01-01-01-01-01-01.wav'
# usage disruptive: python3 main.py -m file -f 'media/03-01-05-02-02-02-03.wav'
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
'''
parser.add_argument("-t", "--type",
                    dest="model_type",
                    required=True,
                    help="Type of emotion classification. Possible values: 'bin' for binary classification, 'mul' for multi-class classification ",
                    metavar="XX")
'''


args = vars(parser.parse_args())
if args['mode'] == 'file' and (args['file_audio']) is None:
    print("ERROR: To use the offline mode you need to specify a full-path audio file.")
    exit()

# Part 1: Voice Activity Detection
# Instantiate Segmenter once
segmenter = Segmenter(vad_engine='smn', detect_gender=False)

# Read audio from wav file
if args['mode'] == 'file':
    filepath = args['file_audio']
    utils.execute_vad_ser(segmenter, filepath)

# Read audio from local microphone
elif args['mode'] == 'mic':
    try:
        while True:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Speak:")  # REMOVE: this is used only for the purpose of demonstration
                audio = r.listen(source, phrase_time_limit=5) # the timout set here is equal to length chosen to cut/pad training
                filepath = "voice_activity_detection/tmp.wav"
                with open(filepath, "wb") as f:
                    f.write(audio.get_wav_data(convert_rate=16000))
                utils.execute_vad_ser(segmenter, filepath, args['model_type'])
    except KeyboardInterrupt:
        print("Press Ctrl-C to terminate while statement")

# Read audio from local microphone
elif args['mode'] == 'real_mic':
    '''
    TO DO: IP protocol to receive audio from real microphone in a continuous flow
    - take the audio from buffer every 5 seconds with overlapping (idea used in Simon)
    - write bytes into temporal wav (there is no latency problem since the write operation 
        of a very small file is very small)
    - execute_vad_ser
    
    '''



