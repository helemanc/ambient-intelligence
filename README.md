# ambient-intelligence
Python application for Disruptive Situations Detection in public transports through Speech Emotion Recognition. 
![methodology](images/proposed_methodology.png)
## Usage
The application can be run from inside this folder through a UNIX terminal using the following command: `python3 main.py`

The arguments that need to be specified when running the script are:
- `-m`: to specify the method of execution. Possible options are:
  - `mic`: for local real-time execution. The audio is captured from the audio
interfaces of the laptop through the SpeechRecognition library for Python.
  - `file`: for offline execution. The audio is provided through a .wav audio file.
  - `real-mic`: for streaming execution in the real environment. The audio is provided by the microphone hosted on a CCTV camera. While the first two modes have been implemented, the last one has yet to be developed.
- `f`: to specify the audio file when `-m` file execution is enabled. After typing the `-f` option, the full path to an audio file should be provided.
- `-p`: to specify the aggregation strategy to be used when using the ensemble. Possible options are:
  - `voting`
  - `avg_1`
  - `avg_2`

Full command example: `python3 main.py -m file -f 'media/03-01-01-01-01-01-01_noise.wav'`