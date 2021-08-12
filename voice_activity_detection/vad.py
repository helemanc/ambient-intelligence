
def analyze_segmentation(seg_result):
    """
    Parameters
    ----------
    seg_result: the result of the segmentation is a list of tuples
                each tuple contains:
                 * label in 'speech', 'music', 'noEnergy'
                 * start time of the segment
                 * end time of the segment

    Returns
    -------
    count_speech: integer representing the number of segments
                  labeled as 'speech'
    """
    count_speech = 0
    for segment in seg_result:
        if segment[0] == 'speech':
            count_speech += 1
    return count_speech


def check_speech(segmenter, filepath):
    """
    Parameters
    ----------
    segmenter: an instance of inaSpeechSegmenter
    filepath: path of the audio recording to analyze

    Returns
    -------
    result: a string representing if the audio in input contains speech
           or something else (labeled as 'noise').
           If at least one segment of audio is recognized as speech, the
           audio contains voice
    """

    count_speech = analyze_segmentation(segmenter(filepath))
    if count_speech > 0:
        result = "speech"
    else:
        result = "noise"
    return result






