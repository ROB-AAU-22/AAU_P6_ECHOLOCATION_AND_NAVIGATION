import peakutils
import numpy as np
import soundfile as sf

def normalize_intensity(audio_data):
    """
    Normalizes intensity of audio data to a range of 0 to 1.
    This is done to find peaks more effectively.
    """
    
    audio_data = np.abs(audio_data)
    return (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))

def find_peaks(sound_file, threshold=0.05, min_distance=0.002, sample_rate=44100):
    """
    Finds peaks in the sound file using peakutils.
    threshold is the minimum height of the peak, min_distance is the minimum distance between peaks in seconds.
    sample_rate is the sample rate of the sound file.
    """
    left_channel = sound_file[:, 0]
    right_channel = sound_file[:, 1]
    
    distance_samples = int(min_distance * sample_rate) # convert to samples
    
    # find peaks in both channels
    left_peaks_indexes = peakutils.indexes(left_channel, thres=threshold, min_dist=distance_samples)
    right_peaks_indexes = peakutils.indexes(right_channel, thres=threshold, min_dist=distance_samples)
    
    return left_peaks_indexes, right_peaks_indexes
        
def cut_sound_file(sound_file, sr, before_threshold=0.005, after_threshold=0.3):
    """
    Cuts the sound file based on the first peak found in either channel.
    Cuts it by before_threshold seconds before the first peak and after_threshold seconds after the first peak.
    """
    normalized_sound_file = normalize_intensity(sound_file)
    
    left_peaks_indexes, right_peaks_indexes = find_peaks(normalized_sound_file, threshold=0.4, min_distance=0.002, sample_rate=sr)
    
    if len(left_peaks_indexes) > 0 and len(right_peaks_indexes) > 0:
        first_peak_index = min(left_peaks_indexes[0], right_peaks_indexes[0])
        start_index = max(0, first_peak_index - int(before_threshold * sr))
        end_index = min(len(sound_file), first_peak_index + int(after_threshold * sr))
        
        cut_sound_file = sound_file[start_index:end_index]
        
        return cut_sound_file
    else:
        print("No peaks found in the sound file. Returning original sound file.")
        return sound_file