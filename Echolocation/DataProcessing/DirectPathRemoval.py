import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
import os

def load_wav_file(file):
    sr, data = wavfile.read(file)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return sr, data

def remove_direct_path(direct_path_file, recording_file, output_file):
    sr_d, d = load_wav_file(direct_path_file)
    sr_y, y = load_wav_file(recording_file)
    
    if sr_d != sr_y:
        raise ValueError("Sample rates of the two files do not match.")
    
    if d.ndim != 2 or y.ndim != 2 or d.shape[1] != 2 or y.shape[1] != 2:
        raise ValueError("Both files must be stereo (2 channels).")
    
    y_reflections = np.zeros_like(y)
    
    for channel in range(2):
        y_channel = y[:, channel]
        d_channel = d[:, channel]
        
        
        # cross correlation to align signals
        corr = correlate(y_channel, d_channel, mode="full")
        lag = np.argmax(corr) - len(d_channel) + 1
        
        # align d and y
        d_aligned = np.zeros_like(y_channel)
        if lag >= 0:
            end_idx = min(len(d_channel), len(y_channel) - lag)
            d_aligned[lag:lag+end_idx] = d_channel[:end_idx]
        else:
            start_idx = -lag
            end_idx = min(len(d_channel) - start_idx, len(y_channel))
            d_aligned[0:end_idx] = d_channel[start_idx:start_idx+end_idx]
        
        y_reflections[:, channel] = y_channel - d_aligned
        
    # save the result
    wavfile.write(output_file, sr_y, (y_reflections * 32767).astype(np.int16))
    
if __name__ == "__main__":
    # usage 
    direct_path_file = "path_to_direct.wav"
    recording_file = "path_to_recording.wav"
    output_file = "output.wav"
    remove_direct_path(direct_path_file, recording_file, output_file)