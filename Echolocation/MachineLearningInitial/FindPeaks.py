import peakutils
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf

def normalize_intensity(audio_data):
    audio_data = np.abs(audio_data)
    return (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data))

def find_peaks(sound_file, threshold=0.05, min_distance=0.002, sample_rate=44100):
    left_channel = sound_file[:, 0]
    right_channel = sound_file[:, 1]
    
    distance_samples = int(min_distance * sample_rate) # convert to samples
    
    # find peaks in both channels
    left_peaks_indexes = peakutils.indexes(left_channel, thres=threshold, min_dist=distance_samples)
    right_peaks_indexes = peakutils.indexes(right_channel, thres=threshold, min_dist=distance_samples)
    
    return left_peaks_indexes, right_peaks_indexes
    
def plot_data(sound_file, sample_rate, left_peaks_indexes, right_peaks_indexes):
    time_axis = np.linspace(0, len(sound_file) / sample_rate, num=len(sound_file)) * 1000
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, sound_file[:, 0], label='Left Channel')
    plt.plot(time_axis[left_peaks_indexes], sound_file[left_peaks_indexes, 0], 'ro', label='Peaks')
    plt.title('Left Channel Signal with Peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, sound_file[:, 1], label='Right Channel')
    plt.plot(time_axis[right_peaks_indexes], sound_file[right_peaks_indexes, 1], 'ro', label='Peaks')
    plt.title('Right Channel Signal with Peaks')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
        
def cut_sound_file(sound_file_path, before_threshold, after_threshold):
    """
    Cuts the sound file based on the first peak found in either channel.
    Cuts it by before_threshold seconds before the first peak and after_threshold seconds after the first peak.
    """
    sound_file, sr = sf.read(sound_file_path)
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
    
def main():
    dataset_root_path = os.path.join("./Echolocation/Data", "dataset")
    chosen_dataset = None
    # checking if the dataset directory exists (whether we have any data)
    if not os.path.exists(dataset_root_path):
        print(f"Dataset directory {dataset_root_path} does not exist.")
        # if we dont have any data, pull it from somewhere (?)
        return
    else:
        # choosing a dataset to train on 
        print("Choose a dataset to train on:")
        for i, dataset in enumerate(os.listdir(dataset_root_path)):
            dataset_path = os.path.join(dataset_root_path, dataset)
            if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
                continue
            else:
                print(f"{dataset} [{i}]")
        choose_dataset_input = input("Enter the index of the dataset you want to train: ")
        chosen_dataset = os.listdir(dataset_root_path)[int(choose_dataset_input)]
        dataset_root_path = os.path.join(dataset_root_path, chosen_dataset)
        print(f"Selected dataset: {dataset_root_path}")
    
    chosen_index = input("Enter the index of the sound file you want to analyze: (1-1000) ")
    chosen_index = int(chosen_index) - 1  # Convert to zero-based index
    
    sound_file_path = os.path.join(dataset_root_path)
    sound_file_path = os.listdir(sound_file_path)[chosen_index]
    sound_file_path = os.path.join(dataset_root_path, sound_file_path, f"{sound_file_path}_sound.wav")
    
    sound_file, sr = sf.read(sound_file_path)
    
    normalized_sound_file = normalize_intensity(sound_file)
    
    left_peaks_indexes, right_peaks_indexes = find_peaks(normalized_sound_file, threshold=0.4, min_distance=0.002, sample_rate=sr)
    
    plot_data(normalized_sound_file, sr, left_peaks_indexes, right_peaks_indexes)
    
    cut_file = cut_sound_file(sound_file_path, before_threshold=0.005, after_threshold=0.3)
    
    new_left_peaks_indexes, new_right_peaks_indexes = find_peaks(normalize_intensity(cut_file), threshold=0.4, min_distance=0.002, sample_rate=sr)
    
    plot_data(normalize_intensity(cut_file), sr, new_left_peaks_indexes, new_right_peaks_indexes)

if __name__ == "__main__":
    main()