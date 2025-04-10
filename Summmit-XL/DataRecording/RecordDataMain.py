import os
import time
import json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.io.wavfile import write
from LidarNode import LaserScanListener

# Constants
NUM_SWEEPS = 1
FILE_DIR = 'dataset/'
CHIRP_FILE = 'chirps/chirp_70Hz-20KHz_10ms.wav'
REC_DURATION = 0.5
SAMPLE_RATE = 44100
DATASET_SOUND_FILE = "{}_sound.wav"
DATASET_DISTANCE_FILE = "{}_distance_data.json"
DATASET_ANGLE_FILE = "{}_angle_data.json"
CONFIG_FILE = "{}_config.json"

# Initialize LaserScanListener
laser_scan_listener = LaserScanListener(FILE_DIR)


def generate_sweep_signal():
    """
    Generates a stereo sweep signal and performs recording.
    Returns the recorded stereo signal.
    """
    # Read the chirp/sweep signal from the specified file
    sweep_signal = wavfile.read(CHIRP_FILE)[1]

    # pad signal to add 1 second to before and after
    zero_padding = np.zeros(int(round(SAMPLE_RATE * REC_DURATION)))
    extended_signal = np.concatenate((zero_padding, sweep_signal, zero_padding))

    # Duplicate the channel for stereo playback
    stereo_sweep_signal = np.column_stack((extended_signal, extended_signal)).astype(np.float32)

    # Play the signal and record simultaneously
    recorded_signal = sd.playrec(stereo_sweep_signal, samplerate=SAMPLE_RATE, channels=2, dtype='float32')
    sd.wait()  # Wait until playback and recording are complete
    return recorded_signal


def create_batch_directory(base_dir, batch_id):
    """
    Creates and returns the path to the directory for a specific batch ID.
    Ensures the directory exists before returning the path.
    """
    batch_directory = os.path.join(base_dir, batch_id)
    os.makedirs(batch_directory)#, exist_ok=True)
    return batch_directory


def generate_batch():
    """
    Generates a batch of data by:
    1. Recording and saving stereo audio files.
    2. Fetching and saving LiDAR data.
    3. Writing a configuration file with metadata for the batch.
    """
    # Generate a unique batch ID based on the current timestamp and subdirectory count
    subdirectory_count = len([d for d in os.listdir(FILE_DIR) if os.path.isdir(os.path.join(FILE_DIR, d))]) + 1
    batch_id = f"{int(time.time())}_{subdirectory_count}"

    # --- STEP 1: Record stereo audio and save files ---
    # Record the stereo sweep signal
    recorded_signal = generate_sweep_signal()

    # Create a directory for this batch
    batch_directory = create_batch_directory(FILE_DIR, batch_id)

    # Define file paths within the batch directory
    file_path = os.path.join(batch_directory, DATASET_SOUND_FILE.format(batch_id))

    # Save the left and right audio channels as separate audio files
    write(file_path, SAMPLE_RATE, recorded_signal)

    # --- STEP 2: Fetch LiDAR data and calculate angles ---
    # Use the LaserScanListener to receive and process LiDAR data
    laser_distance = laser_scan_listener.receive_laser_scan()
    laser_angle = laser_scan_listener.angle_measurement(laser_distance)
    config_distance = {
	"LiDAR_distance": laser_distance,  # LiDAR distance data
    }
    config_angle = {
        "LiDAR_angle": laser_angle,  # LiDAR angle data
    }

    distance_file_path = os.path.join(batch_directory, DATASET_DISTANCE_FILE.format(batch_id))
    with open(distance_file_path, 'w') as config_file:
        json.dump(config_distance, config_file, indent=4)
    angle_file_path = os.path.join(batch_directory, DATASET_ANGLE_FILE.format(batch_id))
    with open(angle_file_path, 'w') as config_file:
        json.dump(config_angle, config_file, indent=4)

    # --- STEP 3: Create a configuration file ---
    # Prepare metadata with filenames and LiDAR data
    config_data = {
        "soundfile": file_path,  # Path to the audio file
        "LiDAR_angle": angle_file_path,  # LiDAR angle data path
        "LiDAR_distance": distance_file_path,  # LiDAR distance data path
    }

    # Write the metadata to a JSON configuration file in the batch directory
    config_file_path = os.path.join(batch_directory, CONFIG_FILE.format(batch_id))
    with open(config_file_path, 'w') as config_file:
        json.dump(config_data, config_file, indent=4)

    #print(f"Batch {batch_id} generated in directory: {batch_directory}")
    print("Batch done: {}".format(batch_id))


if __name__ == '__main__':
    generate_batch()
