import os
import time
import json
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.io.wavfile import write

# from LidarNode import LaserScanListener

# Constants
NUM_SWEEPS = 1
FILE_DIR = 'datasets/'
CHIRP_FILE = 'chirps/Chirps{}/chirp_{}_{}ms.wav'
REC_DURATION = 0.5
SAMPLE_RATE = 96000
DATASET_SOUND_FILE = "{}_sound.wav"
DATASET_DISTANCE_FILE = "{}_distance_data.json"
DATASET_ANGLE_FILE = "{}_angle_data.json"
CONFIG_FILE = "{}_config.json"
WAIT = 5

CHIRPS_CONFIG = {
    "durations_ms": {
        "Short": 1,
        "MediumShort": 5,
        "Medium": 10,
        "MediumLong": 50,
        "Long": 100
    },
    "frequency_Hz": {
        "Low": "70Hz-1kHz",
        "Mid": "1kHz-10kHz",
        "High": "10kHz-20kHz",
        "Wide": "70Hz-20kHz"
    }

}


# Initialize LaserScanListener
# laser_scan_listener = LaserScanListener(FILE_DIR)


def generate_sweep_signal(freq_name, freq_range, duration_value):
    """
    Generates a stereo sweep signal and performs recording.
    Returns the recorded stereo signal.
    """
    # append freq and duration to chirp, depending on frequency range first append High, Mid, Low or Wide
    chirp = CHIRP_FILE.format(freq_name, freq_range, duration_value)

    # Read the chirp/sweep signal from the specified file
    sweep_signal = wavfile.read(chirp)[1]

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
    os.makedirs(batch_directory)  # , exist_ok=True)
    return batch_directory


def generate_batch():
    """
    Generates a batch of data by:
    1. Recording and saving stereo audio files.
    2. Fetching and saving LiDAR data.
    3. Writing a configuration file with metadata for the batch.
    """
    timestamp = int(time.time())
    completed_batches = []

    for duration_name, duration_value in CHIRPS_CONFIG["durations_ms"].items():
        for freq_name, freq_range in CHIRPS_CONFIG["frequency_Hz"].items():
            # wait
            print("Waiting for {} seconds...".format(WAIT))
            # time.sleep(WAIT)

            config_dir_name = f"{freq_name}_{duration_name}"
            config_dir_path = os.path.join(FILE_DIR, config_dir_name)
            os.makedirs(config_dir_path, exist_ok=True)

            # Generate a unique batch ID based on the current timestamp and subdirectory count
            subdirectory_count = len(
                [d for d in os.listdir(config_dir_path) if os.path.isdir(os.path.join(config_dir_path, d))]) + 1 \
                if os.path.exists(config_dir_path) else 1
            batch_id = f"{int(timestamp)}_{subdirectory_count}_{freq_name}_{duration_name}"

            print(f"Generating batch: {batch_id}")

            # --- STEP 1: Record stereo audio ---
            # Record the stereo sweep signal
            recorded_signal = generate_sweep_signal(freq_name, freq_range, duration_value)

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

            # Collect all batch data
            completed_batches.append({
                "batch_id": batch_id,
                "config_dir_path": config_dir_path,
                "recorded_signal": recorded_signal,
                "config_distance": config_distance,
                "config_angle": config_angle,
            })

    print("Done recording batches, saving data...")

    for batch in completed_batches:
        # Create a directory for this batch
        batch_directory = create_batch_directory(batch["config_dir_path"], batch["batch_id"])

        # Define file paths within the batch directory
        file_path = os.path.join(batch_directory, DATASET_SOUND_FILE.format(batch["batch_id"]))

        # Save the left and right audio channels as separate audio files
        write(file_path, SAMPLE_RATE, batch["recorded_signal"])

        distance_file_path = os.path.join(batch_directory, DATASET_DISTANCE_FILE.format(batch["batch_id"]))
        with open(distance_file_path, 'w') as config_file:
            json.dump(batch["config_distance"], config_file, indent=4)

        angle_file_path = os.path.join(batch_directory, DATASET_ANGLE_FILE.format(batch["batch_id"]))
        with open(angle_file_path, 'w') as config_file:
            json.dump(batch["config_angle"], config_file, indent=4)

        # --- STEP 3: Create a configuration file ---
        # Prepare metadata with filenames and LiDAR data
        config_data = {
            "soundfile": file_path,  # Path to the audio file
            "LiDAR_angle": angle_file_path,  # LiDAR angle data path
            "LiDAR_distance": distance_file_path,  # LiDAR distance data path
        }

        # Write the metadata to a JSON configuration file in the batch directory
        config_file_path = os.path.join(batch_directory, CONFIG_FILE.format(batch["batch_id"]))
        with open(config_file_path, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)

        print("Batch saved: {}".format(batch["batch_id"]))

    print("Entire batch saved: {}_{}".format(timestamp, subdirectory_count))


if __name__ == '__main__':
    generate_batch()
